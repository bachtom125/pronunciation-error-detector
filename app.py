
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from typing import List
import torch
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import re
import numpy as np
import cmudict
from io import BytesIO
import os
import logging
from joblib import Memory
from difflib import SequenceMatcher
import eng_to_ipa as ipa_conv
import os
import copy   
from IPython.display import HTML, display
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from pydub import AudioSegment
from Bio import pairwise2
from Bio.pairwise2 import format_alignment

# Set the Numba cache directory to a writable location
os.environ["NUMBA_CACHE_DIR"] = "/tmp"
import librosa

logging.basicConfig(level=logging.INFO)

cmu = cmudict.dict()

# Initialize FastAPI app
app = FastAPI()

# Load the processor and model
MODEL_NAME = "mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme" # wav2vec based phoneme trascriber trained on L2-ARTIC
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
model.eval()

# Check device availability
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

whisper_processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny.en")
whisper_model.eval()
whisper_model.to(device)

# =====================================
# Section: Utils
# =====================================

async def process_audio(audio, device, return_input_values=True):
    """
    Process an uploaded .m4a audio file and prepare input for the model.

    Args:
        audio: The uploaded audio file (e.g., from a web request).
        device: The device (e.g., 'cuda' or 'cpu') to move tensors to.
        return_input_values: Whether to return the processed input tensor.
    
    Returns:
        audio_input: NumPy array of the audio samples.
        input_values: Processed input tensor for the model.
    """
    # Read audio bytes
    audio_bytes = BytesIO(await audio.read())

    # Load the .m4a file using pydub
    audio_segment = AudioSegment.from_file(audio_bytes, format="m4a")

    # Convert the audio to a NumPy array
    audio_samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)

    # Normalize the audio to [-1.0, 1.0]
    max_val = np.iinfo(np.int16).max  # Maximum value for 16-bit PCM
    audio_samples /= max_val

    # If the audio has multiple channels, average them to mono
    if audio_segment.channels > 1:
        audio_samples = audio_samples.reshape(-1, audio_segment.channels).mean(axis=1)

    # Resample the audio to 16kHz using librosa
    audio_input = librosa.resample(audio_samples, orig_sr=audio_segment.frame_rate, target_sr=16000)
    if not return_input_values:
        return audio_input
    
    # Process the audio using the processor
    input_values = processor(audio_input, return_tensors="pt", sampling_rate=16000).input_values

    # Move the tensor to the specified device
    input_values = input_values.to(device)
    
    return audio_input, input_values

def transcribe_into_English(audio_input):
    # Load audio file
    audio_input = whisper_processor(audio_input, sampling_rate=16000, return_tensors="pt", language="en").to(device)

    # Perform transcription
    with torch.no_grad():
        generated_ids = whisper_model.generate(audio_input.input_features)

    # Decode the transcription
    transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription.lower().strip()

def get_nested_position(nested_list, flat_index):
    """
    Finds the nested list and the index within it for a given flat index.

    Args:
        nested_list (list of lists): The list of lists.
        flat_index (int): The flattened index.

    Returns:
        tuple: (nested_list_index, element_index_in_nested_list)
    """
    cumulative_index = 0

    for list_index, sublist in enumerate(nested_list):
        # Check if the flat index falls within the current sublist
        if cumulative_index + len(sublist) > flat_index:
            # Calculate the index within the sublist
            element_index = flat_index - cumulative_index
            return list_index, element_index
        # Update cumulative index
        cumulative_index += len(sublist)
   
    raise IndexError("Index out of range for the flattened list.")

def label_specific_elements_in_reference(reference, start_word_idx, start_element_idx, end_word_idx, end_element_idx, label):
    """
    Labels elements in a nested list between specified start and end indices (inclusive).

    Args:
        reference (list of lists): The original list of lists.
        start_word_idx (int): Index of the starting nested list.
        start_element_idx (int): Index of the starting element in the start list.
        end_word_idx (int): Index of the ending nested list.
        end_element_idx (int): Index of the ending element in the end list.
        label: The label to attach to the elements.

    Returns:
        list of lists: A new list of lists with labels attached where applicable.
    """
    labeled_reference = []
    for word_idx, sublist in enumerate(reference):
        labeled_sublist = []

        for element_idx, element in enumerate(sublist):
            if start_word_idx < end_word_idx:
                # Case 1: start_word_idx < end_word_idx
                if (
                    (word_idx > start_word_idx and word_idx < end_word_idx) or
                    (word_idx == start_word_idx and element_idx >= start_element_idx) or
                    (word_idx == end_word_idx and element_idx <= end_element_idx)
                ):
                    # Attach the label to elements within the inclusive range
                    if isinstance(element, tuple):
                        print(f"There is already a label at index ({word_idx}, {element_idx})") 
                    labeled_sublist.append((element, label))
                else:
                    # Keep elements outside the range unchanged
                    labeled_sublist.append(element)
            elif start_word_idx == end_word_idx:
                # Case 2: start_word_idx == end_word_idx
                if word_idx == start_word_idx and start_element_idx <= element_idx <= end_element_idx:
                    # Attach the label to elements within the inclusive range
                    if isinstance(element, tuple):
                        print(f"There is already a label at index ({word_idx}, {element_idx})") 
                    labeled_sublist.append((element, label))
                else:
                    # Keep elements outside the range unchanged
                    labeled_sublist.append(element)

        labeled_reference.append(labeled_sublist)
    
    return labeled_reference

def clean_text(text: str) -> str:
    """
    Remove punctuation from the input string except for special characters 
    that are part of a word, such as ' in I'm or - in hard-working.

    Parameters:
        text (str): Input string to clean.
        
    Returns:
        str: Cleaned string with allowed special characters retained.
    """
    # Allow letters, spaces, apostrophes, and hyphens within words
    cleaned_text = re.sub(r'[^\w\s\'-]', '', text)  # Remove punctuation except ' and -
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Normalize spaces
    return cleaned_text.lower().strip()

# =====================================
# Section: IPA Phonemes Utils
# =====================================


# WORKING: converting functions to class, currently done with the last function in the class
import re
from difflib import SequenceMatcher
from IPython.display import HTML, display
import cmudict
import copy   
from IPython.display import HTML, display
from Bio import pairwise2
from Bio.pairwise2 import format_alignment

class PronunciationAssessment:
    def __init__(self, transcript, uttered_phonemes):
        # NOTE: removed all long signals ('ː') for compatibility with L2-artic's phoneme set (ssl model training set). American English. 
        # ground truth phonemes are converted into arpabet first, and then into ipa using the arpabet_to_ipa dict, meaning the arpabet_to_ipa dict contains
        # the core ipa phoeneme set

        # NOTE: modifications to the list in https://www.dyslexia-reading-well.com/44-phonemes-in-english.html: 
        # removed 'sc', 'ps', and 'st', in ipa_to_orthography of 's', because I want to assume it's silient 
        self.ipa_to_orthography = {
            'b': ['b', 'bb'],  # Examples: bug, bubble
            'd': ['d', 'dd', 'ed'],  # Examples: dad, add, milled
            'f': ['f', 'ff', 'ph', 'gh', 'lf', 'ft'],  # Examples: fat, cliff, phone, enough, half, often
            'g': ['g', 'gg', 'gh', 'gu', 'gue'],  # Examples: gun, egg, ghost, guest, prologue
            'h': ['h', 'wh'],  # Examples: hop, who
            'dʒ': ['j', 'ge', 'g', 'dge', 'di', 'gg'],  # Examples: jam, wage, giraffe, edge, soldier, exaggerate
            'k': ['k', 'c', 'ch', 'cc', 'lk', 'qu', 'q(u)', 'ck', 'x'],  # Examples: kit, cat, chris, accent, folk, bouquet, queen, rack, box
            'l': ['l', 'll'],  # Examples: live, well
            'm': ['m', 'mm', 'mb', 'mn', 'lm'],  # Examples: man, summer, comb, column, palm
            'n': ['n', 'nn', 'kn', 'gn', 'pn', 'mn'],  # Examples: net, funny, know, gnat, pneumonic, mnemonic
            'p': ['p', 'pp'],  # Examples: pin, dippy
            'r': ['r', 'rr', 'wr', 'rh'],  # Examples: run, carrot, wrench, rhyme
            'ɹ': ['r', 'rr', 'wr', 'rh'],  # Examples: run, carrot, wrench, rhyme
            's': ['s', 'ss', 'c', 'ce', 'se'],  # Examples: sit, less, circle, scene, psycho, listen, pace, course
            't': ['t', 'tt', 'th', 'ed'],  # Examples: tip, matter, thomas, ripped
            'v': ['v', 'f', 'ph', 've'],  # Examples: vine, of, stephen, five
            'w': ['w', 'wh', 'u', 'o'],  # Examples: wit, why, quick, choir
            'z': ['z', 'zz', 's', 'ss', 'x', 'ze', 'se'],  # Examples: zed, buzz, his, scissors, xylophone, craze
            'ʒ': ['s', 'si', 'z'],  # Examples: treasure, division, azure
            'tʃ': ['ch', 'tch', 'tu', 'te'],  # Examples: chip, watch, future, righteous
            'ʃ': ['sh', 'ce', 's', 'ci', 'si', 'ch', 'sci', 'ti'],  # Examples: sham, ocean, sure, special, pension, machine, conscience, station
            'θ': ['th'],  # Example: thongs
            'ð': ['th'],  # Example: leather
            'ŋ': ['ng', 'n', 'ngue'],  # Examples: ring, pink, tongue
            'j': ['y', 'i', 'j'],  # Examples: you, onion, hallelujah
            'æ': ['a', 'ai', 'au'],  # Examples: cat, plaid, laugh
            'eɪ': ['a', 'ai', 'eigh', 'aigh', 'ay', 'er', 'et', 'ei', 'au', 'a_e', 'ea', 'ey'],  # Examples: bay, maid, weigh, straight, pay, foyer, filet, eight, gauge, mate, break, they
            'ɛ': ['e', 'ea', 'u', 'ie', 'ai', 'a', 'eo', 'ei', 'ae'],  # Examples: end, bread, bury, friend, said, many, leopard, heifer, aesthetic
            'i': ['e', 'ee', 'ea', 'y', 'ey', 'oe', 'ie', 'i', 'ei', 'eo', 'ay'],  # Examples: be, bee, meat, lady, key, phoenix, grief, ski, deceive, people, quay
            'ɪ': ['i', 'e', 'o', 'u', 'ui', 'y', 'ie'],  # Examples: it, england, women, busy, guild, gym, sieve
            'aɪ': ['i', 'y', 'igh', 'ie', 'uy', 'ye', 'ai', 'is', 'eigh', 'i_e'],  # Examples: spider, sky, night, pie, guy, stye, aisle, island, height, kite
            'ɒ': ['a', 'ho', 'au', 'aw', 'ough'],  # Examples: swan, honest, maul, slaw, fought
            'oʊ': ['o', 'oa', 'o_e', 'oe', 'ow', 'ough', 'eau', 'oo', 'ew'],  # Examples: open, moat, bone, toe, sow, dough, beau, brooch, sew
            'ʊ': ['o', 'oo', 'u', 'ou'],  # Examples: wolf, look, bush, would
            'ʌ': ['u', 'o', 'oo', 'ou'],  # Examples: lug, monkey, blood, double
            'u': ['o', 'oo', 'ew', 'ue', 'u_e', 'oe', 'ough', 'ui', 'oew', 'ou'],  # Examples: who, loon, dew, blue, flute, shoe, through, fruit, manoeuvre, group
            'ɔɪ': ['oi', 'oy', 'uoy'],  # Examples: join, boy, buoy
            'aʊ': ['ow', 'ou', 'ough'],  # Examples: now, shout, bough
            'ə': ['o', 'a', 'er', 'i', 'ar', 'our', 'ur', 'e'],  # Examples: about, ladder, pencil, dollar, honour, augur
            'eəʳ': ['air', 'are', 'ear', 'ere', 'eir', 'ayer'],  # Examples: chair, dare, pear, where, their, prayer
            'a': ['a'],  # Example: arm
            'ɜʳ': ['ir', 'er', 'ur', 'ear', 'or', 'our', 'yr'],  # Examples: bird, term, burn, pearl, word, journey, myrtle
            'ɔ': ['aw', 'a', 'au', 'or', 'ore', 'oar', 'our', 'augh', 'ar', 'ough'],  # Examples: law, ball, haul,
            'ɪəʳ': ['ear', 'eer', 'ere', 'ier'], # Examples: beer, fear, here, tier
            'ʊəʳ': ['ure', 'our'], # Examples: sure, tour

            # Dialectal Variations
            'ɚ': ['er', 'ir', 'ur', 'ar', 'or'],  # Examples: butter, bird, dollar
            'ɝ': ['er', 'ir', 'ur'],  # Examples: herd, third, turn
            'ʍ': ['wh'],  # Examples: where, which, whale
            'ɑ': ['a', 'ah'],  # Examples: father, spa
            'oʊ': ['o', 'ow', 'oe', 'ough', 'ew']  # Examples: go, snow, foe, though, sew
        }

        self.arpabet_to_ipa = {
            "AA": "a",    # odd
            "AE": "æ",    # at
            # "AH": "ə",    # hut
            "AO": "ɔ",    # ought
            "AW": "aʊ",   # cow 
            "AX": "ə",    # discus
            "AY": "aɪ",   # hide
            "B": "b",     # be
            "CH": "tʃ",   # cheese
            "D": "d",     # dee
            "DH": "ð",    # thee
            "EH": "ɛ",    # Ed
            # "ER": "ɝ",    # hurt
            "EY": "eɪ",   # ate
            "F": "f",     # fee
            "G": "g",     # green
            "HH": "h",    # he
            "IH": "ɪ",    # it
            "IY": "i",    # eat
            "JH": "dʒ",   # gee
            "K": "k",     # key
            "L": "l",     # lee
            "M": "m",     # me
            "N": "n",     # knee
            "NG": "ŋ",    # ping
            "OW": "oʊ",   # oat
            "OY": "ɔɪ",   # toy
            "P": "p",     # pee
            "R": "ɹ",     # read
            "S": "s",     # sea
            "SH": "ʃ",    # she
            "T": "t",     # tea
            "TH": "θ",    # theta
            "UH": "ʊ",    # hood
            "UW": "u",    # two
            "V": "v",     # vee
            "W": "w",     # we
            "Y": "j",     # yield
            "Z": "z",     # zee
            "ZH": "ʒ",     # seizure

            # Vowels with stress affecting IPA
            "AH0": "ə",    # unstressed (about)
            "AH1": "ʌ",    # stressed (hut)
            "AH2": "ʌ",    # secondary stress (hut)
            "ER0": "ɚ",    # unstressed (runner)
            "ER1": "ɝ",    # stressed (bird)
            "ER2": "ɝ",    # secondary stress (bird)
            "EY0": "e",    # unstressed (obey)
            "EY1": "eɪ",   # stressed (day)
            "EY2": "eɪ",   # secondary stress (day)
            "IH0": "ɨ",    # unstressed (possible centralization)
            "IH1": "ɪ",    # stressed (bit)
            "IH2": "ɪ",    # secondary stress (bit)
            "UW0": "ʉ",    # unstressed (possible centralization)
            "UW1": "u",    # stressed (food)
            "UW2": "u",    # secondary stress (food)
            "AO0": "ə",    # unstressed (centralized in some accents)
            "AO1": "ɔ",    # stressed (thought)
            "AO2": "ɔ",    # secondary stress (thought)
            "AE0": "ə",    # unstressed (centralized in some accents)
            "AE1": "æ",    # stressed (cat)
            "AE2": "æ",    # secondary stress (cat)
            "OW0": "o",    # unstressed (less diphthongized)
            "OW1": "oʊ",   # stressed (go)
            "OW2": "oʊ",   # secondary stress (go)
            "UH0": "ɨ",    # unstressed (centralized or reduced)
            "UH1": "ʊ",    # stressed (put)
            "UH2": "ʊ",    # secondary stress (put)

            # unknown phoneme
            "unk": "unk"
        }

        # whether the two phonemes are considered correct (value = 1), acceptable (value = 2), or wrong (value = 0)
        self.phoneme_pair_label = {
            # Completely correct pairs (self-similarity)
            **{(p, p): 1 for p in [
                'b', 'd', 'f', 'g', 'h', 'dʒ', 'k', 'l', 'm', 'n', 'p', 'r', 'ɹ', 's', 't', 'v', 'w', 'z', 'ʒ', 'tʃ',
                'ʃ', 'θ', 'ð', 'ŋ', 'j', 'æ', 'eɪ', 'ɛ', 'i', 'ɪ', 'aɪ', 'ɒ', 'oʊ', 'ʊ', 'ʌ', 'u', 'ɔɪ', 'aʊ', 'ə',
                'eəʳ', 'a', 'ɜʳ', 'ɔ', 'ɪəʳ', 'ʊəʳ', 'ɚ', 'ɝ', 'ʍ', 'ɑ'
            ]},

            # Acceptable substitutions (value = 2)
            **{pair: 2 for pair in [
                ('b', 'p'), ('d', 't'), ('g', 'k'), ('v', 'f'), ('z', 's'), ('ʒ', 'ʃ'), ('ð', 'θ'),
                ('m', 'n'), ('m', 'ŋ'), ('n', 'ŋ'), ('r', 'ɹ'), ('l', 'r'), ('l', 'ɹ'), ('w', 'ʍ'),
                ('j', 'ɹ'), ('f', 'θ'), ('v', 'ð'), ('s', 'ʃ'), ('z', 'ʒ'), ('tʃ', 'dʒ'), ('tʃ', 'ʃ'),
                ('dʒ', 'ʒ'), ('i', 'ɪ'), ('ɪ', 'ɛ'), ('ɛ', 'æ'), ('ə', 'ʌ'), ('ə', 'ɜʳ'), ('ʌ', 'ɜʳ'),
                ('ə', 'ɚ'), ('u', 'ʊ'), ('ʊ', 'oʊ'), ('oʊ', 'ɔ'), ('ɔ', 'ɒ'), ('ɑ', 'ɒ'), ('eɪ', 'ɛ'),
                ('eɪ', 'æ'), ('aɪ', 'ɪ'), ('aʊ', 'ʊ'), ('ɔɪ', 'ɔ'), ('ɝ', 'ɚ'), ('ɪəʳ', 'ɜʳ'), ('ʊəʳ', 'ɔ'),
                ('ð', 'd'), ('ɑ', 'a')
            ] + [(b, a) for (a, b) in [
                ('b', 'p'), ('d', 't'), ('g', 'k'), ('v', 'f'), ('z', 's'), ('ʒ', 'ʃ'), ('ð', 'θ'),
                ('m', 'n'), ('m', 'ŋ'), ('n', 'ŋ'), ('r', 'ɹ'), ('l', 'r'), ('l', 'ɹ'), ('w', 'ʍ'),
                ('j', 'ɹ'), ('f', 'θ'), ('v', 'ð'), ('s', 'ʃ'), ('z', 'ʒ'), ('tʃ', 'dʒ'), ('tʃ', 'ʃ'),
                ('dʒ', 'ʒ'), ('i', 'ɪ'), ('ɪ', 'ɛ'), ('ɛ', 'æ'), ('ə', 'ʌ'), ('ə', 'ɜʳ'), ('ʌ', 'ɜʳ'),
                ('ə', 'ɚ'), ('u', 'ʊ'), ('ʊ', 'oʊ'), ('oʊ', 'ɔ'), ('ɔ', 'ɒ'), ('ɑ', 'ɒ'), ('eɪ', 'ɛ'),
                ('eɪ', 'æ'), ('aɪ', 'ɪ'), ('aʊ', 'ʊ'), ('ɔɪ', 'ɔ'), ('ɝ', 'ɚ'), ('ɪəʳ', 'ɜʳ'), ('ʊəʳ', 'ɔ'),
                ('ð', 'd'), ('ɑ', 'a')
            ] if (b, a) not in [(a, b)]]},

            # Completely wrong pairs (default value = 0)
            **{(p1, p2): 0 for p1 in [
                'b', 'd', 'f', 'g', 'h', 'dʒ', 'k', 'l', 'm', 'n', 'p', 'r', 'ɹ', 's', 't', 'v', 'w', 'z', 'ʒ', 'tʃ',
                'ʃ', 'θ', 'ð', 'ŋ', 'j', 'æ', 'eɪ', 'ɛ', 'i', 'ɪ', 'aɪ', 'ɒ', 'oʊ', 'ʊ', 'ʌ', 'u', 'ɔɪ', 'aʊ', 'ə',
                'eəʳ', 'a', 'ɜʳ', 'ɔ', 'ɪəʳ', 'ʊəʳ', 'ɚ', 'ɝ', 'ʍ', 'ɑ'
            ] for p2 in [
                'b', 'd', 'f', 'g', 'h', 'dʒ', 'k', 'l', 'm', 'n', 'p', 'r', 'ɹ', 's', 't', 'v', 'w', 'z', 'ʒ', 'tʃ',
                'ʃ', 'θ', 'ð', 'ŋ', 'j', 'æ', 'eɪ', 'ɛ', 'i', 'ɪ', 'aɪ', 'ɒ', 'oʊ', 'ʊ', 'ʌ', 'u', 'ɔɪ', 'aʊ', 'ə',
                'eəʳ', 'a', 'ɜʳ', 'ɔ', 'ɪəʳ', 'ʊəʳ', 'ɚ', 'ɝ', 'ʍ', 'ɑ'
            ] if p1 != p2 and (p1, p2) not in [
                ('b', 'p'), ('d', 't'), ('g', 'k'), ('v', 'f'), ('z', 's'), ('ʒ', 'ʃ'), ('ð', 'θ'),
                ('m', 'n'), ('m', 'ŋ'), ('n', 'ŋ'), ('r', 'ɹ'), ('l', 'r'), ('l', 'ɹ'), ('w', 'ʍ'),
                ('j', 'ɹ'), ('f', 'θ'), ('v', 'ð'), ('s', 'ʃ'), ('z', 'ʒ'), ('tʃ', 'dʒ'), ('tʃ', 'ʃ'),
                ('dʒ', 'ʒ'), ('i', 'ɪ'), ('ɪ', 'ɛ'), ('ɛ', 'æ'), ('ə', 'ʌ'), ('ə', 'ɜʳ'), ('ʌ', 'ɜʳ'),
                ('ə', 'ɚ'), ('u', 'ʊ'), ('ʊ', 'oʊ'), ('oʊ', 'ɔ'), ('ɔ', 'ɒ'), ('ɑ', 'ɒ'), ('eɪ', 'ɛ'),
                ('eɪ', 'æ'), ('aɪ', 'ɪ'), ('aʊ', 'ʊ'), ('ɔɪ', 'ɔ'), ('ɝ', 'ɚ'), ('ɪəʳ', 'ɜʳ'), ('ʊəʳ', 'ɔ'),
                ('ð', 'd'), ('ɑ', 'a')
            ] + [(b, a) for (a, b) in [
                ('b', 'p'), ('d', 't'), ('g', 'k'), ('v', 'f'), ('z', 's'), ('ʒ', 'ʃ'), ('ð', 'θ'),
                ('m', 'n'), ('m', 'ŋ'), ('n', 'ŋ'), ('r', 'ɹ'), ('l', 'r'), ('l', 'ɹ'), ('w', 'ʍ'),
                ('j', 'ɹ'), ('f', 'θ'), ('v', 'ð'), ('s', 'ʃ'), ('z', 'ʒ'), ('tʃ', 'dʒ'), ('tʃ', 'ʃ'),
                ('dʒ', 'ʒ'), ('i', 'ɪ'), ('ɪ', 'ɛ'), ('ɛ', 'æ'), ('ə', 'ʌ'), ('ə', 'ɜʳ'), ('ʌ', 'ɜʳ'),
                ('ə', 'ɚ'), ('u', 'ʊ'), ('ʊ', 'oʊ'), ('oʊ', 'ɔ'), ('ɔ', 'ɒ'), ('ɑ', 'ɒ'), ('eɪ', 'ɛ'),
                ('eɪ', 'æ'), ('aɪ', 'ɪ'), ('aʊ', 'ʊ'), ('ɔɪ', 'ɔ'), ('ɝ', 'ɚ'), ('ɪəʳ', 'ɜʳ'), ('ʊəʳ', 'ɔ'),
                ('ð', 'd'), ('ɑ', 'a')
            ] if (b, a) not in [(a, b)]]}
        }

        self.ipa_phonemes = list(self.ipa_to_orthography.keys())
        self.ipa_phonemes.append('unk')
        self.cmu_dict = cmudict.dict()

        # instance-specific variables
        self.transcript = transcript.lower().strip()
        self.uttered_ipa_phonemes = uttered_phonemes
        self.ground_truth_arpabet_phonemes = ""
        self.ground_truth_ipa_phonemes = ""

        self.segmented_uttered_ipa_phonemes = []
        self.segmented_ground_truth_arpabet_phonemes = []
        self.segmented_ground_truth_ipa_phonemes = []

    def get_phoneme_count(self):
        return len(self.ipa_phonemes)

    def has_phoneme(self, phoneme): 
        return phoneme in self.ipa_phonemes

    def convert_transcript_into_phonemes(self, get_all_versions=True):
        """
        Parameters:
            get_all_versions (bool): Default to True. Whether to return all possible phoneme versions for each word.
        Convert a list of word into IPA phonems through ARPABET phonemes.

        Returns:    
            bool: If the conversion is successful.
        """
        if len(self.transcript) == 0:   
            return False
        
        arap_phonemes = []
        for word in self.transcript.split():
            if len(self.cmu_dict[word]) != 0:
                if not get_all_versions:
                    arpa_phons = self.clean_single_arpabet_phoneme_list(self.cmu_dict[word][0])
                else:
                    phon_vers = self.cmu_dict[word]
                    arpa_phons = [self.clean_single_arpabet_phoneme_list(phons) for phons in phon_vers]
                arap_phonemes.append(arpa_phons)  # Use the first phoneme representation
            else:
                arap_phonemes.append([['unk']])  # Append 'UNK' for unknown words\

        self.segmented_ground_truth_arpabet_phonemes = arap_phonemes
        if not get_all_versions:
            ipa_phonemes = []
            for word in arap_phonemes:
                cur_phonemes = []
                for phon in word:
                    cur_phonemes.append(self.arpabet_to_ipa[phon])
                ipa_phonemes.append(cur_phonemes)
        else: 
            ipa_phonemes = []
            for word in arap_phonemes:
                cur_word = []
                for ver in word:
                    cur_ver = []
                    for phon in ver:
                        cur_ver.append(self.arpabet_to_ipa[phon])
                    cur_word.append(cur_ver)
                ipa_phonemes.append(cur_word)

        self.segmented_ground_truth_ipa_phonemes = ipa_phonemes
        return True
        
    def remove_ipa_stress_markers(self, phonemes):
        """
        Parameters:
            phonemes (str): A string of phonemes (e.g. "ˈɪŋɡlɪʃ")
        """
        return re.sub(r"[ˈˌ]", "", phonemes)
    
    def remove_ipa_length_markers(self, phonemes):
        """
        Parameters:
            phonemes (str): A string of phonemes (e.g. "ˈɪŋɡlɪʃ")
        """
        return re.sub(r"[ːˑ]", "", phonemes)
    
    def remove_ipa_break_markers(self, phonemes):
        """
        Parameters:
            phonemes (str): A string of phonemes (e.g. "ˈɪŋɡlɪʃ")
        """
        return re.sub(r"[.‖|]", "", phonemes)
    
    def remove_ipa_tone_markers(self, phonemes):
        """
        Parameters:
            phonemes (str): A string of phonemes (e.g. "ˈɪŋɡlɪʃ")
        """
        return re.sub(r"[˥˦˧˨˩]", "", phonemes)
    
    def remove_ipa_global_markers(self, phonemes):
        """
        Parameters:
            phonemes (str): A string of phonemes (e.g. "ˈɪŋɡlɪʃ")
        """
        return re.sub(r"[↗↘]", "", phonemes)
    
    def remove_ipa_diacritics(self, phonemes):
        """
        Parameters:
            phonemes (str): A string of phonemes (e.g. "ˈɪŋɡlɪʃ")
        """
        return re.sub(r"[̩̯̪̠̟̹̜̬̥̤̰̼̩̯̝̞̊̃̚]", "", phonemes)
    
    def remove_tie_bars(self, phonemes):
        """
        Removes all tie bars (͡) from a string of phonemes.

        Parameters:
            phonemes (str): A string of phonemes (e.g. "ˈɪŋɡlɪʃ")
        """
        return phonemes.replace('͡', '')
    
    def correct_shenanigans(self, ipa_phonemes):
        """
        Manually correct phoneme-related problems, mostly arising from converstion from arpabet to ipa or from ssl's inferred ipa
        Parameters:
            ipa_phonemes (list of lists): Nested list of phonemes.
        """
        new_ipa_phonemes = ""
        for word in ipa_phonemes.split():
            if len(new_ipa_phonemes) > 0:
                new_ipa_phonemes += " " 
            cur_word = ""
            for i, phoneme in enumerate(list(word)):
                if phoneme == "ʌ":
                    if i == 0 or i == len(word) - 1 or len(word) > 4:
                        cur_word += "ə"
                    else:
                        cur_word += phoneme
                else:
                    cur_word += phoneme
            new_ipa_phonemes += cur_word
        return new_ipa_phonemes
    
    def clean_ipa_phonemes(self):
        """
        Clean uttered phonemes by removing stress, length, break, tone, global, and diacritic markers, as well as tie bars.
        """
        phonemes = self.uttered_ipa_phonemes
        phonemes = self.remove_ipa_stress_markers(phonemes)
        phonemes = self.remove_ipa_length_markers(phonemes)
        phonemes = self.remove_ipa_break_markers(phonemes)
        phonemes = self.remove_ipa_tone_markers(phonemes)
        phonemes = self.remove_ipa_global_markers(phonemes)
        phonemes = self.remove_ipa_diacritics(phonemes)
        phonemes = self.remove_tie_bars(phonemes)
        phonemes = self.correct_shenanigans(phonemes)
        self.uttered_ipa_phonemes = phonemes
        
        return True
    
    def remove_stress_indicator_from_arpabet_phonemes(self, arpabet_phoneme_list):
        """
        Remove all stress markers (trailing numbers), excluding AH and ER (due to their nuances, refer to the arpa_to_ipa dict for detail)

        Parameters:
        arpabet_phoneme_list (list of lists): Nested list of phonemes.

        Returns:
            list of lists: Updated nested list with numbers removed from phonemes.
        """
        cleaned_phon_list = []
        for word_phonemes in arpabet_phoneme_list:
            cleaned_phon_list = []
            for phoneme in word_phonemes:
                if not phoneme.startswith(('AH', 'ER')):
                    cleaned_phon_list.append(re.sub(r'\d', '', phoneme))
                else:
                    cleaned_phon_list.append(phoneme)
            cleaned_phon_list.append(cleaned_phon_list)

        return cleaned_phon_list
    
    def remove_stress_indicator_from_single_arpabet_phoneme_list(self, phon_list):
        """
        Remove all stress markers (trailing numbers), excluding AH and ER (due to their nuances, refer to the arpa_to_ipa dict for detail)

        Parameters:
        phon_list (list(str)): The list of arpabet phoneme

        Returns:
            str: Updated phoneme with numbers removed.
        """
        cleaned_phon_list = []
        for phoneme in phon_list:
            if not phoneme.startswith(('AH', 'ER')):
                cleaned_phon_list.append(re.sub(r'\d', '', phoneme))
            else:
                cleaned_phon_list.append(phoneme)

        return cleaned_phon_list
            
    def clean_arpabet_phonemes(self, arpabet_phoneme_list):
        """
        Parameters:
            arpabet_phoneme_list (list of lists): Nested list of phonemes.
        """
        cleaned_phonemes = self.remove_stress_indicator_from_arpabet_phonemes(arpabet_phoneme_list)
        return cleaned_phonemes
    
    def clean_single_arpabet_phoneme_list(self, phon_list):
        """
        Parameters:
            phon_list (list(str)): The list of arpabet phoneme
        """
        cleaned_phon = self.remove_stress_indicator_from_single_arpabet_phoneme_list(phon_list)
        return cleaned_phon
    
    def split_phoneme_sequence(self):
        """
        Splits a the uttered phoneme sequence (of a string of phoneme with each word separated by a space) into individual phonemes based on the IPA dictionary keys.
        """
        sequence = self.uttered_ipa_phonemes.strip()
        i = 0
        keys = sorted(self.ipa_phonemes, key=len, reverse=True)  # Prioritize longer matches
        
        sequence_phonemes = []
        word_phonemes = []
        while i < len(sequence):
            # if reaches the end of a word
            if sequence[i] == ' ':
                if word_phonemes:
                    sequence_phonemes.append(word_phonemes)
                    word_phonemes = []
                i += 1
                continue
            match = None

            # otherwise
            for key in keys:
                if sequence[i:i+len(key)] == key:
                    match = key
                    word_phonemes.append(match)
                    i += len(key)
                    break
            if not match:  # No phoneme matched
                word_phonemes.append('unk')
                i += 1

        if word_phonemes:
            sequence_phonemes.append(word_phonemes)
        self.segmented_uttered_ipa_phonemes = sequence_phonemes
    
    def evaluate_pronunciation(self, reference: list, pronunciation: list):
        """
        Evaluate the pronunciation of a word or sentence by comparing it to a reference.
        
        Args:
            reference (list(list(str))): List of words, each word is a list of phonemes representing the correct pronunciation.
            pronunciation (list(list(str))): List of words, each word is a list of phonemes representing the pronunciation to be evaluated.

        Returns:
            list(dict): A list of dictionaries (one for each word) containing the evaluation results.
        """
        smushed_ref = []
        smushed_pron = []

        smushed_ref = [item for word in reference for item in word]
        smushed_pron = [item for word in pronunciation for item in word]

        matcher = SequenceMatcher(None, smushed_ref, smushed_pron)
        alignment = matcher.get_opcodes()
        
        # Initialize results for errors and labels
        errors = {"matches": [], "substitutions": [], "insertions": [], "deletions": []}
        labels = copy.deepcopy(reference)
        processed_indices = set()  # Track indices in the reference that are processed
        
        # Process each alignment operation
        for tag, i1, i2, j1, j2 in alignment:
            if tag == "equal":
                # Matches: Add to errors and label as 1
                errors["matches"].extend(smushed_ref[i1:i2])
                start_word_idx, start_element_idx = get_nested_position(reference, i1)
                end_word_idx, end_element_idx = get_nested_position(reference, i2 - 1)

                labels = label_specific_elements_in_reference(labels, start_word_idx, start_element_idx, end_word_idx, end_element_idx, 1)
                # labels.extend([(phoneme, 1) for phoneme in reference[i1:i2]])
                processed_indices.update(range(i1, i2))
            elif tag == "replace":
                # Substitutions: Check phoneme-by-phoneme
                ref_segment = smushed_ref[i1:i2]
                pron_segment = smushed_pron[j1:j2]
                # go through each pair of phoneme in ref and pron segment, if they are labeled 2 or 1 in the phoneme_pair_label, remove them as mistakes
                original_i1 = i1
                original_i2 = i2
                for ref_phoneme, pron_phoneme in zip(ref_segment, pron_segment):
                    if (ref_phoneme, pron_phoneme) in self.phoneme_pair_label:
                        if self.phoneme_pair_label[(ref_phoneme, pron_phoneme)] in [1, 2]:
                            processed_indices.add(i1)
                            i1 += 1  # Move to the next index in the reference
                            j1 += 1  # Move to the next index in the pronunciation
                    
                if i1 > original_i1:
                    start_word_idx, start_element_idx = get_nested_position(reference, original_i1)
                    end_word_idx, end_element_idx = get_nested_position(reference, i1 - 1)
                    labels = label_specific_elements_in_reference(labels, start_word_idx, start_element_idx, end_word_idx, end_element_idx, 1)
                
                if i1 >= original_i2: # if no more phoneme in reference left to process
                    continue

                start_word_idx, start_element_idx = get_nested_position(reference, i1)
                end_word_idx, end_element_idx = get_nested_position(reference, i2 - 1)

                labels = label_specific_elements_in_reference(labels, start_word_idx, start_element_idx, end_word_idx, end_element_idx, 0)
                processed_indices.update(range(i1, i2))        

                for ref_phoneme, pron_phoneme in zip(ref_segment, pron_segment):
                    if ref_phoneme != pron_phoneme:
                        errors["substitutions"].append((ref_phoneme, pron_phoneme))
                        # labels.append((ref_phoneme, 0))
                        processed_indices.add(i1)
                        i1 += 1  # Move to the next index in the reference
                
                # Handle leftover phonemes in reference (deletions)
                if len(ref_segment) > len(pron_segment):
                    for leftover in ref_segment[len(pron_segment):]:
                        errors["deletions"].append(leftover)
                        # labels.append((leftover, 0))
                        processed_indices.add(i1)
                        i1 += 1
                
                # Handle leftover phonemes in pronunciation (insertions)
                if len(pron_segment) > len(ref_segment):
                    for leftover in pron_segment[len(ref_segment):]:
                        errors["insertions"].append(leftover)
            elif tag == "insert":
                # Insertions: Add to errors, no effect on reference labels
                errors["insertions"].extend(smushed_pron[j1:j2])
            elif tag == "delete":
                # Deletions: Add to errors and label as 0
                errors["deletions"].extend(smushed_ref[i1:i2])
                start_word_idx, start_element_idx = get_nested_position(reference, i1)
                end_word_idx, end_element_idx = get_nested_position(reference, i2 - 1)

                labels = label_specific_elements_in_reference(labels, start_word_idx, start_element_idx, end_word_idx, end_element_idx, 0)
                # labels.extend([(phoneme, 0) for phoneme in reference[i1:i2]])
                processed_indices.update(range(i1, i2))
                
        # Post-check: Ensure all phonemes in the reference are processed
        for i, phoneme in enumerate(smushed_ref):
            if i not in processed_indices:
                errors["deletions"].append(phoneme)
                start_word_idx, start_element_idx = get_nested_position(reference, i)
                end_word_idx, end_element_idx = get_nested_position(reference, i)

                labels = label_specific_elements_in_reference(labels, start_word_idx, start_element_idx, end_word_idx, end_element_idx, 0)
                # labels.append((phoneme, 0))
        
        return errors, labels
    
    def map_boundary(self, segmented_ground_truth_list, segmented_uttered_list):
        """
        Maps the boundaries of each word in the ground truth to the corresponding part in the uttered list.
        Rewrites to self.segmented_uttered_ipa_phonemes
        Args:
            segmented_ground_truth_list (list): A single list of phonemes, word are separed with space 
            segmented_uttered_list (list): A single list of phonemes, word are separed with space
        """
        
        alignments = pairwise2.align.globalms(
            segmented_ground_truth_list, segmented_uttered_list, 
            match=1,  # Score for match
            mismatch=-1,  # Penalty for mismatch
            open=-2,  # Penalty for opening a gap
            extend=-1,  # Penalty for extending a gap,
            gap_char=['-']
        )
        best_alignment = alignments[0]

        # Extract the aligned sequences
        aligned_ground_truth = best_alignment.seqA
        aligned_uttered = best_alignment.seqB

        # Process the alignment to group corresponding characters
        segments = []
        current_segment = []
        for g_char, u_char in zip(aligned_ground_truth, aligned_uttered):
            if g_char == " ":  # Word boundary in ground truth
                if current_segment:  # Append collected segment
                    segments.append(current_segment)

                    current_segment = []
            else:
                if g_char != "-" and g_char != " ":  # Only consider characters from uttered list
                    current_segment.append(u_char)

        # Append the last segment, if any
        if current_segment:
            segments.append(current_segment)

        print(segments)
        # Output the segmented uttered list
        self.segmented_uttered_ipa_phonemes = segments
        
    def evaluate_full_pronunciation(self):
        """
        Evaluates the full pronunciation of the utterance against the ground truth. 
        self.segmented_ground_truth_ipa_phonemes and self.segmented_uttered_ipa_phonemes need to be available.
        """
        if len(self.segmented_ground_truth_ipa_phonemes) == 0 or len(self.segmented_uttered_ipa_phonemes) == 0:
            raise ValueError("Segmented ground truth and uttered phonemes must be non-empty.")
        
        one_ground_truth = []
        one_uttered = []
        for word in self.segmented_ground_truth_ipa_phonemes:
            one_ground_truth.extend(word[0])
            one_ground_truth.append(" ")
        for word in self.segmented_uttered_ipa_phonemes:
            one_uttered.extend(word)
            one_uttered.append(" ")

        # correctly add spaces to uttered phonemes
        self.map_boundary(one_ground_truth, one_uttered)

        final_label_list = []
        for reference, uttered in zip(self.segmented_ground_truth_ipa_phonemes, self.segmented_uttered_ipa_phonemes):
            final_label_list.append(self.evaluate_pronunciation_for_word(uttered, reference))
        return final_label_list
    
    def evaluate_pronunciation_for_word(self, uttered: list, reference: list):
        """
        Evaluates pronunciation for a word.
        
        Args:
            uttered (list): A list of phonemes representing the uttered phonemes for this word.
            reference (list): A list of list, each nested list being a possible pronunciation (ground truth) of the word.
        
        Returns:
            list(tuple): Each tuple is (phoneme_label)
        """
        max_score = None
        final_label_list = []
        for ground_truth in reference:
            score = 0
            label_list = []

            alignments = pairwise2.align.globalms(
                ground_truth, uttered, 
                match=1,  # Score for match
                mismatch=-1,  # Penalty for mismatch
                open=-2,  # Penalty for opening a gap
                extend=-1,  # Penalty for extending a gap,
                gap_char=['-']
            )

            # Extract the aligned sequences
            aligned_ground_truth, aligned_uttered, _, _, _ = alignments[0]

            # Iterate through the characters in the aligned sequences
            for gt_char, utt_char in zip(aligned_ground_truth, aligned_uttered):
                # Skip gaps in the ground truth
                if gt_char == '-' or gt_char == ' ':
                    continue

                # Assign a label based on the tuple (gt_char, utt_char)
                if utt_char != '-':  # Only consider matched characters, not gaps in uttered
                    key = (gt_char, utt_char)
                    if key in self.phoneme_pair_label and self.phoneme_pair_label[key] in [1, 2]:
                        label = 1
                        score += 1
                    else:
                        label = 0
                else:
                    label = 0  # Default label for unmatched characters

                # Append the result as a tuple (ground_truth_char, label)
                label_list.append((gt_char, label))

            if max_score is None or score > max_score:
                max_score = score
                final_label_list = label_list
        # Return the label list
        return final_label_list

    def map_phonemes_to_segments(self, phoneme_labels, word):
        """
        Maps each phoneme in the phoneme set to its corresponding segment (orthography) in the word.
        
        Args:
            phoneme_labels (list): List of phoneme labels in order.
            word (str): The word to map the phonemes to.

        Returns:
            list: List of tuples, each containing a phoneme and its corresponding segment.
        """
        result = []
        remaining_word = word

        for phoneme_tup in phoneme_labels:
            phoneme = phoneme_tup[0]
        
            if phoneme not in self.ipa_to_orthography:
                # Skip the phoneme if not found in the map
                continue

            possible_spellings = self.ipa_to_orthography[phoneme]
            # Sort spellings by length in descending order to prioritize the longest match
            possible_spellings.sort(key=len, reverse=True)

            matched_spelling = None
            skipped_characters = []

            while remaining_word: # WORKING: if possible_spellings are not exhaustive, will consider the rest a silient grapheme
                for spelling in possible_spellings:
                    if remaining_word.startswith(spelling):
                        matched_spelling = spelling
                        break

                if matched_spelling:
                    break

                # If no match, treat the current character as part of a silent grapheme
                skipped_characters.append(remaining_word[0])
                remaining_word = remaining_word[1:]

            if not matched_spelling: # reach the end of word but no match, possibly meaning the possible_spellings are not exhaustive
                matched_spelling = "" 

            # Add skipped characters to the result as silent graphemes
            for char in skipped_characters:
                result.append((('', 1), char))

            # Add the phoneme and matched spelling to the result
            result.append((phoneme_tup, matched_spelling))

            # Update the remaining word by removing the matched spelling
            if matched_spelling:
                remaining_word = remaining_word[len(matched_spelling):]

        if remaining_word:
            result.append((('', 1), remaining_word))
            print(f"Unmapped segment of the word remains: '{remaining_word}'")

        return result
    
    def map_phonemes_to_segments_for_api(self, phoneme_labels, word):
        """
        Maps each phoneme in the phoneme set to its corresponding segment (orthography) in the word.
        Same as above, but different format to return the API call
        Args:
            phoneme_labels (list): List of phoneme labels in order.
            word (str): The word to map the phonemes to.

        Returns:
            list: List of tuples, each containing a phoneme and its corresponding segment.
        """
        result = {"word": word, "details": []}
        remaining_word = word

        for phoneme_tup in phoneme_labels:
            phoneme = phoneme_tup[0]
        
            if phoneme not in self.ipa_to_orthography:
                # Skip the phoneme if not found in the map
                continue

            possible_spellings = self.ipa_to_orthography[phoneme]
            # Sort spellings by length in descending order to prioritize the longest match
            possible_spellings.sort(key=len, reverse=True)

            matched_spelling = None
            skipped_characters = []

            while remaining_word: # WORKING: if possible_spellings are not exhaustive, will consider the rest a silient grapheme
                for spelling in possible_spellings:
                    if remaining_word.startswith(spelling):
                        matched_spelling = spelling
                        break

                if matched_spelling:
                    break

                # If no match, treat the current character as part of a silent grapheme
                skipped_characters.append(remaining_word[0])
                remaining_word = remaining_word[1:]

            if not matched_spelling: # reach the end of word but no match, possibly meaning the possible_spellings are not exhaustive
                matched_spelling = "" 

            # Add skipped characters to the result as silent graphemes
            for char in skipped_characters:
                result["details"].append({
                    "phoneme": "",  # No phoneme
                    "word_segment": char,
                    "label": 1  # Assuming label for silent graphemes is 1
                })

            # Add the phoneme and matched spelling to the result
            result["details"].append({
                "phoneme": phoneme_tup[0],
                "word_segment": matched_spelling,
                "label": phoneme_tup[1]  # Assuming `phoneme_tup[1]` is the label
            })

            # Update the remaining word by removing the matched spelling
            if matched_spelling:
                remaining_word = remaining_word[len(matched_spelling):]

        if remaining_word: # WORKING: if possible_spellings are not exhaustive, will consider the rest a silient grapheme
            result["details"].append({
                "phoneme": "",  # No phoneme
                "word_segment": remaining_word,
                "label": 1  
            })
            print(f"Unmapped segment of the word remains: '{remaining_word}'")

        return result
    
    def generate_labels(self, display=True):
        results = []
        labels = self.evaluate_full_pronunciation()
        for label, word in zip(labels, self.transcript.split()):
            results.append(self.map_phonemes_to_segments(label, word))

        if display:
            self.display_ipa_phonemes_with_labels_and_segments(results, self.transcript)
        return results    
    
    def generate_labels_for_api(self):
        results = []
        labels = self.evaluate_full_pronunciation()
        for label, word in zip(labels, self.transcript.split()):
            results.append(self.map_phonemes_to_segments_for_api(label, word))

        return results    
    
    def handle_label_shenanigans(self, labels):
        """
        Handle label shenanigans manually.
        - if θ is the last phoneme in a word, and it's labelled 0, change it to 1
        """
        for word in labels:
            if word[-1][0] == "θ" and word[-1][1] == 0:
                word[-1] = ("θ", 1)
        return labels
        
    def display_ipa_phonemes_with_labels_and_segments(self, data, words):
        """
        Display phonemes and their corresponding segments with labels.
        Incorrect phonemes and segments are displayed in red.

        Parameters:
        data (list of lists): Each sublist represents a word, and each element is ((phoneme, label), corresponding_segment).
        words (list of str): List of corresponding words for the data.
        """
        # Initialize containers for styled phonemes and styled words
        styled_phonemes = []
        styled_words = []

        for word_data, word in zip(data, words):
            # Process phonemes and segments for each word
            styled_phoneme_word = []
            styled_word = []
            for ((phoneme, label), segment) in word_data:
                if label == 0:
                    # Incorrect phoneme or segment
                    styled_phoneme_word.append(f"<span style='color:red;'>{phoneme}</span>")
                    styled_word.append(f"<span style='color:red;'>{segment}</span>")
                else:
                    # Correct phoneme and segment
                    styled_phoneme_word.append(f"<span>{phoneme}</span>")
                    styled_word.append(f"<span>{segment}</span>")

            # Join phonemes for the current word and add to the phoneme container
            styled_phonemes.append("".join(styled_phoneme_word))
            styled_words.append("".join(styled_word))
        # Combine phonemes and words for display
        phoneme_content = " ".join(styled_phonemes)
        word_content = " ".join(styled_words)

        # Construct complete HTML
        html_content = f"<div style='font-size:20px;'>{phoneme_content} - <b>{word_content}</b></div>"

        # Display
        display(HTML(html_content))
        
# health check
@app.get("/")
def home():
    return "Healthy bro!"

import time # temp

# taking in both audio and transcript from the user
@app.post("/predict")
async def predict(audio: UploadFile, transcript: str = Form(...)):
    """
    Predict phoneme labels from uploaded audio and provided transcript.

    Args:
        audio (UploadFile): Uploaded audio file (WAV/MP3).
        transcript (str): Ground truth transcript.

    Returns:
        JSONResponse: Contains phoneme labels.
    """
    logging.info("Received prediction request!")

    # Validate file extension
    allowed_extensions = {"wav", "mp3", "m4a"}
    filename = audio.filename.lower()
    start_time = time.time()

    if not filename.endswith(tuple(allowed_extensions)):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only WAV and MP3 files are supported.",
        )

    # Load and preprocess the audio
    try:
        audio_input, input_values = await process_audio(audio, device)
        
        end_time = time.time()
        print(f"Time from call to finish processing audio: {end_time - start_time} seconds")
        
        # clean transcript
        if transcript.lower().strip() == "//":
            transcript = transcribe_into_English(audio_input)
            print("HERE1")
        transcript = clean_text(transcript).strip()
        another_end_time = time.time()
        logging.info(f"Transcript: {transcript}, Time taken from processed audio to finish transcription: {another_end_time - end_time} seconds")

        # Perform inference
        with torch.no_grad():
            logits = model(input_values).logits

        # Decode the phonemes
        predicted_ids = torch.argmax(logits, dim=-1)
        uttered_phonemes = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0] 
        end_time = time.time()
        print("Time taken for inference:", end_time - start_time)
        
        # init PronunciationAssessment instance
        cur = PronunciationAssessment(transcript, uttered_phonemes)
        cur.convert_transcript_into_phonemes()
        cur.clean_ipa_phonemes()
        cur.split_phoneme_sequence()
        print(cur.uttered_ipa_phonemes)
        print(cur.segmented_ground_truth_ipa_phonemes)
        print(cur.segmented_uttered_ipa_phonemes)

        # generate the final labels
        labels = cur.generate_labels_for_api()
        return JSONResponse(content={"labels": labels})
    
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during processing.")

# taking in audio only and returning the transcript
@app.post("/transcribe")
async def transcribe(audio: UploadFile):
    """
    Transcribe the uploaded audio and return the transcript.

    Args:
        audio (UploadFile): Uploaded audio file (WAV/MP3).

    Returns:
        JSONResponse: Contains the transcript.
    """
    logging.info("Received transcription request!")

    # Validate file extension
    allowed_extensions = {"wav", "mp3", "m4a"}
    filename = audio.filename.lower()
    start_time = time.time()
    if not filename.endswith(tuple(allowed_extensions)):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only WAV and MP3 files are supported.",
        )

    # Load and preprocess the audio
    try:
        audio_input = await process_audio(audio, device, False)

        # Get transcript
        end_time = time.time()
        print(f"Time from call to finish processing audio: {end_time - start_time} seconds")
        transcript = transcribe_into_English(audio_input)
        transcript = clean_text(transcript).strip()
        another_end_time = time.time()
        logging.info(f"Transcript: {transcript}, Time taken from processed audio to finish transcription: {another_end_time - end_time} seconds")

        return JSONResponse(content={"transcript": transcript})

    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during processing.")

# if __name__ == '__main__':
#     port = os.environ.get("PORT", 10000)  # Default to 10000 if PORT is not set
#     logging.info(f"Starting server on PORT {port}")
#     uvicorn.run("app:app", host="0.0.0.0", port=int(port), log_level="info")