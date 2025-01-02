
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

whisper_processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny")
whisper_model.eval()
whisper_model.to(device)

# =====================================
# Section: Utils
# =====================================

def load_audio(audio_path, target_sr=16000):
  """Load an audio file and resample it to 16kHz."""
  audio, sr = librosa.load(audio_path, sr=target_sr)
  return audio

def transcribe_into_English(audio_input):
    # Load audio file
    audio_input = whisper_processor(audio_input, sampling_rate=16000, return_tensors="pt")

    # Perform transcription
    with torch.no_grad():
        generated_ids = whisper_model.generate(audio_input.input_features)

    # Decode the transcription
    transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription.lower()

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
    Remove all characters from the input string except for letters (A-Z, a-z) and spaces.
    
    Parameters:
        text (str): Input string to clean.
        
    Returns:
        str: Cleaned string containing only letters and spaces.
    """
    # Use regex to keep only letters and spaces
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    return cleaned_text

# =====================================
# Section: IPA Phonemes Utils
# =====================================


class IPA:
    def __init__(self):
        # NOTE: removed all long signals ('ː') for compatibility with L2-artic's phoneme set (ssl model training set). American English. 
        # ground truth phonemes are converted into arpabet first, and then into ipa using the arpabet_to_ipa dict, meaning the arpabet_to_ipa dict contains
        # the core ipa phoeneme set
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
            's': ['s', 'ss', 'c', 'sc', 'ps', 'st', 'ce', 'se'],  # Examples: sit, less, circle, scene, psycho, listen, pace, course
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
            'ə': ['o', 'a', 'er', 'i', 'ar', 'our', 'ur'],  # Examples: about, ladder, pencil, dollar, honour, augur
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

            # nuances where stress affect the resulting corresponding ipa phoneme
            "AH0": "ə",    # unstressed (about)
            "AH1": "ʌ",    # stressed (hut)
            "AH2": "ʌ",    # secondary stress (hut)
            "ER0": "ɚ",    # unstressed (runner)
            "ER1": "ɝ",    # stressed (bird)
            "ER2": "ɝ",    # secondary stress (bird)

            # unknown phoneme
            "unk": "unk"
        }

        self.ipa_phonemes = list(self.ipa_to_orthography.keys())
        self.ipa_phonemes.append('unk')
        self.cmu_dict = cmudict.dict()

    def get_phoneme_count(self):
        return len(self.ipa_phonemes)

    def has_phoneme(self, phoneme): 
        return phoneme in self.ipa_phonemes

    def convert_words_into_phonemes(self, words):
        """
        Convert a list of word into IPA phonems through ARPABET phonemes.
        """
        arap_phonemes = []
        for word in words:
            if word in self.cmu_dict:
                arpa_phons = self.cmu_dict[word][0]
                arap_phonemes.append(arpa_phons)  # Use the first phoneme representation
            else:
                arap_phonemes.append(['unk'])  # Append 'UNK' for unknown words
        arap_phonemes = self.clean_arpabet_phonemes(arap_phonemes)
        print(arap_phonemes)
        ipa_phonemes = []
        for word in arap_phonemes:
            cur_phonemes = []
            for phon in word:
                cur_phonemes.append(self.arpabet_to_ipa[phon])
            ipa_phonemes.append(cur_phonemes)
            
        return ipa_phonemes
        
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
    
    def clean_phonemes(self, phonemes):
        """
        Parameters:
            phonemes (str): A string of phonemes (e.g. "ˈɪŋɡlɪʃ")
        """
        phonemes = self.remove_ipa_stress_markers(phonemes)
        phonemes = self.remove_ipa_length_markers(phonemes)
        phonemes = self.remove_ipa_break_markers(phonemes)
        phonemes = self.remove_ipa_tone_markers(phonemes)
        phonemes = self.remove_ipa_global_markers(phonemes)
        phonemes = self.remove_ipa_diacritics(phonemes)
        phonemes = self.remove_tie_bars(phonemes)
        phonemes = self.correct_shenanigans(phonemes)
        return phonemes
    
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
            cleaned_word = []
            for phoneme in word_phonemes:
                if not phoneme.startswith(('AH', 'ER')):
                    cleaned_word.append(re.sub(r'\d', '', phoneme))
                else:
                    cleaned_word.append(phoneme)
            cleaned_phon_list.append(cleaned_word)

        return cleaned_phon_list
    
    def clean_arpabet_phonemes(self, arpabet_phoneme_list):
        """
        Parameters:
            arpabet_phoneme_list (list of lists): Nested list of phonemes.
        """
        cleaned_phonemes = self.remove_stress_indicator_from_arpabet_phonemes(arpabet_phoneme_list)
        return cleaned_phonemes
    
    def split_phoneme_sequence(self, sequence: str):
        """
        Splits a phoneme sequence (of one word) into individual phonemes based on the IPA dictionary keys.
        """
        phonemes = []
        i = 0
        keys = sorted(self.ipa_phonemes, key=len, reverse=True)  # Prioritize longer matches
        
        while i < len(sequence):
            match = None
            for key in keys:
                if sequence[i:i+len(key)] == key:
                    match = key
                    phonemes.append(match)
                    i += len(key)
                    break
            if not match:  # No phoneme matched
                phonemes.append('unk')
                i += 1
        return phonemes
    
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
                
                start_word_idx, start_element_idx = get_nested_position(reference, i1)
                end_word_idx, end_element_idx = get_nested_position(reference, i2 - 1)

                labels = label_specific_elements_in_reference(labels, start_word_idx, start_element_idx, end_word_idx, end_element_idx, 0)

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
            result.append((('', 0), remaining_word))
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

        if remaining_word:
            result["details"].append({
                "phoneme": "",  # No phoneme
                "word_segment": remaining_word,
                "label": 0  # assume insertion
            })
            print(f"Unmapped segment of the word remains: '{remaining_word}'")

        return result
    
    def generate_segment_labels(self, ground_truth_phonemes, uttered_phonemes, transcript):
        # this assumes that the two lists are the same in lengths
        combined_labels = []

        for uttered_phons_set, ground_truth_phons_set, word in zip(uttered_phonemes.split(), self.remove_ipa_stress_markers(ground_truth_phonemes).split(), transcript.split()):
            u_phons = self.split_phoneme_sequence(uttered_phons_set)
            g_phons = self.split_phoneme_sequence(ground_truth_phons_set)
            errors, labels = self.evaluate_pronunciation(g_phons, u_phons)
            phoneme_segment_map = self.map_phonemes_to_segments(labels, word)
            combined_labels.append(phoneme_segment_map)
            
        return combined_labels
    
    def generate_segment_labels_from_lists(self, ground_truth_phonemes, uttered_phonemes, transcript):
        # same as generate_segment_labels above, but takes lists (aleady segmented ipa phonemes) as input 
        errors, labels = self.evaluate_pronunciation(ground_truth_phonemes, uttered_phonemes)
        combined_labels = []
        for word_phon_labels, word in zip(labels, transcript.split()):
            phoneme_segment_map = self.map_phonemes_to_segments(word_phon_labels, word)
            combined_labels.append(phoneme_segment_map)
            
        return combined_labels
    
    def generate_segment_labels_from_lists_for_api(self, ground_truth_phonemes, uttered_phonemes, transcript):
        # same as generate_segment_labels above, but takes lists (aleady segmented ipa phonemes) as input 
        errors, labels = self.evaluate_pronunciation(ground_truth_phonemes, uttered_phonemes)
        combined_labels = []
        for word_phon_labels, word in zip(labels, transcript.split()):
            phoneme_segment_map = self.map_phonemes_to_segments_for_api(word_phon_labels, word)
            combined_labels.append(phoneme_segment_map)
            
        return combined_labels
    
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
    allowed_extensions = {"wav", "mp3"}
    filename = audio.filename.lower()

    if not filename.endswith(tuple(allowed_extensions)):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only WAV and MP3 files are supported.",
        )

    # Load and preprocess the audio
    try:
        audio_bytes = BytesIO(await audio.read())
        audio_input, sr = librosa.load(audio_bytes, sr=16000)
        input_values = processor(audio_input, return_tensors="pt", sampling_rate=16000).input_values
        input_values = input_values.to(device)

        # Get transcript
        transcript = transcribe_into_English(audio_input)
        transcript = clean_text(transcript)
        print(f"Transcript: {transcript}")

        # Perform inference
        with torch.no_grad():
            logits = model(input_values).logits

        # Decode the phonemes
        predicted_ids = torch.argmax(logits, dim=-1)
        uttered_phonemes = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0] 
        ground_truth_phonemes_split = IPA().convert_words_into_phonemes(transcript.split())
       
        uttered_phonemes = IPA().clean_phonemes(uttered_phonemes)
        uttered_phonemes_split = [IPA().split_phoneme_sequence(word) for word in uttered_phonemes.split()]
        
        print("Uttered:", uttered_phonemes_split)
        print("Ground :", ground_truth_phonemes_split)
        print(transcript)

        labels = IPA().generate_segment_labels_from_lists_for_api(ground_truth_phonemes_split, uttered_phonemes_split, transcript)

        return JSONResponse(content={"labels": labels})
    
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during processing.")

# if __name__ == '__main__':
#     port = os.environ.get("PORT", 10000)  # Default to 10000 if PORT is not set
#     logging.info(f"Starting server on PORT {port}")
#     uvicorn.run("app:app", host="0.0.0.0", port=int(port), log_level="info")
