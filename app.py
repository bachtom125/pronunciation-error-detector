
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

def load_audio(audio_path, target_sr=16000):
  """Load an audio file and resample it to 16kHz."""
  audio, sr = librosa.load(audio_path, sr=target_sr)
  return audio

ipa_phonemes = [
    # Vowels
    "i", "y", "ɨ", "ʉ", "ɪ", "ʏ", "e", "ø", "ɘ", "ɵ", "ə", "ɛ", "œ", "æ", "a", "ɶ", 
    "ɒ", "ʌ", "ɔ", "o", "ɤ", "u", "ɯ", "ʊ", "ɜ", "ɞ", "ɐ",  # Monophthongs
    "ɚ", "ɝ",  # Rhotacized vowels
    "aɪ", "aʊ", "ɔɪ", "eɪ", "oʊ",  # Common diphthongs in English
    # Consonants
    "p", "b", "t", "d", "k", "g", "ʔ",  # Plosives
    "m", "ɱ", "n", "ɳ", "ɲ", "ŋ", "ɴ",  # Nasals
    "ʙ", "r", "ɾ", "ɽ",  # Trills and taps
    "ɸ", "β", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "ʂ", "ʐ", "ç", "ʝ", "x", "ɣ", "χ", "ʁ", "ħ", "ʕ", "h", "ɦ",  # Fricatives
    "ʋ", "ɹ", "ɻ", "j", "ɰ",  # Approximants
    "l", "ɭ", "ʎ", "ʟ",  # Laterals
    "w", "ɥ", "ʍ", "ɧ",  # Coarticulated
    "ʦ", "ʣ", "ʧ", "ʤ", "ʨ", "ʥ", # Affricates (Common examples)
    # Diacritics and other symbols
    "ˈ", "ˌ",  # Stress (primary, secondary)
    "ː", "ˑ",  # Length (long, half-long)
    ".", "|", "‖",  # Breaks (syllable, minor group, major group)
    "˞",  # Rhoticity
    "̩", "̯" ,
    "̪", "̠", "̟", "̹", "̜", "̬", "̥", "̊", "̤", "̰", "̼", "̩", "̯", "̃", "̚", "̝", "̞",
    "˥", "˦", "˧", "˨", "˩",  # Tone levels
    "↗", "↘"  # Global rise and fall
]

arpabet_to_ipa = {
    # Vowels
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ʌ",
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "EH": "ɛ",
    "ER": "ɝ",
    "EY": "eɪ",
    "IH": "ɪ",
    "IY": "i",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "UH": "ʊ",
    "UW": "u",
    "AX": "ə",
    "IX": "ɨ",

    # Consonants
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "P": "p",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ"
}

# Invert the dictionary to map IPA to ARPAbet
ipa_to_arpabet = {v: k for k, v in arpabet_to_ipa.items()}

# NOTE: removed all long signals ('ː') for compatibility with eng_to_ipa lib. American English.
ipa_to_orthography = {
    # Vowels
    "i": ["ee", "ea", "e", "ie", "ei", "ey"],
    "ɪ": ["i", "y", 'e'],
    "e": ["e", "ea"],
    "ɛ": ["e", "ea", "ai"],
    "æ": ["a"],
    "ɑ": ["ar", "a", "o"],
    "ɒ": ["o"],
    "ɔ": ["aw", "au", "o", "oar", "ore", "a", "oo"],
    "ʊ": ["u", "oo", "ou"],
    "u": ["oo", "u", "ew", "ou", "o"],
    "ʌ": ["u", "o"],
    "ɜ": ["er", "ur", "ir", "ear", "or"],
    "ə": ["a", "e", "o", "u"],

    # Diphthongs
    "eɪ": ["a", "ai", "ay", "ey", "eigh"],
    "aɪ": ["i", "y", "igh", "ie", "uy"],
    "ɔɪ": ["oi", "oy"],
    "oʊ": ["o", "oa", "ow", "ough"],  # Added
    "aʊ": ["ou", "ow"],
    "əʊ": ["o", "oe", "oa", "ow"],
    "ɪə": ["ear", "eer", "ere"],
    "eə": ["air", "are", "ere"],
    "ʊə": ["ure", "our"],
    "aɪə": ["ire", "ier"],  # Additional diphthongs
    "aʊə": ["our", "hour"],
    "juː": ["u", "ew", "ue", "eu"],  # Found in "few", "cue", etc.

    # Consonants
    "p": ["p", "pp"],
    "b": ["b", "bb"],
    "t": ["t", "tt"],
    "d": ["d", "dd"],
    "k": ["k", "c", "ck", "ch", "cc"],
    "g": ["g", "gg"],
    "f": ["f", "ff", "ph"],
    "v": ["v", "vv"],
    "θ": ["th"],
    "ð": ["th"],
    "s": ["s", "ss", "c", "ce"],
    "z": ["z", "zz", "s", "x"],
    "ʃ": ["sh", "ss", "ch"],
    "ʒ": ["s", "si", "z"],
    "h": ["h"],
    "m": ["m", "mm"],
    "n": ["n", "nn"],
    "ŋ": ["ng", "n"],
    "l": ["l", "ll"],
    "r": ["r", "rr"],
    "ɹ": ["r", "rr"],
    "j": ["y"],
    "w": ["w", "wh"],
    "ʧ": ["ch", "tch"],
    "ʦ": ["ts", "tz", "z"],  # As in "cats," "blitz," or "pizza" (loanwords)
    "ʣ": ["ds", "dz"],       # Rare in English, examples in loanwords like "adze"
    "ʤ": ["j", "dg", "dge", "g"],  # As in "judge," "edge," or "giant"
    "ʨ": ["q", "ch"],        # Rare in English, examples in borrowed words
    "ʥ": ["j"],      
    "dʒ": ["j", "g", "dg"],
    "x": ["ch"],
    "ʔ": [],  # Glottal stop has no direct orthographic equivalent
}

def convert_ipa_to_arpabet(ipa_words):
    """
    Convert a list of IPA words (strings of concatenated phonemes) to ARPAbet words.

    :param ipa_words: List of IPA words where each word is a string of concatenated phonemes.
    :return: List of lists, where each inner list contains ARPAbet phonemes for a word.
    """
    arpabet_words = []
    for word in ipa_words:
        # Break the word into phonemes
        phonemes = []  # Collect matched phonemes
        i = 0
        while i < len(word):
            matched = False
            # Match multi-character IPA phonemes first
            for ipa_phoneme in sorted(ipa_to_arpabet.keys(), key=len, reverse=True):
                if word[i:].startswith(ipa_phoneme):
                    phonemes.append(ipa_to_arpabet[ipa_phoneme])
                    i += len(ipa_phoneme)
                    matched = True
                    break
            # If no match, add an unknown marker and move forward
            if not matched:
                phonemes.append("<UNK>")
                i += 1
        # Append the list of phonemes for the word
        arpabet_words.append(phonemes)
    return arpabet_words

def remove_numbers_from_phonemes(phon_list):
    """
    Remove all numbers from phonemes in a nested list.

    Parameters:
        phon_list (list of lists): Nested list of phonemes.

    Returns:
        list of lists: Updated nested list with numbers removed from phonemes.
    """
    cleaned_phon_list = []
    for word_phonemes in phon_list:
        cleaned_word = [re.sub(r'\d', '', phoneme) for phoneme in word_phonemes]
        cleaned_phon_list.append(cleaned_word)
    return cleaned_phon_list

def align_phoneme_sequences(truth_words, uttered_words, gap_penalty=1, substitution_cost=1):
    """
    Align phoneme sequences separated by words.

    Parameters:
        truth_words (list of lists): Ground truth phoneme sequences grouped by words.
        uttered_words (list of lists): Uttered phoneme sequences grouped by words.
        gap_penalty (int): Penalty for gaps.
        substitution_cost (int): Cost for substitutions.

    Returns:
        alignment (list of tuples): Aligned phoneme sequences with '-' for gaps.
    """
    def align_two_sequences(seq1, seq2):
        """
        Align two sequences using dynamic programming.
        """
        n = len(seq1)
        m = len(seq2)
        dp = np.zeros((n + 1, m + 1))

        # Initialize DP table
        for i in range(n + 1):
            dp[i][0] = i * gap_penalty
        for j in range(m + 1):
            dp[0][j] = j * gap_penalty

        # Fill DP table
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                match_cost = 0 if seq1[i - 1] == seq2[j - 1] else substitution_cost
                dp[i][j] = min(
                    dp[i - 1][j - 1] + match_cost,  # Match or substitution
                    dp[i - 1][j] + gap_penalty,    # Deletion
                    dp[i][j - 1] + gap_penalty     # Insertion
                )

        # Traceback to find alignment
        alignment_seq1 = []
        alignment_seq2 = []
        i, j = n, m
        while i > 0 or j > 0:
            if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if seq1[i - 1] == seq2[j - 1] else substitution_cost):
                alignment_seq1.append(seq1[i - 1])
                alignment_seq2.append(seq2[j - 1])
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + gap_penalty:
                alignment_seq1.append(seq1[i - 1])
                alignment_seq2.append('-')
                i -= 1
            else:
                alignment_seq1.append('-')
                alignment_seq2.append(seq2[j - 1])
                j -= 1

        return alignment_seq1[::-1], alignment_seq2[::-1]

    # Align each word pair
    alignment = []
    for truth_word, uttered_word in zip(truth_words, uttered_words):
        aligned_truth, aligned_uttered = align_two_sequences(truth_word, uttered_word)
        alignment.append((aligned_truth, aligned_uttered))

    return alignment

def generate_phoneme_labels(data):
    """
    Generate phoneme labels for comparison of expected and uttered phonemes.

    Parameters:
    data (list of tuples): Each tuple contains (expected phonemes, uttered phonemes).

    Returns:
    list of tuples: Each tuple contains (phonemes, labels).
                    Phonemes are from the expected list, and labels are binary (0: correct, 1: incorrect).
    """
    results = []
    for expected, uttered in data:
        labels = [
            0 if exp == utt else 1
            for exp, utt in zip(expected, uttered)
        ]
        results.append((expected, labels))
    return results

def convert_words_to_phonemes(words, cmu_dict):
  phonemes = []
  for word in words:
    if word in cmu_dict:
      phonemes.append(cmu_dict[word][0])  # Use the first phoneme representation
    else:
      phonemes.append('<UNK>')  # Append 'UNK' for unknown words
  return phonemes

def remove_ipa_stress_markers(ipa):
    return ipa.replace("ˈ", "").replace("ˌ", "")

def evaluate_pronunciation(reference: list, pronunciation: list):
    """
    Evaluate the pronunciation of a word or sentence by comparing it to a reference.
    
    Args:
        reference (list): A list of phonemes representing the correct pronunciation.
        pronunciation (list): A list of phonemes representing the pronunciation to be evaluated.

    Returns:
        dict: A dictionary containing the evaluation results.
    """
    matcher = SequenceMatcher(None, reference, pronunciation)
    alignment = matcher.get_opcodes()
    
    # Initialize results for errors and labels
    errors = {"matches": [], "substitutions": [], "insertions": [], "deletions": []}
    labels = []
    processed_indices = set()  # Track indices in the reference that are processed
    
    # Process each alignment operation
    for tag, i1, i2, j1, j2 in alignment:
        if tag == "equal":
            # Matches: Add to errors and label as 1
            errors["matches"].extend(reference[i1:i2])
            labels.extend([(phoneme, 1) for phoneme in reference[i1:i2]])
            processed_indices.update(range(i1, i2))
        elif tag == "replace":
            # Substitutions: Check phoneme-by-phoneme
            ref_segment = reference[i1:i2]
            pron_segment = pronunciation[j1:j2]
            
            for ref_phoneme, pron_phoneme in zip(ref_segment, pron_segment):
                if ref_phoneme != pron_phoneme:
                    errors["substitutions"].append((ref_phoneme, pron_phoneme))
                    labels.append((ref_phoneme, 0))
                    processed_indices.add(i1)
                    i1 += 1  # Move to the next index in the reference
            
            # Handle leftover phonemes in reference (deletions)
            if len(ref_segment) > len(pron_segment):
                for leftover in ref_segment[len(pron_segment):]:
                    errors["deletions"].append(leftover)
                    labels.append((leftover, 0))
                    processed_indices.add(i1)
                    i1 += 1
            
            # Handle leftover phonemes in pronunciation (insertions)
            if len(pron_segment) > len(ref_segment):
                for leftover in pron_segment[len(ref_segment):]:
                    errors["insertions"].append(leftover)
        elif tag == "insert":
            # Insertions: Add to errors, no effect on reference labels
            errors["insertions"].extend(pronunciation[j1:j2])
        elif tag == "delete":
            # Deletions: Add to errors and label as 0
            errors["deletions"].extend(reference[i1:i2])
            labels.extend([(phoneme, 0) for phoneme in reference[i1:i2]])
            processed_indices.update(range(i1, i2))
    
    # Post-check: Ensure all phonemes in the reference are processed
    for i, phoneme in enumerate(reference):
        if i not in processed_indices:
            errors["deletions"].append(phoneme)
            labels.append((phoneme, 0))
    
    return errors, labels

def split_phoneme_sequence(sequence):
    """
    Splits a phoneme sequence into individual phonemes based on the IPA dictionary keys.
    """
    phonemes = []
    i = 0
    keys = sorted(ipa_phonemes, key=len, reverse=True)  # Prioritize longer matches
    while i < len(sequence):
        match = None
        for key in keys:
            if sequence[i:i+len(key)] == key:
                match = key
                phonemes.append(match)
                i += len(key)
                break
        if not match:  # No phoneme matched
            raise ValueError(f"Unknown phoneme in sequence: {sequence[i:]}")
    return phonemes

def map_phonemes_to_segments(phoneme_labels, word):
    """
    Maps each phoneme in the phoneme set to its corresponding segment in the word.
    
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
    
        if phoneme not in ipa_to_orthography:
            # Skip the phoneme if not found in the map
            continue

        possible_spellings = ipa_to_orthography[phoneme]
        # Sort spellings by length in descending order to prioritize the longest match
        possible_spellings.sort(key=len, reverse=True)

        matched_spelling = None
        skipped_characters = []

        while remaining_word:
            for spelling in possible_spellings:
                if remaining_word.startswith(spelling):
                    matched_spelling = spelling
                    break

            if matched_spelling:
                break

            # If no match, treat the current character as part of a silent grapheme
            skipped_characters.append(remaining_word[0])
            remaining_word = remaining_word[1:]

        if not matched_spelling:
            matched_spelling = ""  # Treat as silent grapheme

        # Add skipped characters to the result as silent graphemes
        for char in skipped_characters:
            result.append((('', 1), char))

        # Add the phoneme and matched spelling to the result
        result.append((phoneme_tup, matched_spelling))

        # Update the remaining word by removing the matched spelling
        if matched_spelling:
            remaining_word = remaining_word[len(matched_spelling):]
        print(result)
    if remaining_word:
        raise ValueError(f"Unmapped portion of the word remains: '{remaining_word}'")

    return result

def generate_segment_labels(ground_truth_phonemes, uttered_phonemes, transcript):
    # this assumes that the two lists are the same in lengths
    combined_labels = []

    for uttered_phons_set, ground_truth_phons_set, word in zip(uttered_phonemes.split(), remove_ipa_stress_markers(ground_truth_phonemes).split(), transcript.split()):
        u_phons = split_phoneme_sequence(uttered_phons_set)
        g_phons = split_phoneme_sequence(ground_truth_phons_set)
        errors, labels = evaluate_pronunciation(g_phons, u_phons)
        phoneme_segment_map = map_phonemes_to_segments(labels, word)
        combined_labels.append(phoneme_segment_map)
        
    return combined_labels

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

        # Perform inference
        with torch.no_grad():
            logits = model(input_values).logits

        # Decode the phonemes
        predicted_ids = torch.argmax(logits, dim=-1)
        uttered_phonemes = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0] 
        ground_truth_phonemes = ipa_conv.convert(transcript)
        labels = generate_segment_labels(ground_truth_phonemes, uttered_phonemes, transcript)

        return JSONResponse(content={"labels": labels})
    
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during processing.")

# if __name__ == '__main__':
#     port = os.environ.get("PORT", 10000)  # Default to 10000 if PORT is not set
#     logging.info(f"Starting server on PORT {port}")
#     uvicorn.run("app:app", host="0.0.0.0", port=int(port), log_level="info")
