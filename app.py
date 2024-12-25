
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

# handles librosa caching
os.environ["LIBROSA_CACHE_DIR"] = "/tmp/librosa"
os.makedirs(cache_dir, exist_ok=True)
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

# Original ARPAbet to IPA mapping from SoapBox Labs
arpabet_to_ipa = {
    "AA": "a", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "EH": "ɛ", "ER": "ɚ", "EY": "eɪ", "IH": "ɪ", "IY": "i", "OW": "oʊ",
    "OY": "ɔɪ", "UH": "ʊ", "UW": "u", "B": "b", "CH": "t͡ʃ", "D": "d",
    "DH": "ð", "F": "f", "G": "ɡ", "HH": "h", "JH": "dʒ", "K": "k",
    "L": "l", "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "ɹ",
    "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ"
}

# Invert the dictionary to map IPA to ARPAbet
ipa_to_arpabet = {v: k for k, v in arpabet_to_ipa.items()}

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
      phonemes.extend(cmu_dict[word][0])  # Use the first phoneme representation
    else:
      phonemes.append('<UNK>')  # Append 'UNK' for unknown words
  return phonemes

# log url
@app.on_event("startup")
async def startup_event():
    port = os.getenv("PORT", "7860")  # Default to 7860 if PORT is not set
    base_url = f"https://{os.getenv('HF_SPACE_ID')}.hf.space"
    logging.info(f"Application is running!")
    logging.info(f"Root URL: {base_url}/")
    logging.info(f"Predict Endpoint: {base_url}/predict")

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
        uttured_transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # Convert uttered IPA into SAMPA (for comparison)
        uttured_phons = convert_ipa_to_arpabet(uttured_transcript.split())

        # Convert ground truth text into SAMPA (for comparison) and remove stress markers
        trans_phons = [convert_words_to_phonemes([word], cmu) for word in transcript.split()]
        cleaned_trans_phons = remove_numbers_from_phonemes(trans_phons)

        # Generate labels
        alignment = align_phoneme_sequences(cleaned_trans_phons, uttured_phons)
        phoneme_labels = generate_phoneme_labels(alignment)

        return JSONResponse(content={"phoneme_labels": phoneme_labels})
    
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during processing.")

# if __name__ == '__main__':
#     port = os.environ.get("PORT", 10000)  # Default to 10000 if PORT is not set
#     logging.info(f"Starting server on PORT {port}")
#     uvicorn.run("app:app", host="0.0.0.0", port=int(port), log_level="info")
