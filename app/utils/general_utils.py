import re
import logging
import torch
from tempfile import NamedTemporaryFile
import numpy as np
import librosa
from pydub import AudioSegment
import subprocess
import os
from fastapi import FastAPI, UploadFile, Form, HTTPException
from io import BytesIO
from utils.cache import audio_cache
import asyncio
    
async def process_audio(audio, device):
    """
    Process an uploaded audio file and prepare input for the model.

    Args:
        audio: The uploaded audio file.
        device: The device (e.g., 'cuda' or 'cpu') to move tensors to.

    Returns:
        cache_entry: A dictionary containing processed audio and model input.
    """
    filename = audio.filename

    # Check cache for processed audio
    if await audio_cache.contains(filename):
        logging.info(f"Audio '{filename}' found in cache.")
        return await audio_cache.get(filename)
    
    # Prevent race conditions during cache writes
    async with audio_cache.lock:
        # Double-check after acquiring lock
        if audio_cache.contains_without_lock(filename):
            logging.info(f"Audio '{filename}' found in cache after lock.")
            return audio_cache.contains_without_lock(filename)
        logging.info(f"Processing audio '{filename}'.")

        # Read the audio file into a temporary file
        with NamedTemporaryFile(delete=False, suffix=".m4a") as temp_m4a:
            temp_m4a_path = temp_m4a.name
            temp_m4a.write(await audio.read())

        # Convert M4A to WAV using FFmpeg
        temp_wav_path = temp_m4a_path.replace(".m4a", ".wav")
        try:
            subprocess.run(
                [
                    "ffmpeg", "-i", temp_m4a_path,  # Input file
                    "-ar", "16000",                 # Resample to 16kHz
                    "-ac", "1",                     # Convert to mono
                    temp_wav_path                   # Output file
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
            raise HTTPException(status_code=500, detail="Failed to process audio file.")
        finally:
            os.remove(temp_m4a_path)  # Clean up the temporary M4A file

        try:
            # Read and preprocess the audio
            audio_segment = AudioSegment.from_file(temp_wav_path, format="wav")
            audio_samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            max_val = np.iinfo(np.int16).max
            audio_samples /= max_val

            if audio_segment.channels > 1:
                audio_samples = audio_samples.reshape(-1, audio_segment.channels).mean(axis=1)

            audio_input = librosa.resample(audio_samples, orig_sr=audio_segment.frame_rate, target_sr=16000)
            # input_values = processor(audio_input, return_tensors="pt", sampling_rate=16000).input_values.to(device)

            # Cache the processed audio
            cache_entry = {"audio_input": audio_input, "input_values": None, "ssl_logits": None}
            audio_cache.set_without_lock(filename, cache_entry)
            return cache_entry
        finally:
            os.remove(temp_wav_path)
    
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