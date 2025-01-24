from fastapi import FastAPI, UploadFile, Form, HTTPException, APIRouter, Depends
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
import logging
from joblib import Memory
from difflib import SequenceMatcher
import eng_to_ipa as ipa_conv
import copy   
from IPython.display import HTML, display
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from pydub import AudioSegment
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import asyncio
from cachetools import TTLCache
import time 
import os
from tempfile import NamedTemporaryFile
import subprocess
import librosa

# package imports
from services.evaluate_pronunciation import PronunciationEvalService
from utils.general_utils import clean_text    

router = APIRouter()

@router.post("/predict", summary="Evaluate pronunciation")
async def evaluate_pronunciation(audio: UploadFile, transcript: str = Form(...)):
    """
    Predict phoneme labels from uploaded audio and provided transcript.

    Args:
        audio (UploadFile): Uploaded audio file (WAV/MP3).
        transcript (str): Ground truth transcript.

    Returns:
        JSONResponse: Contains phoneme labels.
    """
    try:
        # Call the service to process and transcribe the audio
        service = PronunciationEvalService(transcript, audio)
        labels = await service.generate_labels()

        response = {'labels': labels}
        return JSONResponse(content=response)

    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during processing.")