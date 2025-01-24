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
from services.transcribe import TranscriptionService
from utils.general_utils import clean_text    

router = APIRouter()

service = TranscriptionService()
@router.post("/transcribe", summary="Trancribe audio into English")
async def transcribe(audio: UploadFile):
    """
    Transcribe the uploaded audio and return the transcript.

    Args:
        audio (UploadFile): Uploaded audio file.

    Returns:
        JSONResponse: Contains the transcript.
    """
    try:
        # Call the service to process and transcribe the audio
        transcript = await service.transcribe_audio(audio)
        transcript = clean_text(transcript).strip()

        response = {'transcript': transcript}
        return JSONResponse(content=response)

    except ValueError as ve:
        logging.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during processing.")