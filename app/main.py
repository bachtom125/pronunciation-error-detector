
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
import asyncio
from cachetools import TTLCache

# Set the Numba cache directory to a writable location
os.environ["NUMBA_CACHE_DIR"] = "/tmp"
import librosa
logging.basicConfig(level=logging.INFO)

# package imports
from routes.transcribe import router as transcriber_router
from routes.predict import router as pronunciation_evaluation_router
# Initialize FastAPI app
app = FastAPI(title="Talkiee AI", version="1.0.0")
        
# health check
@app.get("/")
def home():
    return "Healthy bro!"

app.include_router(transcriber_router, tags=["transcribe"])
app.include_router(pronunciation_evaluation_router, tags=["pronunciation_evaluation"])

# if __name__ == '__main__':
#     port = os.environ.get("PORT", 10000)  # Default to 10000 if PORT is not set
#     logging.info(f"Starting server on PORT {port}")
#     uvicorn.run("main:app", host="0.0.0.0", port=int(port), log_level="info")