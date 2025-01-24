import logging
import time
import asyncio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from models.ssl_singleton import ssl_model
from utils.general_utils import process_audio, clean_text
from modules.pronunciation_coach.pronunciation_assessor import PronunciationAssessor
from utils.cache import audio_cache
# process -> call infereence -> structure output -> return 

class PronunciationEvalService:
    def __init__(self, transcript, audio):
        """
        Initialize the transcription service.

        Args:
            transcript (str): Ground truth transcript.
            audio (UploadFile): Uploaded audio file.
        """
        self.ssl_model = ssl_model
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = 'cpu' # TEMP for testing
        self.transcript = clean_text(transcript).strip()
        self.audio = audio
        self.filename = audio.filename
        self.uttered_phonemes = None
        self.assessor = None

    async def get_uttered_phonemes(self):  
        # check if cache has filename
        audio = self.audio
        start_time = time.time()
        audio_inputs = None
        if await audio_cache.contains(self.filename):
            async with audio_cache.lock:
                if audio_cache.cache[self.filename]["uttered_phonemes"] != None: 
                    logging.info(f"Audio '{self.filename}' found in cache.")
                    end_time  = time.time()
                    logging.info(f"Time from for getting uttered phonemes: {end_time - start_time} seconds")
                    return audio_cache.cache[self.filename]["uttered_phonemes"] 
                else:
                    logging.info(f"Audio '{self.filename}' found in cache but not inferenced. Running inference...")
                    audio_inputs = audio_cache.cache[self.filename]["audio_input"]
        else: 
            logging.info(f"Audio '{self.filename}' not found in cache. Running inference...")

        if audio_inputs is None:
            cache_entry = await process_audio(audio, self.device)
            audio_inputs = cache_entry["audio_input"]

        uttered_phonemes = await self.ssl_model.infer_and_save_to_cache(self.filename, audio_inputs, self.device)
        end_time  = time.time()
        logging.info(f"Time for getting uttered phonemes: {end_time - start_time} seconds")
        return uttered_phonemes
    
    async def generate_labels(self):
        self.uttered_phonemes = await self.get_uttered_phonemes()
        start_time = time.time()
        self.assessor = PronunciationAssessor(self.transcript, self.uttered_phonemes)
        self.assessor.convert_transcript_into_phonemes()
        self.assessor.clean_ipa_phonemes()
        self.assessor.split_phoneme_sequence()

        labels = self.assessor.generate_labels_for_api()
        end_time = time.time()
        print("Time taken for label generation after getting uttered phonemes:", end_time - start_time)

        return labels   