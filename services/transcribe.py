import logging
import time
import asyncio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from models.transcriber_singleton import transcriber_model
from models.ssl_singleton import ssl_model
from utils.general_utils import process_audio, clean_text

# from utils.transcribe_utils import transcribe_into_English, clean_text
# process -> call infereence -> structure output -> return 

class TranscriptionService:
    def __init__(self):
        """
        Initialize the transcription service.
        """

        self.transcriber_model = transcriber_model
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = 'cpu' # TEMP for testing

    async def transcribe_audio(self, audio):
        """
        Process the uploaded audio file and return its transcription.

        Args:
            audio (UploadFile): Uploaded audio file.

        Returns:
            str: The transcript.
        """
        logging.info("Received transcription request!")

        try:
            # Step 1: Process the audio and check cache
            start_time = time.time()
            cache_entry = await process_audio(audio, self.device)
            audio_input = cache_entry["audio_input"]
            
            # Step 2: Start SSL inference in the background
            asyncio.create_task(ssl_model.infer_and_save_to_cache(audio.filename, audio_input, self.device))

            # Step 3: Get the transcript using Whisper
            end_time = time.time()
            logging.info(f"Time from call to finish processing audio: {end_time - start_time} seconds")
            transcript = self.transcriber_model.transcribe_into_English(audio_input)
            # Log processing time
            another_end_time = time.time()
            logging.info(f"Transcript: {transcript}, Time taken from processed audio to finish transcription: {another_end_time - end_time} seconds")

            return transcript

        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            raise
