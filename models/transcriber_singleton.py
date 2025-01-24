import os
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from utils.general_utils import process_audio
import asyncio
import librosa

class TranscriberSingleton:
    _instance = None  

    def __new__(cls, model_name="openai/whisper-tiny.en", device=None):
        if cls._instance is None:
            cls._instance = super(TranscriberSingleton, cls).__new__(cls)
            cls._instance._initialize(model_name, device)
        return cls._instance

    def _initialize(self, model_name, device):
        # Set device (CPU or GPU)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        # Load processor and model
        print(f"Loading Whisper processor and model into {device}...")  # This will only happen once
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)  # Move model to the specified device

    def transcribe_into_English(self, audio_input):
        # Load audio file
        audio_input = self.processor(audio_input, sampling_rate=16000, return_tensors="pt", language="en").to(self.device)

        # Perform transcription
        with torch.no_grad():
            generated_ids = self.model.generate(audio_input.input_features)

        # Decode the transcription
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription.lower().strip()
    
    def transcribe_from_file_path(self, file_path, target_sr=16000):
        with open(file_path, "rb") as f:
            audio_input, sr = librosa.load(file_path, sr=target_sr)
        return self.transcribe_into_English(audio_input)

transcriber_model = TranscriberSingleton()