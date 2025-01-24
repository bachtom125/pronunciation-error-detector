import os
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from utils.general_utils import process_audio
import asyncio
import librosa
from utils.cache import audio_cache

class SSLSingleton:
    _instance = None  

    def __new__(cls, model_name="mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme", device=None):
        if cls._instance is None:
            cls._instance = super(SSLSingleton, cls).__new__(cls)
            cls._instance._initialize(model_name, device)
        return cls._instance

    def _initialize(self, model_name, device):
        # Set device (CPU or GPU)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        # Load processor and model
        print("Loading SSL processor and model...")  # This will only happen once
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)  # Move model to the specified device

    # an infernce function taking in processed audio input and returning the predictions
    def infer(self, audio_input, device):
        inputs = self.processor(audio_input, sampling_rate=16000, return_tensors="pt")
        inputs = inputs.to(self.device)

        with torch.no_grad():
            logits = self.model(inputs.input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        uttered_phonemes = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return uttered_phonemes

    async def infer_and_save_to_cache(self, file_name, audio_input, device):
        uttered_phonemes = self.infer(audio_input, device)
        async with audio_cache.lock:
            new_cache = audio_cache.cache[file_name]
            new_cache["uttered_phonemes"] = uttered_phonemes
            audio_cache.cache[file_name] = new_cache
        return uttered_phonemes
ssl_model = SSLSingleton()