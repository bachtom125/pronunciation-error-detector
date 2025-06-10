"""Audio processing utilities for pronunciation detection."""

import logging
import os
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Any, Optional
import numpy as np
import librosa
from fastapi import UploadFile
from pydub import AudioSegment

from core.exceptions import (
    AudioProcessingError,
    UnsupportedAudioFormatError,
    AudioTooLongError
)
from config.settings import Settings

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio file processing and validation."""
    
    def __init__(self, settings: Settings = None):
        self._settings = settings or Settings()
        self._supported_formats = self._settings.audio.supported_formats
        self._target_sample_rate = self._settings.audio.sample_rate
        self._target_channels = self._settings.audio.channels
        self._max_duration = self._settings.audio.max_audio_length
    
    def validate_audio_file(self, audio: UploadFile) -> None:
        """Validate audio file format and basic properties."""
        if not audio.filename:
            raise AudioProcessingError("Audio file must have a filename")
        
        # Extract file extension
        file_extension = Path(audio.filename).suffix.lower().lstrip('.')
        
        if file_extension not in self._supported_formats:
            raise UnsupportedAudioFormatError(file_extension)
        
        logger.debug(f"Audio file validation passed: {audio.filename}")
    
    async def process_audio(self, audio: UploadFile) -> np.ndarray:
        """
        Process uploaded audio file and return normalized audio array.
        
        Args:
            audio: The uploaded audio file
            
        Returns:
            Processed audio as numpy array
            
        Raises:
            AudioProcessingError: If processing fails
            AudioTooLongError: If audio is too long
        """
        try:
            self.validate_audio_file(audio)
            
            # Create temporary files
            with NamedTemporaryFile(delete=False, suffix=f".{Path(audio.filename).suffix}") as temp_input:
                temp_input_path = temp_input.name
                temp_input.write(await audio.read())
            
            temp_output_path = temp_input_path.replace(Path(audio.filename).suffix, ".wav")
            
            try:
                # Convert to WAV with target specifications
                await self._convert_to_wav(temp_input_path, temp_output_path)
                
                # Load and validate duration
                audio_data = self._load_and_validate_audio(temp_output_path)
                
                logger.info(f"Successfully processed audio: {audio.filename}")
                return audio_data
                
            finally:
                # Cleanup temporary files
                self._cleanup_temp_files([temp_input_path, temp_output_path])
                
        except (AudioProcessingError, UnsupportedAudioFormatError, AudioTooLongError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing audio {audio.filename}: {e}")
            raise AudioProcessingError(f"Failed to process audio: {str(e)}")
    
    async def _convert_to_wav(self, input_path: str, output_path: str) -> None:
        """Convert audio file to WAV format using FFmpeg."""
        try:
            cmd = [
                "ffmpeg", "-i", input_path,
                "-ar", str(self._target_sample_rate),
                "-ac", str(self._target_channels),
                "-y",  # Overwrite output file
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            logger.debug(f"FFmpeg conversion successful: {input_path} -> {output_path}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion failed: {e.stderr}")
            raise AudioProcessingError(f"Audio conversion failed: {e.stderr}")
        except FileNotFoundError:
            raise AudioProcessingError("FFmpeg not found. Please install FFmpeg.")
    
    def _load_and_validate_audio(self, file_path: str) -> np.ndarray:
        """Load audio file and validate its properties."""
        try:
            # Load audio using pydub for better format support
            audio_segment = AudioSegment.from_file(file_path, format="wav")
            
            # Check duration
            duration_seconds = len(audio_segment) / 1000.0
            if duration_seconds > self._max_duration:
                raise AudioTooLongError(duration_seconds, self._max_duration)
            
            # Convert to numpy array
            audio_samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            
            # Normalize
            max_val = np.iinfo(np.int16).max
            audio_samples = audio_samples / max_val
            
            # Handle stereo -> mono conversion
            if audio_segment.channels > 1:
                audio_samples = audio_samples.reshape(-1, audio_segment.channels).mean(axis=1)
            
            # Resample if necessary
            if audio_segment.frame_rate != self._target_sample_rate:
                audio_samples = librosa.resample(
                    audio_samples, 
                    orig_sr=audio_segment.frame_rate, 
                    target_sr=self._target_sample_rate
                )
            
            logger.debug(f"Audio loaded: duration={duration_seconds:.2f}s, samples={len(audio_samples)}")
            return audio_samples
            
        except (AudioTooLongError, AudioProcessingError):
            raise
        except Exception as e:
            logger.error(f"Error loading audio from {file_path}: {e}")
            raise AudioProcessingError(f"Failed to load audio: {str(e)}")
    
    def _cleanup_temp_files(self, file_paths: list) -> None:
        """Clean up temporary files."""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Cleaned up temporary file: {file_path}")
            except OSError as e:
                logger.warning(f"Failed to cleanup temporary file {file_path}: {e}")
    
    def get_audio_info(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Get information about processed audio."""
        return {
            "sample_rate": self._target_sample_rate,
            "channels": self._target_channels,
            "duration_seconds": len(audio_data) / self._target_sample_rate,
            "samples": len(audio_data),
            "max_amplitude": float(np.max(np.abs(audio_data))),
            "rms": float(np.sqrt(np.mean(audio_data**2)))
        } 