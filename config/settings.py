import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    transcriber_model_name: str = "openai/whisper-tiny.en"
    ssl_model_name: str = "mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme"
    device: str = "cpu"  # or "cuda" if available
    cache_ttl: int = 300  # seconds
    cache_maxsize: int = 100


@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    sample_rate: int = 16000
    channels: int = 1
    max_audio_length: int = 300  # seconds
    supported_formats: tuple = ("wav", "mp3", "m4a", "flac")


@dataclass
class APIConfig:
    """Configuration for API settings."""
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    request_timeout: int = 300  # seconds
    cors_origins: list = None


@dataclass
class Settings:
    """Main application settings."""
    environment: str = os.getenv("ENVIRONMENT", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Sub-configurations
    model: ModelConfig = ModelConfig()
    audio: AudioConfig = AudioConfig()
    api: APIConfig = APIConfig()
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.api.cors_origins is None:
            self.api.cors_origins = ["*"] if self.debug else []
        
        # Override device if CUDA is available and requested
        if os.getenv("FORCE_CPU", "false").lower() != "true":
            try:
                import torch
                if torch.cuda.is_available():
                    self.model.device = "cuda"
            except ImportError:
                pass 