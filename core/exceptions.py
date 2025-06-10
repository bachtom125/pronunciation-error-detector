"""Custom exceptions for the pronunciation error detection application."""


class PronunciationDetectionError(Exception):
    """Base exception for pronunciation detection errors."""
    
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class AudioProcessingError(PronunciationDetectionError):
    """Exception raised when audio processing fails."""
    
    def __init__(self, message: str = "Audio processing failed"):
        super().__init__(message, "AUDIO_PROCESSING_ERROR")


class ModelInferenceError(PronunciationDetectionError):
    """Exception raised when model inference fails."""
    
    def __init__(self, message: str = "Model inference failed"):
        super().__init__(message, "MODEL_INFERENCE_ERROR")


class UnsupportedAudioFormatError(PronunciationDetectionError):
    """Exception raised when audio format is not supported."""
    
    def __init__(self, format_type: str):
        message = f"Unsupported audio format: {format_type}"
        super().__init__(message, "UNSUPPORTED_AUDIO_FORMAT")


class AudioTooLongError(PronunciationDetectionError):
    """Exception raised when audio file is too long."""
    
    def __init__(self, duration: float, max_duration: float):
        message = f"Audio too long: {duration}s (max: {max_duration}s)"
        super().__init__(message, "AUDIO_TOO_LONG")


class CacheError(PronunciationDetectionError):
    """Exception raised when cache operations fail."""
    
    def __init__(self, message: str = "Cache operation failed"):
        super().__init__(message, "CACHE_ERROR")


class TranscriptionError(PronunciationDetectionError):
    """Exception raised when transcription fails."""
    
    def __init__(self, message: str = "Transcription failed"):
        super().__init__(message, "TRANSCRIPTION_ERROR")


class PronunciationAssessmentError(PronunciationDetectionError):
    """Exception raised when pronunciation assessment fails."""
    
    def __init__(self, message: str = "Pronunciation assessment failed"):
        super().__init__(message, "PRONUNCIATION_ASSESSMENT_ERROR") 