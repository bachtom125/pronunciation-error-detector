import logging
import os
import uvicorn
from fastapi import FastAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set environment variables for better performance
os.environ["NUMBA_CACHE_DIR"] = "/tmp"

# Package imports
from routes.transcribe import router as transcriber_router
from routes.predict import router as pronunciation_evaluation_router
from config.settings import Settings

# Initialize settings
settings = Settings()

# Initialize FastAPI app
app = FastAPI(
    title="Pronunciation Error Detection API",
    version="1.0.0",
    description="AI-powered pronunciation assessment and transcription service"
)

@app.get("/", tags=["health"])
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "pronunciation-error-detection"}

@app.get("/health", tags=["health"])
def detailed_health_check():
    """Detailed health check with system information."""
    return {
        "status": "healthy",
        "service": "pronunciation-error-detection",
        "version": "1.0.0",
        "environment": settings.environment
    }

# Include routers
app.include_router(transcriber_router, prefix="/api/v1", tags=["transcription"])
app.include_router(pronunciation_evaluation_router, prefix="/api/v1", tags=["pronunciation"])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logging.info(f"Starting server on port {port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=settings.environment == "development"
    )