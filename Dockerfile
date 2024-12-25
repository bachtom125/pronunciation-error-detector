# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the Hugging Face cache directory
ENV TRANSFORMERS_CACHE=/workspace/transformers_cache

# Create the cache directory and ensure it's writable
RUN mkdir -p /workspace/transformers_cache && chmod -R 777 /workspace/transformers_cache

# Copy the rest of the application code
COPY . .

# Expose the application port
EXPOSE 7860

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
