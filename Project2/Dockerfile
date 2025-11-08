# Production-optimized Dockerfile for memory-constrained environments
# This version focuses purely on reducing the Docker image size
# WITHOUT modifying training hyperparameters

FROM python:3.11-slim

WORKDIR /app

# Install only essential build dependencies and clean up in same layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Copy requirements
COPY requirements.txt .

# Install CPU-only PyTorch (much smaller than CUDA version)
# This is the KEY optimization for memory-constrained environments
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies without cache
RUN pip install --no-cache-dir -r requirements.txt

# Remove build dependencies to reduce image size further
RUN apt-get purge -y --auto-remove build-essential

# Copy application code
COPY train_glue_transformer.py .

# Set environment variables to optimize memory usage at runtime
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV HF_HOME=/tmp/huggingface

# Default command - NO hyperparameter changes
CMD ["python", "train_glue_transformer.py"]