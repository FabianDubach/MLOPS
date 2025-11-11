# Memory-optimized Dockerfile for GitHub Codespaces
# Key optimizations: CPU-only PyTorch, model pre-caching, tokenizer parallelism disabled

FROM python:3.11-slim

# Set environment variables EARLY to prevent memory issues
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Copy requirements
COPY requirements.txt .

# Install CPU-only PyTorch first (much smaller than CUDA version)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY train_glue_transformer.py .

# PRE-CACHE the model and tokenizer during build to avoid runtime memory spikes
RUN python - <<'PY'
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = 'distilbert-base-uncased'
print(f'Pre-caching {model_name}...')
AutoTokenizer.from_pretrained(model_name, use_fast=True)
AutoModelForSequenceClassification.from_pretrained(model_name)
print(f'Successfully cached {model_name}')
PY

# Remove build dependencies to reduce final image size
RUN apt-get purge -y --auto-remove build-essential

# Default command with memory-optimized parameters
CMD ["python", "train_glue_transformer.py"]