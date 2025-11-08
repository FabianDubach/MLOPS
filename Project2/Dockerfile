# Use Python slim image fopr a smaller footprint
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies needed for PyTorch and transformers
# Keep this minimal to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirement list
COPY requirements.txt .

# Install CPU-only PyTorch and other dependencies
# Using CPU-only version significantly reduces image size
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy your script
COPY . .

# Default command — runs your Python script
# You can override args when running the container
CMD ["python", "train_glue_transformer.py"]