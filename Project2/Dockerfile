# Use an official PyTorch image with CUDA/CPU support (choose depending on your hardware)
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Set working directory inside container
WORKDIR /app

# Copy requirement list first and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your script
COPY . .

# Default command — runs your Python script
# You can override args when running the container
CMD ["python", "train_glue_transformer.py"]