## Prerequisites

- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Optional: NVIDIA GPU drivers and Docker NVIDIA runtime for GPU support

## Build the Docker Image

Clone the repository and navigate to the project folder:

```bash
git clone https://github.com/FabianDubach/MLOPS.git
cd MLOPS/Project2
docker build -t train_glue_transformer .
```

## Running the Training Script

### Using CPU
```bash
docker run --rm --env-file .env train_glue_transformer
```

### Using GPU
```bash
docker run --rm --gpus all --env-file .env train_glue_transformer
```

### Adjusting Hyperparameters

You can override defaults by passing arguments:
```bash
docker run --rm --env-file .env train_glue_transformer \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --max_seq_length 128 \
    --learning_rate 2e-5
```

### Environment Variables

Create a .env file in the same folder for your WandB API key and other environment variables:
```bash
WANDB_API_KEY=your_api_key_here
```

Important: Make sure .env is not committed to the repository. Otherwise people can just access your WandB.

### Notes

- The script defaults to training the mrpc task of the GLUE benchmark.
- Training uses PyTorch Lightning with HuggingFace Transformers.
- Reduce max_seq_length or batch size if you encounter memory issues.