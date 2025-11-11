# Project 2: Containerization

## Prerequisites

- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))


## Clone Repository

Clone the repository:

```bash
git clone https://github.com/FabianDubach/MLOPS.git
```


### Environment Variables

Create a .env file in the same folder for your WandB API key and other environment variables:

```bash
WANDB_API_KEY=your_api_key_here
```

Important: Make sure .env is not committed to the repository. Otherwise people can just access your WandB.


## Build Docker Image

Navigate to the project folder and build the docker image:

```bash
cd MLOPS/Project2/docker_files
docker build -t train_glue_transformer .
```


## Running the Training Script

You can run the script with this command:

```bash
docker run --rm --env-file .env train_glue_transformer
```

### Adjusting Hyperparameters

You can override all defaults by passing arguments:

```bash
docker run --rm --env-file .env train_glue_transformer \
    --model_name_or_path distilbert-base-uncased \
    --task_name mrpc \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --max_seq_length 128 \
    --learning_rate 2e-5 \
    --weight_decay 0 \
    --adam_epsilon 1e-8 \
    --warmup_steps 50 \
    --dropout_rate 0.1 \
    --gradient_clip_val 1.0 \
    --epochs 3 \
    --seed 42 \
    --accelerator auto \
    --devices 1 \
    --wandb_project HyperparameterTuning
```


## Notes

- CPU only script