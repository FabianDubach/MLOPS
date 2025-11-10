#!/usr/bin/env bash
set -euo pipefail

echo "=== Building Docker image (mrpc-cpu) ==="
docker build --no-cache -t mrpc-cpu -f Dockerfile.cpu .

echo "=== Running containerized training run ==="
docker run --rm \
  --env-file .env \
  -v "$PWD/models:/app/models" \
  mrpc-cpu \
  python main.py --checkpoint_dir models \
    --epochs 3 --seed 42 \
    --train_batch_size 32 --eval_batch_size 64 \
    --optimizer_name adamw --lr 1e-4 \
    --lr_scheduler_type linear --warmup_ratio 0.11 --weight_decay 0.015 \
    --run_name local_docker

echo "=== Training finished. Model artifacts saved in ./models ==="
