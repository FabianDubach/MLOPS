# MRPC DistilBERT: Containerized Training

This repository provides a reproducible workflow for training **DistilBERT** on the **GLUE MRPC** dataset.  
It supports both **local** and **Docker-based** execution and logs all experiments to **Weights & Biases**:  
[W&B Project Dashboard](https://wandb.ai/nevinhelfenstein-hslu-mlops/containerization?nw=nwusernevinhelfenstein)

---

## Folder Structure
```
.
├── src/                       # model and data modules
├── main.py                    # CLI entry point
├── Dockerfile.cpu             # CPU-only Dockerfile
├── requirements-*.txt         # dependency pins
├── execution_scripts/         # build/run helper scripts
├── models/                    # training outputs (not in the git but will be created during execution)
└── README.md
```

---

## 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/HSLU.MLOps.Containerization.git
cd HSLU.MLOps.Containerization
```

---

## 2. Configure Weights & Biases
Create a `.env` file in the project root:
```bash
WANDB_API_KEY=your_wandb_api_key
```

All training runs will automatically log to W&B if this key is provided.  
To **disable W&B logging**, add the CLI flag `--no_wandb` when launching a run:
```bash
python main.py --no_wandb ...
```

**Behavior summary:**
- The execution scripts in `execution_scripts/` **enable W&B logging by default**.
- If you run the Docker image directly (without those scripts), W&B logging is **disabled by default** inside the image.
- You can always override the behavior with the `--no_wandb` or `--run_name` CLI flags.

---

## 3. Build the Docker Image
**CPU image**:
```bash
docker build -t mrpc-cpu -f Dockerfile.cpu .
```
---

## 4. Run Training

### Local (no Docker)
```bash
python main.py   --checkpoint_dir models   --epochs 3   --seed 42   --train_batch_size 32   --eval_batch_size 64   --optimizer_name adamw   --lr 1e-4   --lr_scheduler_type linear   --warmup_ratio 0.11   --weight_decay 0.015   --run_name local
```

### Local Docker
```bash
bash execution_scripts/build_and_run_docker.sh
```
This script rebuilds the image and runs a containerized training job with `--run_name local_docker`.

### GitHub Codespaces
Run the same image in the cloud:
```bash
docker build -t mrpc-cpu -f Dockerfile.cpu .
docker run --rm --env-file .env -v "$PWD/models:/app/models" mrpc-cpu   python main.py --checkpoint_dir models     --epochs 3 --seed 42     --train_batch_size 32 --eval_batch_size 64     --optimizer_name adamw --lr 1e-4     --lr_scheduler_type linear --warmup_ratio 0.11 --weight_decay 0.015     --run_name github_codespace
```

---

## 5. Output and Model Reuse
Trained models are stored in:
```
models/<run_name>/
```
Load them in Python:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained("models/<run_name>")
tokenizer = AutoTokenizer.from_pretrained("models/<run_name>")
```

---

## 6. Notes
- Defaults match the best configuration from Project 1:  
  epochs = 3, seed = 42, train/eval batch = 32/64, AdamW, linear scheduler, lr = 1e-4, warmup_ratio = 0.11, weight_decay = 0.015.  
- Logs automatically to W&B if `WANDB_API_KEY` is provided.  
- Use `--no_wandb` to disable logging explicitly.    


