import argparse
import os
from dotenv import load_dotenv
import wandb
from src.train import launch_run

DEFAULT_WANDB = {
    "PROJECT": "containerization",
    "ENTITY": "nevinhelfenstein-hslu-mlops",
    "GROUP": "containerization",
}

def setup_wandb_defaults():
    """Load .env and initialize default W&B environment."""
    load_dotenv()

    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        raise ValueError("WANDB_API_KEY not found. Please set it in your .env file.")

    wandb.login(key=api_key)
    print(
        f"W&B configured for project='{DEFAULT_WANDB['PROJECT']}', "
        f"entity='{DEFAULT_WANDB['ENTITY']}', group='{DEFAULT_WANDB['GROUP']}'"
    )

def build_parser():
    p = argparse.ArgumentParser(description="Train DistilBERT on GLUE MRPC with Lightning.")

    p.add_argument("--checkpoint_dir", type=str, default="models", help="Where to save checkpoints.")
    p.add_argument("--project", type=str, default=DEFAULT_WANDB["PROJECT"])
    p.add_argument("--entity", type=str, default=DEFAULT_WANDB["ENTITY"])
    p.add_argument("--group", type=str, default=DEFAULT_WANDB["GROUP"])
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging.")

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--max_seq_length", type=int, default=128)
    p.add_argument("--grad_accum", type=int, default=1)
    
    p.add_argument("--optimizer_name", type=str, default="adamw", choices=["adamw", "adam"])
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.015)
    p.add_argument("--lr_scheduler_type", type=str, default="linear", choices=["linear", "cosine", "constant"])
    p.add_argument("--warmup_ratio", type=float, default=0.11)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)

    return p

def main():
    args = build_parser().parse_args()

    if not args.no_wandb:
        setup_wandb_defaults()

    launch_run(
        checkpoint_dir=args.checkpoint_dir,
        project=args.project,
        entity=args.entity,
        group=args.group,
        run_name=args.run_name,
        epochs=args.epochs,
        seed=args.seed,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        max_seq_length=args.max_seq_length,
        grad_accum=args.grad_accum,
        optimizer_name=args.optimizer_name,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        log_to_wandb=not args.no_wandb,
    )

if __name__ == "__main__":
    main()
