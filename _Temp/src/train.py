from __future__ import annotations
from typing import Optional
import os
import json
from datetime import datetime

import lightning as L
from lightning.pytorch.loggers import WandbLogger

from .data import GLUEDataModule
from .model import GLUETransformer

def make_run_name(**cfg):
    def sci(x):
        try: return f"{float(x):.0e}"
        except Exception: return str(x)
    def trim(x):
        s = f"{x}"
        return s.rstrip("0").rstrip(".") if "." in s else s
    parts = [
        f"opt_{cfg.get('optimizer_name','adamw')}",
        f"lr_{sci(cfg.get('learning_rate',1e-4))}",
        f"sched_{cfg.get('lr_scheduler_type','linear')}",
        f"wd_{trim(cfg.get('weight_decay',0.01))}",
        f"warmup_{trim(cfg.get('warmup_ratio',0.1))}",
        f"tb_{cfg.get('train_batch_size',32)}",
        f"ga_{cfg.get('grad_accum',1)}",
        f"ebs_{cfg.get('train_batch_size',32) * cfg.get('grad_accum',1)}",
    ]
    return "-".join(parts)

def launch_run(
    *,
    checkpoint_dir: str = "models",
    project: str = "mrpc-distilbert",
    entity: Optional[str] = None,
    group: Optional[str] = None,
    run_name: Optional[str] = None,
    epochs: int = 3,
    seed: int = 42,
    train_batch_size: int = 32,
    eval_batch_size: int = 64,
    max_seq_length: int = 128,
    grad_accum: int = 1,
    optimizer_name: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.015,
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.11,
    warmup_steps: int = 0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    log_to_wandb: bool = True,
):
    L.seed_everything(seed)

    cfg = dict(
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        max_seq_length=max_seq_length,
        grad_accum=grad_accum,
        epochs=epochs,
        seed=seed,
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    dm = GLUEDataModule(
        model_name_or_path="distilbert-base-uncased",
        task_name="mrpc",
        max_seq_length=cfg["max_seq_length"],
        train_batch_size=cfg["train_batch_size"],
        eval_batch_size=cfg["eval_batch_size"],
    )
    dm.setup("fit")

    model = GLUETransformer(
        model_name_or_path="distilbert-base-uncased",
        num_labels=dm.num_labels,
        task_name="mrpc",
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        optimizer_name=cfg["optimizer_name"],
        beta1=cfg["beta1"],
        beta2=cfg["beta2"],
        eps=cfg["eps"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_steps=cfg["warmup_steps"],
        warmup_ratio=cfg["warmup_ratio"],
        train_batch_size=cfg["train_batch_size"],
        eval_batch_size=cfg["eval_batch_size"],
    )

    auto_name = make_run_name(**cfg) if run_name is None else run_name

    if log_to_wandb:
        wandb_logger = WandbLogger(
            project=project, entity=entity, group=group,
            name=auto_name, log_model=False,
        )
        wandb_logger.experiment.config.update(
            {**cfg, "effective_batch_size": cfg["train_batch_size"] * cfg["grad_accum"]},
            allow_val_change=True,
        )
        logger = wandb_logger
    else:
        logger = None

    trainer = L.Trainer(
        max_epochs=cfg["epochs"],
        accelerator="auto",
        devices=1,
        accumulate_grad_batches=cfg["grad_accum"],
        logger=logger,
        num_sanity_val_steps=0,
        log_every_n_steps=25,
        enable_progress_bar=True,
        enable_checkpointing=True,
        default_root_dir=checkpoint_dir,
    )

    trainer.fit(model, datamodule=dm)

    save_dir = os.path.join(checkpoint_dir, auto_name)
    os.makedirs(save_dir, exist_ok=True)
    model.model.save_pretrained(save_dir)
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True).save_pretrained(save_dir)

    meta = {
        "save_dir": save_dir,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        **cfg,
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return save_dir
