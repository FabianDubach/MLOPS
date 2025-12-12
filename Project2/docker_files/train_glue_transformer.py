import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import datasets
import evaluate
import lightning as L
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


# ============================================================================
# DATA MODULE
# ============================================================================

class GLUEDataModule(L.LightningDataModule):
    """DataModule for GLUE benchmark tasks"""
    
    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = "mrpc",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        self.dataset = datasets.load_dataset("glue", self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        datasets.load_dataset("glue", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        # num_workers=0 is critical for Docker/Codespaces to avoid multiprocessing memory issues
        return DataLoader(
            self.dataset["train"], 
            batch_size=self.train_batch_size, 
            shuffle=True,
            num_workers=0
        )

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(
                self.dataset["validation"], 
                batch_size=self.eval_batch_size,
                num_workers=0
            )
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(self.dataset[x], batch_size=self.eval_batch_size, num_workers=0) 
                for x in self.eval_splits
            ]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(
                self.dataset["test"], 
                batch_size=self.eval_batch_size,
                num_workers=0
            )
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(self.dataset[x], batch_size=self.eval_batch_size, num_workers=0) 
                for x in self.eval_splits
            ]

    def convert_to_features(self, example_batch, indices=None):
        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, padding="max_length", truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features


# ============================================================================
# MODEL
# ============================================================================

class GLUETransformer(L.LightningModule):
    """Transformer model for GLUE tasks"""
    
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        learning_rate: float,
        warmup_steps: int,
        weight_decay: float,
        adam_epsilon: float,
        dropout_rate: float,
        train_batch_size: int,
        eval_batch_size: int,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        # Load config and set dropout
        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.config.attention_probs_dropout_prob = dropout_rate
        self.config.hidden_dropout_prob = dropout_rate

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metric = evaluate.load(
            "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )

        # Only store validation outputs - don't accumulate training outputs to save memory
        self.validation_step_outputs = []

        # Track best validation accuracy
        self.best_val_accuracy = 0.0

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        
        # Log loss for each step without storing outputs (memory optimization)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # Return loss directly without accumulating predictions to save memory
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]
        self.validation_step_outputs.append({"loss": val_loss, "preds": preds, "labels": labels})
        return val_loss

    def on_validation_epoch_end(self):
        if self.hparams.task_name == "mnli":
            for i, output in enumerate(self.validation_step_outputs):
                # matched or mismatched
                split = self.hparams.eval_splits[i].split("_")[-1]
                preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in output]).mean()
                self.log(f"val_loss_{split}", loss, prog_bar=True)
                split_metrics = {
                    f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
                }
                self.log_dict(split_metrics, prog_bar=True)
            self.validation_step_outputs.clear()
            return

        preds = torch.cat([x["preds"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()

        # Calculate validation metrics
        val_metrics = self.metric.compute(predictions=preds, references=labels)
        val_accuracy = val_metrics.get('accuracy', val_metrics.get('f1', 0.0))

        # Update best validation accuracy
        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy

        # Log validation metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", val_accuracy, prog_bar=True)
        self.log("best_val_accuracy", self.best_val_accuracy, prog_bar=True)
        self.log_dict(val_metrics, prog_bar=True)

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train GLUE model with hyperparameter tuning")
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="distilbert-base-uncased",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="mrpc",
        choices=["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli", "ax"],
    )
    
    # Training hyperparameters
    parser.add_argument("--lr", "--learning_rate", type=float, default=2e-5, dest="learning_rate")
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    
    # Data arguments - CHANGED DEFAULT from 256 to 128 to reduce memory usage
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=128, 
                        help="Maximum sequence length (default: 128, was 256)")
    
    # Training arguments
    parser.add_argument("--epochs", "--num_epochs", type=int, default=3, dest="num_epochs")
    parser.add_argument("--seed", type=int, default=42)
    
    # Output arguments
    parser.add_argument("--checkpoint_dir", type=str, default="models")
    
    # Wandb arguments
    parser.add_argument("--wandb_project", type=str, default="HyperparameterTuning")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--log_model", action="store_true")
    
    # Device arguments
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    
    return parser.parse_args()


def create_run_name(args):
    """Generate a descriptive run name from hyperparameters"""
    return (
        f"{args.model_name_or_path.split('/')[-1]}_"
        f"lr{args.learning_rate}_"
        f"wd{args.weight_decay}_"
        f"warmup{args.warmup_steps}_"
        f"eps{args.adam_epsilon}_"
        f"dropout{args.dropout_rate}_"
        f"maxseq{args.max_seq_length}_"
        f"gradclip{args.gradient_clip_val}"
    )


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main training function"""
    args = parse_args()
    
    # Set seed for reproducibility
    L.seed_everything(args.seed)
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data module
    print(f"Loading data for task: {args.task_name}")
    dm = GLUEDataModule(
        model_name_or_path=args.model_name_or_path,
        task_name=args.task_name,
        max_seq_length=args.max_seq_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )
    dm.setup("fit")
    
    # Initialize model
    print(f"Initializing model: {args.model_name_or_path}")
    model = GLUETransformer(
        model_name_or_path=args.model_name_or_path,
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        adam_epsilon=args.adam_epsilon,
        dropout_rate=args.dropout_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )
    
    # Create wandb run name
    run_name = args.wandb_run_name if args.wandb_run_name else create_run_name(args)
    
    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=run_name,
        log_model=args.log_model,
    )
    
    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=args.num_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=wandb_logger,
        gradient_clip_val=args.gradient_clip_val,
        default_root_dir=str(checkpoint_dir),
    )
    
    # Train
    print(f"Starting training for {args.num_epochs} epochs...")
    print(f"Hyperparameters:")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Dropout rate: {args.dropout_rate}")
    print(f"  Gradient clip: {args.gradient_clip_val}")
    print(f"  Max seq length: {args.max_seq_length}")
    print(f"  Batch size: {args.train_batch_size}")
    
    trainer.fit(model, datamodule=dm)
    
    print(f"\nTraining complete!")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Best validation accuracy: {model.best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()