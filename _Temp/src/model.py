from typing import Optional
import torch
from sklearn.metrics import accuracy_score, f1_score
import lightning as L
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)

class GLUETransformer(L.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        optimizer_name: str = "adamw",
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        lr_scheduler_type: str = "linear",
        warmup_steps: int = 0,
        warmup_ratio: float | None = None,
        train_batch_size: int = 32,
        eval_batch_size: int = 64,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, config=self.config
        )
        self.validation_step_outputs = []

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        return outputs[0]  # loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        else:
            preds = logits.squeeze()

        labels = batch["labels"]
        self.validation_step_outputs.append({"loss": val_loss, "preds": preds, "labels": labels})
        return val_loss

    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) == 0:
            return
        preds = torch.cat([x["preds"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()

        acc = float(accuracy_score(labels, preds))
        f1 = float(f1_score(labels, preds))

        self.log("val_loss", loss, prog_bar=True)
        self.log("accuracy", acc, prog_bar=True)
        self.log("f1", f1, prog_bar=True)

        self.validation_step_outputs.clear()
        
    def configure_optimizers(self):
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
        betas = (self.hparams.beta1, self.hparams.beta2)
        if self.hparams.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                betas=betas,
                eps=self.hparams.eps,
            )
        else:
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                betas=betas,
                eps=self.hparams.eps,
            )

        warmup_steps = self.hparams.warmup_steps
        if self.hparams.warmup_ratio is not None and self.trainer is not None:
            warmup_steps = int(self.hparams.warmup_ratio * self.trainer.estimated_stepping_batches)

        sched_type = (self.hparams.lr_scheduler_type or "linear").lower()
        if sched_type == "cosine":
            sched = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=self.trainer.estimated_stepping_batches
            )
        elif sched_type == "constant":
            sched = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps
            )
        else:
            sched = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=self.trainer.estimated_stepping_batches
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }
