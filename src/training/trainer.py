"""
Training loop for brain tumor segmentation models.

Handles: training step, validation step (with sliding-window inference),
best-checkpoint saving, CSV logging, optional W&B logging, and seed setting
for reproducibility.
"""

import os
import csv
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Compose


def set_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic mode (slightly slower, but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    """
    Encapsulates the full training loop for a segmentation model.

    Logs train loss, val loss, and val Dice per epoch to CSV. Saves the
    best-validation-Dice checkpoint. Supports optional Weights & Biases logging.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        num_classes: int = 1,
        learning_rate: float = 1e-4,
        max_epochs: int = 20,
        patch_size: tuple = (128, 128, 128),
        sw_batch_size: int = 4,
        output_dir: str = "outputs",
        run_name: str = "run",
        use_wandb: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        self.max_epochs = max_epochs
        self.patch_size = patch_size
        self.sw_batch_size = sw_batch_size
        self.run_name = run_name
        self.use_wandb = use_wandb

        # Loss and optimizer
        self.loss_fn = DiceCELoss(
            to_onehot_y=(num_classes > 1),
            softmax=(num_classes > 1),
            sigmoid=(num_classes == 1),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Dice metric for validation
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")

        # Post-processing for predictions and labels before computing Dice
        if num_classes == 1:
            self.post_pred = Compose([AsDiscrete(threshold=0.5)])
            self.post_label = Compose([])  # already 0/1
        else:
            self.post_pred = Compose([AsDiscrete(argmax=True, to_onehot=num_classes)])
            self.post_label = Compose([AsDiscrete(to_onehot=num_classes)])

        # Output paths for checkpoints and logs
        self.checkpoint_dir = Path(output_dir) / "checkpoints"
        self.log_dir = Path(output_dir) / "logs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / f"{run_name}.csv"

        # Track best validation Dice for best-checkpoint saving
        self.best_val_dice = -1.0

        # Initialize W&B if requested
        if use_wandb:
            import wandb
            self.wandb = wandb
        else:
            self.wandb = None

    def _train_epoch(self) -> float:
        """Run one training epoch. Returns average training loss."""
        self.model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.loss_fn(logits, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        return running_loss / max(n_batches, 1)

    def _validate(self) -> Dict[str, float]:
        """Run validation using sliding-window inference. Returns val_loss and val_dice."""
        self.model.eval()
        running_loss = 0.0
        n_batches = 0
        self.dice_metric.reset()

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                # Sliding-window inference: full-volume evaluation
                logits = sliding_window_inference(
                    inputs=images,
                    roi_size=self.patch_size,
                    sw_batch_size=self.sw_batch_size,
                    predictor=self.model,
                    overlap=0.25,
                )

                loss = self.loss_fn(logits, labels)
                running_loss += loss.item()
                n_batches += 1

                # Convert to discrete predictions for Dice
                pred = self.post_pred(logits)
                gt = self.post_label(labels)
                self.dice_metric(y_pred=pred, y=gt)

        val_loss = running_loss / max(n_batches, 1)
        val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        return {"val_loss": val_loss, "val_dice": val_dice}

    def _save_checkpoint_if_best(self, val_dice: float, epoch: int) -> None:
        """Save the model if it beats the previous best validation Dice."""
        if val_dice > self.best_val_dice:
            self.best_val_dice = val_dice
            ckpt_path = self.checkpoint_dir / f"{self.run_name}_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "val_dice": val_dice,
                },
                ckpt_path,
            )
            print(f"  Saved best checkpoint: val_dice={val_dice:.4f}")

    def _log_epoch(self, epoch: int, train_loss: float, val_loss: float, val_dice: float) -> None:
        """Append a row to the CSV log; also log to W&B if enabled."""
        write_header = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["epoch", "train_loss", "val_loss", "val_dice"])
            writer.writerow([epoch, train_loss, val_loss, val_dice])

        if self.wandb is not None:
            self.wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_dice,
            })

    def run(self) -> Dict[str, float]:
        """Run the full training loop. Returns final summary stats."""
        print(f"Starting training: {self.run_name}, {self.max_epochs} epochs.")
        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_epoch()
            val_metrics = self._validate()
            val_loss = val_metrics["val_loss"]
            val_dice = val_metrics["val_dice"]

            print(
                f"Epoch {epoch:3d}/{self.max_epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_dice={val_dice:.4f}"
            )

            self._log_epoch(epoch, train_loss, val_loss, val_dice)
            self._save_checkpoint_if_best(val_dice, epoch)

        print(f"Training complete. Best val_dice: {self.best_val_dice:.4f}")
        return {"best_val_dice": self.best_val_dice}