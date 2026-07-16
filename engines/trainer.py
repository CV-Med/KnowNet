import math
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional, Dict
from utils.logger import Logger
from utils.metrics import DiceScore, region_dice
from utils.experiment_recorder import ExperimentRecorder
from losses.loss import DeepSupervisionLoss


class WarmupCosine(LambdaLR):
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=1e-6):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(eta_min, 0.5 * (1.0 + math.cos(math.pi * progress)))
        super().__init__(optimizer, lr_lambda)


class WarmupPoly(LambdaLR):
    def __init__(self, optimizer, warmup_steps, total_steps, power=0.9, eta_min=1e-6):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.power = power
        self.eta_min = eta_min
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(eta_min, (1.0 - progress) ** power)
        super().__init__(optimizer, lr_lambda)


class Trainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 config: dict, recorder: Optional[ExperimentRecorder] = None,
                 checkpoint_dir: str = "weights"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.recorder = recorder
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device(config.get("device", "cuda:0"))

        training_config = config.get("training", {})
        self.epochs = training_config.get("epochs", 200)
        self.lr = training_config.get("learning_rate", 1e-4)
        self.weight_decay = training_config.get("weight_decay", 1e-4)
        self.gradient_clip = training_config.get("gradient_clip", 1.0)
        self.early_stop_patience = training_config.get("early_stop_patience", 30)
        self.accum_steps = training_config.get("accum_steps", 1)
        self.val_every = training_config.get("val_every", 5)
        self.warmup_epochs = training_config.get("warmup_epochs", 10)

        self.num_classes = config["data"]["num_classes"]
        self.criterion = DeepSupervisionLoss(
            num_classes=self.num_classes,
            main_weight=config["loss"]["main_weight"],
            aux_weights=config["loss"]["aux_weights"],
        )
        self.metric_fn = DiceScore(num_classes=self.num_classes)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        total_steps = self.epochs * len(train_loader) // self.accum_steps
        warmup_steps = self.warmup_epochs * len(train_loader) // self.accum_steps
        sched_name = training_config.get("scheduler", "poly")
        sched_kwargs = training_config.get("scheduler_kwargs", {})
        if sched_name == "cosine":
            self.scheduler = WarmupCosine(
                self.optimizer, warmup_steps, total_steps, eta_min=sched_kwargs.get("eta_min", 1e-6)
            )
        else:
            self.scheduler = WarmupPoly(
                self.optimizer, warmup_steps, total_steps,
                power=sched_kwargs.get("power", 0.9), eta_min=sched_kwargs.get("eta_min", 1e-6)
            )
        self.logger = Logger(log_dir="./logs", name="train")

        self.best_metric = 0.0
        self.best_epoch = 0
        self.best_val_metrics = {}
        self.early_stop_counter = 0
        self.loss_history = []

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_norm = 0.0
        num_batches = 0
        self.optimizer.zero_grad()

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)

            logits, aux_outputs, uncertainty = self.model(images)
            loss = self.criterion(logits, aux_outputs, targets, uncertainty)
            loss = loss / self.accum_steps
            loss.backward()

            per_step_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip
            )

            if (batch_idx + 1) % self.accum_steps == 0:
                accum_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
                total_norm += accum_norm.item()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                num_batches += 1

            total_loss += loss.item() * self.accum_steps

        avg_loss = total_loss / len(self.train_loader)
        avg_norm = total_norm / max(1, num_batches)
        return {"loss": avg_loss, "grad_norm": avg_norm}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        dice_scores = []
        all_pred, all_target = [], []
        for images, targets in self.val_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)
            logits, _, _ = self.model(images)
            pred = logits.argmax(dim=1)
            dice_scores.append(self.metric_fn(pred, targets))
            all_pred.append(pred)
            all_target.append(targets)
        avg_dice = torch.stack(dice_scores).mean(dim=0)
        all_pred = torch.cat(all_pred, dim=0)
        all_target = torch.cat(all_target, dim=0)
        reg = region_dice(all_pred, all_target)
        return {
            "dice": avg_dice.mean().item(),
            "dice_ET": reg["ET"],
            "dice_WT": reg["WT"],
            "dice_TC": reg["TC"],
            "dice_NEC": avg_dice[1].item(),
            "dice_ED": avg_dice[2].item(),
        }

    def fit(self, epochs: Optional[int] = None, load_checkpoint: bool = True):
        if epochs is not None:
            self.epochs = epochs
        os.makedirs("logs", exist_ok=True)

        if load_checkpoint:
            best_ckpt = os.path.join(self.checkpoint_dir, "best_checkpoint.pth")
            if os.path.exists(best_ckpt):
                state = torch.load(best_ckpt, map_location=self.device)
                self.model.load_state_dict(state)
                self.logger.info(f"Auto-loaded checkpoint from {best_ckpt}")
            else:
                self.logger.info("No checkpoint found, training from scratch")
        else:
            self.logger.info("Training from scratch (checkpoint loading disabled)")

        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_epoch()
            lr_current = self.scheduler.get_last_lr()[0]
            epoch_log = {"epoch": epoch, "train_loss": train_metrics["loss"]}
            msg = (
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"Loss: {train_metrics['loss']:.4f} | "
                f"GN: {train_metrics.get('grad_norm', 0):.2f} | "
                f"LR: {lr_current:.2e}"
            )

            if epoch % self.val_every == 0:
                val_metrics = self.validate()
                current_dice = val_metrics["dice"]
                epoch_log["val_loss"] = current_dice
                epoch_log.update({k: v for k, v in val_metrics.items() if k != "dice"})
                msg += (
                    f" | Dice: {current_dice:.4f}"
                    f"  WT:{val_metrics['dice_WT']:.3f}"
                    f" TC:{val_metrics['dice_TC']:.3f}"
                    f" ET:{val_metrics['dice_ET']:.3f}"
                )
                if current_dice > self.best_metric:
                    self.best_metric = current_dice
                    self.best_epoch = epoch
                    self.best_val_metrics = val_metrics
                    self.early_stop_counter = 0
                    torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "best_checkpoint.pth"))
                else:
                    self.early_stop_counter += 1
            else:
                msg += " | Val: skipped"

            self.loss_history.append(epoch_log)
            with open("logs/losses.json", "w") as f:
                json.dump(self.loss_history, f, indent=2)

            self.logger.info(msg)

            latest_ckpt = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": epoch,
                "best_metric": self.best_metric,
                "best_epoch": self.best_epoch,
                "best_val_metrics": self.best_val_metrics,
            }
            torch.save(latest_ckpt, os.path.join(self.checkpoint_dir, "latest_checkpoint.pth"))

            if self.early_stop_counter >= self.early_stop_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        if self.recorder is not None:
            self.recorder.record_metric("best_dice", self.best_metric)
            for k, v in self.best_val_metrics.items():
                if k.startswith("dice_"):
                    self.recorder.record_metric(f"best_{k}", v)
            self.recorder.record_failure(
                scenario="training_diverged",
                effect="NaN at epoch 103 in round 3",
                analysis="Gradient conflict in deep supervision path; mitigated by per-step + accum dual clipping",
            )
            self.recorder.save()
