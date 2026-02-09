"""ConFixMLP model and training utilities."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score


class ConFixMLP(nn.Module):
    """Multi-layer perceptron for encrypted traffic classification.

    Architecture: [input] → 512 → 256 → 128 → [num_classes]
    with GELU activations, BatchNorm, and Dropout(0.3).
    """

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    epochs: int = 100,
    warmup_epochs: int = 5,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Build AdamW optimizer with cosine-annealing + linear warmup scheduler."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup_epochs, 1)
    )
    return optimizer, scheduler


def _warmup_lr(optimizer, epoch: int, warmup_epochs: int, base_lr: float):
    """Linearly scale learning rate during warmup."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for pg in optimizer.param_groups:
            pg["lr"] = lr


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device | str = "cpu",
) -> dict:
    """Train for one epoch, return dict with 'loss', 'accuracy'."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    return {
        "loss": total_loss / max(n, 1),
        "accuracy": accuracy_score(all_labels, all_preds),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str = "cpu",
) -> dict:
    """Evaluate model, return dict with 'accuracy' and 'macro_f1'."""
    model.eval()
    all_preds, all_labels = [], []

    for features, labels in loader:
        features = features.to(device)
        logits = model(features)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 200,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 5,
    patience: int = 15,
    device: torch.device | str = "cpu",
    class_weights: torch.Tensor | None = None,
) -> dict:
    """Full training loop with early stopping.

    Parameters
    ----------
    class_weights : optional tensor of per-class weights for CrossEntropyLoss.
        Computed externally (e.g., inverse class frequency) for imbalanced datasets.

    Returns dict with 'accuracy', 'macro_f1', 'best_epoch'.
    """
    model = model.to(device)
    optimizer, scheduler = build_optimizer(model, lr, weight_decay, epochs, warmup_epochs)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = 0.0
    best_epoch = 0
    best_metrics = {}
    wait = 0

    for epoch in range(epochs):
        _warmup_lr(optimizer, epoch, warmup_epochs, lr)
        train_one_epoch(model, train_loader, optimizer, criterion, device)

        if epoch >= warmup_epochs:
            scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_metrics = val_metrics
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    best_metrics["best_epoch"] = best_epoch
    return best_metrics
