#!/usr/bin/env python
"""Module 2: augment → train ConFixMLP → evaluate."""

import argparse
import numpy as np
import yaml

from confix.data import load_features, augment_with_configurations, get_dataloader
from confix.graph import build_knn_graph, reweight_sgtsne
from confix.community import parallel_dt
from confix.model import ConFixMLP, train
from confix.metrics import accuracy, macro_f1


def main():
    parser = argparse.ArgumentParser(description="Confix training pipeline")
    parser.add_argument("--data", required=True, help="Path to data file (CSV or NPZ)")
    parser.add_argument("--labels", default=None, help="Path to corrected labels (NPZ)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    X, y = load_features(args.data)

    if args.labels:
        data = np.load(args.labels)
        y = data["y_corrected"]
        print(f"Loaded corrected labels from {args.labels}")

    # Build graph and communities for augmentation
    adj = build_knn_graph(X, k=cfg["graph"]["k"])
    adj = reweight_sgtsne(adj)
    partitions = parallel_dt(
        adj,
        gamma_range=tuple(cfg["community"]["gamma_range"]),
        n_resolutions=cfg["community"]["n_resolutions"],
        seed=args.seed,
    )
    X_aug = augment_with_configurations(X, partitions, embed_dim=cfg["augmentation"]["embed_dim"])
    print(f"Augmented features: {X.shape[1]} → {X_aug.shape[1]}")

    # Train/val split (80/20)
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(y))
    split = int(0.8 * len(y))
    train_idx, val_idx = idx[:split], idx[split:]

    tcfg = cfg["training"]
    train_loader = get_dataloader(X_aug[train_idx], y[train_idx], batch_size=tcfg["batch_size"])
    val_loader = get_dataloader(X_aug[val_idx], y[val_idx], batch_size=tcfg["batch_size"], shuffle=False)

    num_classes = int(y.max()) + 1
    model = ConFixMLP(X_aug.shape[1], num_classes, dropout=cfg["model"]["dropout"])

    result = train(
        model, train_loader, val_loader,
        epochs=tcfg["epochs"], lr=tcfg["lr"], weight_decay=tcfg["weight_decay"],
        warmup_epochs=tcfg["warmup_epochs"], patience=tcfg["patience"],
        device=args.device,
    )

    print(f"Best epoch: {result['best_epoch']}")
    print(f"Val accuracy: {result['accuracy']:.4f}")
    print(f"Val macro-F1: {result['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
