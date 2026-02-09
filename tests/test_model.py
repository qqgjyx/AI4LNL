"""Tests for confix.model."""

import numpy as np
import torch

from confix.model import ConFixMLP, build_optimizer, train_one_epoch, evaluate, train
from confix.data import get_dataloader


def test_confix_mlp_forward():
    model = ConFixMLP(input_dim=10, num_classes=3)
    x = torch.randn(8, 10)
    out = model(x)
    assert out.shape == (8, 3)


def test_confix_mlp_custom_dropout():
    model = ConFixMLP(input_dim=10, num_classes=5, dropout=0.5)
    x = torch.randn(4, 10)
    out = model(x)
    assert out.shape == (4, 5)


def test_build_optimizer():
    model = ConFixMLP(input_dim=10, num_classes=3)
    opt, sched = build_optimizer(model, lr=1e-3, epochs=50)
    assert isinstance(opt, torch.optim.AdamW)
    assert sched is not None


def test_train_one_epoch(synthetic_features, synthetic_labels):
    model = ConFixMLP(input_dim=synthetic_features.shape[1], num_classes=3)
    loader = get_dataloader(synthetic_features, synthetic_labels, batch_size=16, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    metrics = train_one_epoch(model, loader, opt, criterion)
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert metrics["loss"] > 0


def test_evaluate(synthetic_features, synthetic_labels):
    model = ConFixMLP(input_dim=synthetic_features.shape[1], num_classes=3)
    loader = get_dataloader(synthetic_features, synthetic_labels, batch_size=16, shuffle=False)
    metrics = evaluate(model, loader)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert 0 <= metrics["accuracy"] <= 1


def test_train_loop(synthetic_features, synthetic_labels):
    model = ConFixMLP(input_dim=synthetic_features.shape[1], num_classes=3)
    train_loader = get_dataloader(synthetic_features, synthetic_labels, batch_size=16)
    val_loader = get_dataloader(synthetic_features, synthetic_labels, batch_size=16, shuffle=False)
    result = train(model, train_loader, val_loader, epochs=3, patience=2)
    assert "accuracy" in result
    assert "macro_f1" in result
    assert "best_epoch" in result
