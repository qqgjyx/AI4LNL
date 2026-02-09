#!/usr/bin/env python
"""Module 1: graph → community detection → label correction."""

import argparse
import numpy as np
import yaml

from confix.data import load_features, inject_noise
from confix.graph import build_knn_graph, reweight_sgtsne
from confix.community import parallel_dt
from confix.correction import (
    compute_label_distribution,
    compute_label_consistency,
    compute_merge_split_consistency,
    consensus_vote,
    correct_labels,
)
from confix.metrics import repair_rate


def main():
    parser = argparse.ArgumentParser(description="Confix label correction pipeline")
    parser.add_argument("--data", required=True, help="Path to data file (CSV or NPZ)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--noise-type", default=None, help="Override noise type")
    parser.add_argument("--noise-rate", type=float, default=None)
    parser.add_argument("--output", default="corrected_labels.npz")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    X, y_clean = load_features(args.data)
    noise_type = args.noise_type or cfg["noise"]["type"]
    noise_rate = args.noise_rate if args.noise_rate is not None else cfg["noise"]["rate"]

    y_noisy, noise_mask = inject_noise(y_clean, noise_type, noise_rate, seed=args.seed)
    print(f"Injected {noise_type} noise at rate {noise_rate:.1%} — {noise_mask.sum()} labels flipped")

    adj = build_knn_graph(X, k=cfg["graph"]["k"])
    adj = reweight_sgtsne(adj)
    print(f"Built KNN graph: {adj.shape[0]} nodes, {adj.nnz} edges")

    partitions = parallel_dt(
        adj,
        gamma_range=tuple(cfg["community"]["gamma_range"]),
        n_resolutions=cfg["community"]["n_resolutions"],
        seed=args.seed,
    )
    print(f"Detected communities at {len(partitions)} resolutions")

    dist = compute_label_distribution(partitions, y_noisy)
    lc = compute_label_consistency(dist)
    msc = compute_merge_split_consistency(partitions)
    consensus = consensus_vote(dist, lc, msc)
    y_corrected, corrected_mask = correct_labels(
        y_noisy, consensus,
        theta_low=cfg["correction"]["theta_low"],
        delta=cfg["correction"]["delta"],
    )

    rr = repair_rate(y_clean, y_noisy, y_corrected)
    print(f"Corrected {corrected_mask.sum()} labels — repair rate: {rr:.3f}")

    np.savez(args.output, y_corrected=y_corrected, y_clean=y_clean, y_noisy=y_noisy)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
