#!/usr/bin/env python3
from __future__ import annotations

from ml.train_scatter_split_2d import train_scatter_split_2d


def main() -> None:
    common = dict(
        num_epochs=50,
        num_neutrons=500000,
        batch_size=2048,
        hidden_dim=128,
        depth=4,
        expansion_factor=2,
        sampling="random",
        seed=42,
    )
    train_scatter_split_2d(source_type="boundary", **common)
    print("=== SCATTER-SPLIT BOUNDARY DONE ===")
    train_scatter_split_2d(source_type="internal", **common)
    print("=== SCATTER-SPLIT INTERNAL DONE ===")


if __name__ == "__main__":
    main()
