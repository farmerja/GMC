"""CLI for the standalone hierarchical steady-2D training workflow."""

from __future__ import annotations

import argparse

from .training import HierarchicalTrainingConfig, run_hierarchical_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train standalone hierarchical steady-2D models.")
    parser.add_argument("--source", choices=("internal", "boundary"), default="internal")
    parser.add_argument("--num-neutrons", type=int, default=100000)
    parser.add_argument("--sigma-t", type=float, default=1.0)
    parser.add_argument("--distribution", type=int, default=0)
    parser.add_argument("--min-width", type=float, default=0.01)
    parser.add_argument("--max-width", type=float, default=10.0)
    parser.add_argument("--min-height", type=float, default=0.01)
    parser.add_argument("--max-height", type=float, default=10.0)
    parser.add_argument("--sampling", choices=("random", "qmc", "lhs"), default="random")
    parser.add_argument("--tasks", nargs="+", default=["all"], help="Subset of tasks: face joint/exit track")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", default="ml/hierarchical_steady_2d/runs")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = HierarchicalTrainingConfig(
        source=args.source,
        num_neutrons=args.num_neutrons,
        sigma_t=args.sigma_t,
        distribution=args.distribution,
        min_width=args.min_width,
        max_width=args.max_width,
        min_height=args.min_height,
        max_height=args.max_height,
        sampling=args.sampling,
        tasks=tuple(args.tasks),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
        seed=args.seed,
        device=args.device,
        output_root=args.output_root,
        run_name=args.run_name,
    )
    results = run_hierarchical_training(config)
    print(f"Hierarchical steady-2D training complete. Artifacts: {results['run_dir']}")


if __name__ == "__main__":
    main()
