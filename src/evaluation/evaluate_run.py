from __future__ import annotations

import argparse

from src.evaluation.metrics_overall import write_metrics_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate traced or legacy e2e ABSA run")
    parser.add_argument("run_dir")
    args = parser.parse_args()
    metrics = write_metrics_report(args.run_dir)
    for key, value in sorted(metrics.items()):
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
