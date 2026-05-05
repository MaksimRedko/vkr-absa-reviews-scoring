from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.sentiment.common import DEFAULT_RUN_DIR, MODE_B, run_mode


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run sentiment benchmark mode B (sentence evidence)")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--dataset", default="")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--window-tokens", type=int, default=6)
    args = parser.parse_args(argv)

    out_dir = run_mode(
        MODE_B,
        run_dir=args.run_dir,
        dataset_path=args.dataset or None,
        window_tokens=args.window_tokens,
        relevance_threshold=args.threshold,
        temperature=args.temperature,
    )
    print(out_dir)


if __name__ == "__main__":
    main()
