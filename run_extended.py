"""Run the reviewer-driven controlled extensions (correlation, groups, gradual inclusion, PCA, residuals).

Existing baseline outputs are not overwritten: new files use new names (extended_*, 14_..20_ figures).
"""

from __future__ import annotations

import argparse

from src.extended_analysis import run_extended


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_extended(n_boot=args.n_boot, seed=args.seed)


if __name__ == "__main__":
    main()
