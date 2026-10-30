"""Build and cache the train / val / test path banks for one (dynamics, product).

Run once per product before the training jobs (the slurm chain in submit_all.sh does this).
The training jobs also build a missing bank themselves, guarded by a lock file.

    python -u deep_hedging_v0/experiments/fast/build_bank.py --dynamics heston --product doc
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from experiments.fast.config import DYNAMICS, PRODUCTS, MarketConfig, default_bank_config
from experiments.fast.path_bank import SPLITS, bank_path, get_bank


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dynamics", required=True, choices=DYNAMICS)
    p.add_argument("--product", required=True, choices=PRODUCTS)
    p.add_argument("--bank-dir", type=Path, default=ROOT.parent / ".path_banks")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    market, bank = MarketConfig(), default_bank_config(args.dynamics, args.product, smoke=args.smoke)
    for split in SPLITS:
        data = get_bank(args.bank_dir, args.dynamics, args.product, split, market, bank, device)
        V = data["V"]
        print(f"{split}: {bank_path(args.bank_dir, args.dynamics, args.product, split, market, bank).name} "
              f"S {tuple(data['S'].shape)} V {tuple(V.shape)} V0 mean {float(V[..., 0].mean()):.4f} "
              f"built in {data['meta']['build_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
