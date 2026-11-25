"""Run every baseline on every objective; print a table, save CSV + comparison plots.

    python scripts/run_baselines.py
    python scripts/run_baselines.py --config configs/overloaded.yaml --episodes 50
"""

# ---------------------------------------------------------------------------
# Generative AI disclosure: Claude (Anthropic) was used to help draft the code in this project, as well as debugging syntax issues if the code broke.# 
# ---------------------------------------------------------------------------

import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hospital_cao.env import HospitalConfig, load_config
from hospital_cao.env.rewards import REWARDS
from hospital_cao.baselines import ALL_BASELINES
from hospital_cao.evaluation.evaluate import evaluate_with_std
from hospital_cao.evaluation.plots import plot_comparison
from hospital_cao.utils.logging import save_rows_csv

ap = argparse.ArgumentParser()
ap.add_argument("--config", nargs="*", default=[], help="YAML config file(s), merged in order")
ap.add_argument("--episodes", type=int, default=20)
ap.add_argument("--tag", default="default", help="label for output files")
args = ap.parse_args()

base = load_config(*args.config) if args.config else HospitalConfig()
cols = ["return", "treated", "deaths", "deteriorations", "mean_wait", "doctor_util", "mean_queue"]
rows = []

for objective in REWARDS:
    cfg = base.with_overrides(objective=objective)
    print(f"\n=== objective: {objective} ({args.episodes} episodes, {cfg.horizon} ticks each) ===")
    print(f"{'policy':<14}" + "".join(f"{c:>15}" for c in cols))
    results = {}
    for name, cls in ALL_BASELINES.items():
        res = evaluate_with_std(cfg, cls(seed=123), n_episodes=args.episodes)
        results[name] = res
        print(f"{name:<14}" + "".join(f"{res[c][0]:>15.2f}" for c in cols))
        rows.append({"objective": objective, "policy": name,
                     **{f"{c}_mean": round(res[c][0], 4) for c in cols},
                     **{f"{c}_std": round(res[c][1], 4) for c in cols}})
    plot_comparison(results, ["return", "deaths", "mean_wait"],
                    ROOT / "results" / "figures" / f"baselines_{args.tag}_{objective}.png",
                    title=f"Baselines - {objective} ({args.tag})")

save_rows_csv(rows, ROOT / "results" / "logs" / f"baselines_{args.tag}.csv")
print(f"\nSaved CSV to results/logs/baselines_{args.tag}.csv and plots to results/figures/")
