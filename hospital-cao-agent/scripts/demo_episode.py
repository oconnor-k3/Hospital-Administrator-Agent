"""Watch one episode: text trace of the first ticks, summary, and an episode plot.

    python scripts/demo_episode.py --policy acuity_first --config configs/overloaded.yaml
"""
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hospital_cao.env import HospitalConfig, HospitalEnv, load_config
from hospital_cao.baselines import ALL_BASELINES
from hospital_cao.evaluation.plots import plot_episode

ap = argparse.ArgumentParser()
ap.add_argument("--config", nargs="*", default=[])
ap.add_argument("--policy", default="fifo", choices=list(ALL_BASELINES))
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--trace_ticks", type=int, default=6, help="ticks to print as text")
args = ap.parse_args()

cfg = load_config(*args.config) if args.config else HospitalConfig()
env = HospitalEnv(cfg, record_history=True)
policy = ALL_BASELINES[args.policy](seed=args.seed)

env.reset(seed=args.seed)
done, total = False, 0.0
print(env.render_text())
while not done:
    action = policy.act(env)
    _, r, done, _, info = env.step(action)
    total += r
    if action == env.K and env.t <= args.trace_ticks:       # one line per elapsed tick
        print(f"\n[action: end turn | tick reward {r:+.2f}]")
        print(env.render_text())

print("\nSummary:", {k: round(v, 2) for k, v in env.summary().items()}, f"return={total:.1f}")
out = ROOT / "results" / "figures" / f"episode_{args.policy}_seed{args.seed}.png"
plot_episode(env.history, cfg, out, title=f"{args.policy} | {cfg.objective} | arrival {cfg.arrival_rate}")
print("Saved plot:", out.relative_to(ROOT))

