"""Cost curve: success vs number of target demos, ours against the baseline.

Reads a jsonl where every line is one evaluated (method, seed, task, n_demos) cell:

    {"method": "baseline", "seed": 0, "task": "libero_goal_0", "n_demos": 5,
     "success": 0.35, "n_episodes": 20}

Aggregation follows the task spec: mean over the three target tasks, spread over
training seeds.

    python cost_curve.py runs/results.jsonl --plot
"""

import argparse
import collections
import json
import pathlib
import statistics


def load(path):
    rows = [json.loads(l) for l in pathlib.Path(path).read_text().splitlines() if l.strip()]
    per_cell = collections.defaultdict(list)  # (method, n_demos, seed) -> success per task
    for r in rows:
        per_cell[(r["method"], r["n_demos"], r["seed"])].append(r["success"])

    per_budget = collections.defaultdict(list)  # (method, n_demos) -> mean over tasks, per seed
    for (method, n, _), successes in per_cell.items():
        per_budget[(method, n)].append(statistics.mean(successes))

    curve = collections.defaultdict(dict)  # method -> n_demos -> (mean, std, n_seeds)
    for (method, n), seeds in per_budget.items():
        curve[method][n] = (
            statistics.mean(seeds),
            statistics.pstdev(seeds) if len(seeds) > 1 else 0.0,
            len(seeds),
        )
    return curve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", default="runs/results.jsonl", nargs="?")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    curve = load(args.results)
    budgets = sorted({n for m in curve.values() for n in m})

    print(f"{'demos':>6}" + "".join(f"{m:>22}" for m in curve))
    for n in budgets:
        line = f"{n:>6}"
        for m in curve:
            if n in curve[m]:
                mean, std, k = curve[m][n]
                line += f"{mean:>14.3f} ±{std:.3f} ({k})"
            else:
                line += f"{'—':>22}"
        print(line)

    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        for m in curve:
            ns = sorted(curve[m])
            ax.errorbar(ns, [curve[m][n][0] for n in ns], yerr=[curve[m][n][1] for n in ns],
                        marker="o", capsize=3, label=m)
        ax.set_xlabel("target demos")
        ax.set_ylabel("success rate")
        ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        out = pathlib.Path("runs") / "cost_curve.png"
        fig.savefig(out, dpi=150)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
