"""Cost curve and the LAPO module breakdown from the final A100 numbers."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEMOS = [5, 10, 25]
MIX = [0.742, 0.817, 0.917]
LORA = [0.575, 0.717, 0.858]
LAPO_MODULES = [
    ("representation\ncosine", 0.000, 0.143),
    ("latent-policy\ncosine", 0.006, 0.995),
]


def main():
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(DEMOS, MIX, "o-", color="#1d4ed8", linewidth=2, label="Mix seen")
    axes[0].plot(DEMOS, LORA, "o-", color="#c2410c", linewidth=2, label="LoRA r=32")
    axes[0].plot([0], [0], "o", color="#64748b", label="Zero-shot")
    axes[0].plot([5], [0], "o", color="#7c3aed", label="Bonus A (LAPO)")
    for x, m, l in zip(DEMOS, MIX, LORA):
        axes[0].annotate(f"{m:.3f}", (x, m), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=9, color="#1d4ed8")
        axes[0].annotate(f"{l:.3f}", (x, l), textcoords="offset points",
                         xytext=(0, -14), ha="center", fontsize=9, color="#c2410c")
    axes[0].set_xlabel("target-демонстраций")
    axes[0].set_ylabel("success")
    axes[0].set_title("Cost curve, среднее по трём задачам")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_xticks([0, 5, 10, 25])
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    axes[0].grid(alpha=0.25)

    labels = [name for name, _, _ in LAPO_MODULES]
    positions = range(len(labels))
    axes[1].bar([p - 0.19 for p in positions], [a for _, a, _ in LAPO_MODULES],
                width=0.38, color="#cbd5e1", label="начало обучения")
    axes[1].bar([p + 0.19 for p in positions], [b for _, _, b in LAPO_MODULES],
                width=0.38, color="#7c3aed", label="конец обучения")
    for position, (_, _, end) in zip(positions, LAPO_MODULES):
        axes[1].text(position + 0.19, end, f"{end:.3f}", ha="center", va="bottom", fontsize=10)
    axes[1].set_xticks(list(positions))
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("cosine")
    axes[1].set_ylim(0, 1.15)
    axes[1].set_title("LAPO: один модуль обучился, другой нет")
    axes[1].legend(frameon=False, fontsize=9, loc="upper left")
    axes[1].grid(alpha=0.25, axis="y")

    figure.tight_layout()
    out = Path("assets/cost-curve-and-lapo.png")
    figure.savefig(out, dpi=160)
    print(out)


if __name__ == "__main__":
    main()
