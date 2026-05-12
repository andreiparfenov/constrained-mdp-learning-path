import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

Path("results").mkdir(exist_ok=True)

COLORS = {
    "A": "#999999",   # grey  — baseline
    "B": "#4477AA",   # blue  — Phi only
    "C": "#EE6677",   # red   — QA reward only
    "D": "#228833",   # green — full model (your contribution)
}

def smooth(values, window=50):
    return np.convolve(values, np.ones(window) / window, mode="valid")

def plot_learning_curves(results, save_path="results/learning_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for cond, data in results.items():
        color = COLORS[cond]
        label = f"Cond. {cond}: {data['label']}"

        rewards_smooth  = smooth(data["rewards"])
        mastered_smooth = smooth(data["mastered"])

        axes[0].plot(rewards_smooth, color=color, label=label, linewidth=2)
        axes[1].plot(mastered_smooth, color=color, label=label, linewidth=2)

    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total reward (smoothed)")
    axes[0].set_title("Reward convergence across conditions")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Skills mastered per episode (smoothed)")
    axes[1].set_title("Learning efficiency: skills mastered")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Learning curves saved to {save_path}")
    plt.show()


def plot_comparison_bar(results, save_path="results/comparison_bar.png"):
    FINAL_WINDOW = 200
    conds  = list(results.keys())
    labels = [f"Cond. {c}" for c in conds]

    final_rewards  = [np.mean(results[c]["rewards"][-FINAL_WINDOW:])  for c in conds]
    final_mastered = [np.mean(results[c]["mastered"][-FINAL_WINDOW:]) for c in conds]

    x = np.arange(len(conds))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bars1 = axes[0].bar(x, final_rewards, width,
                        color=[COLORS[c] for c in conds], edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Average reward (final 200 episodes)")
    axes[0].set_title("Final reward comparison")
    axes[0].grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars1, final_rewards):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    bars2 = axes[1].bar(x, final_mastered, width,
                        color=[COLORS[c] for c in conds], edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Avg skills mastered per episode")
    axes[1].set_title("Learning efficiency comparison")
    axes[1].grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars2, final_mastered):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    # Annotation: condition D = your full model
    axes[0].annotate("← Your model", xy=(3, final_rewards[3]),
                     xytext=(2.2, final_rewards[3] + 0.05),
                     arrowprops=dict(arrowstyle="->", color="black"),
                     fontsize=9, color="black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Comparison bar chart saved to {save_path}")
    plt.show()


def print_results_table(results):
    FINAL_WINDOW = 200
    print("\n" + "=" * 75)
    print(f"{'Condition':<12} {'Phi':<8} {'QA Reward':<12} "
          f"{'Avg Reward':<15} {'Avg Skills Mastered'}")
    print("=" * 75)

    phi_map = {"A": "No", "B": "Yes", "C": "No",  "D": "Yes"}
    qar_map = {"A": "No", "B": "No",  "C": "Yes", "D": "Yes"}

    for cond, data in results.items():
        avg_r = np.mean(data["rewards"][-FINAL_WINDOW:])
        avg_m = np.mean(data["mastered"][-FINAL_WINDOW:])
        marker = " ← YOUR MODEL" if cond == "D" else ""
        print(f"  Cond. {cond}    {phi_map[cond]:<8} {qar_map[cond]:<12} "
              f"{avg_r:<15.4f} {avg_m:.4f}{marker}")

    print("=" * 75)

    # Improvement of D over A
    r_A = np.mean(results["A"]["rewards"][-FINAL_WINDOW:])
    r_D = np.mean(results["D"]["rewards"][-FINAL_WINDOW:])
    m_A = np.mean(results["A"]["mastered"][-FINAL_WINDOW:])
    m_D = np.mean(results["D"]["mastered"][-FINAL_WINDOW:])

    print(f"\nCondition D vs A (full model vs baseline):")
    print(f"  Reward improvement:  {(r_D - r_A) / max(abs(r_A), 1e-10) * 100:+.1f}%")
    print(f"  Mastery improvement: {(m_D - m_A) / max(abs(m_A), 1e-10) * 100:+.1f}%")


def plot_bdo_distribution(results_path="data/mdp_results.pkl",
                          save_path="results/bdo_analysis.png"):
    print("\n[BDO distribution analysis would go here]")
    print("  Track: fraction of steps that are preserve / update / transient")
    print("  Expected: Condition D has healthier preserve/update ratio")
    print("  Condition A has more transient events (no ZPD guidance)")


if __name__ == "__main__":
    with open("data/mdp_results.pkl", "rb") as f:
        results = pickle.load(f)

    print_results_table(results)
    plot_learning_curves(results)
    plot_comparison_bar(results)
    plot_bdo_distribution()

    print("\nAll evaluation complete.")
    print("Key output files:")
    print("  results/learning_curves.png")
    print("  results/comparison_bar.png")
