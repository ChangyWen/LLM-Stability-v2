import json
import math
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from scipy.stats import fisher_exact, MonteCarloMethod
from scipy.stats import entropy
from scipy import stats
import seaborn as sns


def get_retained_keys(result_files):
    retained_ids_list = []
    for file_name in result_files:
        with open(file_name, "r") as f:
            idx_set = set()
            for line in f:
                item = json.loads(line)
                idx = item["idx"]
                answer_counts = item["answer_counts"]
                if answer_counts is None:
                    continue
                total_count = sum(answer_counts.values())
                if total_count < 35:
                    continue
                idx_set.add(idx)
            retained_ids_list.append(idx_set)
    return set.intersection(*retained_ids_list)


def plot_statistics(file_to_metrics):
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.edgecolor": "gray",
        "axes.linewidth": 0.8,
    })

    keys = [
        f"{model_name}-Disable_temp0.3", f"{model_name}_temp0.3",
        f"{model_name}-Disable_temp0.6", f"{model_name}_temp0.6",
        f"{model_name}-Disable_temp0.9", f"{model_name}_temp0.9",
    ]
    group_labels = [f"0.3", f"0.6", f"0.9"]
    avg = [file_to_metrics[k]["avg"] for k in keys]
    ci = [file_to_metrics[k]["ci"] for k in keys]
    yerr = [upper - mean for mean, (lower, upper) in zip(avg, ci)]
    diffs = [avg[i*2] - avg[i*2 + 1] for i in range(len(group_labels))]  # Non-R. − R.
    avg_accuracy = [file_to_metrics[k]["avg_accuracy"] for k in keys]

    x = np.arange(len(keys))
    bar_width = 0.8

    # Modern color palette
    palette = sns.color_palette("flare", 7)
    colors = [palette[0]] * 2 + [palette[2]] * 2 + [palette[4]] * 2
    group_colors = [palette[0], palette[2], palette[4]]

    fig, ax = plt.subplots(dpi=1024)
    ax2 = ax.twinx()

    # Draw bars one by one so we can customize alpha
    bars = []
    for i, (mean, err, key, color) in enumerate(zip(avg, yerr, keys, colors)):
        alpha_val = 0.6 if "Disable" in key else 1.0
        bar = ax.bar(
            x[i], mean, bar_width,
            yerr=err, capsize=5,
            color=color, edgecolor="black",
            linewidth=0.6, alpha=alpha_val
        )
        bars.append(bar[0])

    # Add mean and CI labels
    for i, bar in enumerate(bars):
        mean = avg[i]
        lower, upper = ci[i]
        center = bar.get_x() + bar.get_width() / 2

        # CI upper label (bold)
        ax.text(center, upper + 0.015, f"{upper:.2f}", ha="center", va="bottom", fontsize=8, color="dimgray", fontweight="bold")

        # CI lower label (bold)
        ax.text(center, lower - 0.025, f"{lower:.2f}", ha="center", va="top", fontsize=8, color="dimgray", fontweight="bold")

    # Simplified xtick labels for subconditions
    sublabels = ["Non-R.", "R."] * (len(keys) // 2)
    ax.set_xticks(x)
    ax.set_xticklabels(sublabels, rotation=0, ha="center")

    # Add group labels (centered under every two bars)
    group_positions = [0.5 + i * 2 for i in range(len(group_labels))]
    for i, (pos, label) in enumerate(zip(group_positions, group_labels)):
        ax.text(pos, -0.09, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=10, fontweight="bold", color=group_colors[i], rotation=0)

    # Remove default xlabel
    ax.set_xlabel("")
    # Add custom xlabel lower down (in axis coords)
    ax.text(0.5, -0.28, "Model", transform=ax.transAxes, ha="center", va="top", fontsize=11, fontweight="bold")

    ax.set_ylabel("Entropy", fontsize=11, fontweight="bold")
    ax.set_title("MedMCQA – Entropy (Mean ± 95% CI)", pad=15, weight="bold")

    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)

    # Use the same group positions you computed for group labels (centers of each pair)
    # If defined later, recompute: group_positions = [0.5 + i * 2 for i in range(len(group_labels))]
    # ax2.plot(group_positions, diffs, marker="o", linewidth=2.0, markersize=6, zorder=5, linestyle="--", label="ΔEntropy (Non-R. − R.)", color="gray")
    # Plot gray dashed connecting line
    ax2.plot(group_positions, diffs, color="gray", linestyle=":", linewidth=1.5, zorder=4, alpha=0.7)

    # Plot colored markers (group-colored)
    for pos, diff, color in zip(group_positions, diffs, group_colors):
        ax2.scatter(pos, diff, color=color, s=45, edgecolor="black", linewidth=0.6, zorder=5, label="_nolegend_")

    # Add invisible handle for legend entry
    ax2.plot([], [], color="gray", linestyle=":", marker="o", markerfacecolor="white", markeredgecolor="black", label="ΔEntropy (Non-R. − R.)")

    # Show legend
    ax2.legend(loc="best", frameon=True, fancybox=True, fontsize=9)

    # Zero baseline for reference
    # ax2.axhline(0, linestyle=":", linewidth=1.0, color="gray")
    ax2.set_ylim(bottom=0)

    # Right-side y-axis label
    ax2.set_ylabel("ΔEntropy", fontsize=11, fontweight="bold")

    # Optional: make the right spine subtle
    ax2.spines["right"].set_color("gray")
    ax2.spines["right"].set_linewidth(0.8)

    # Optional: keep secondary grid off (primary y-grid already on)
    ax2.grid(False)

    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax2.set_axisbelow(True)
    for spine in ["top", "left"]:
        ax2.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_file, bbox_inches="tight")
    plt.close()


def get_statistics(result_files, retained_ids_list):
    file_to_metrics = {}
    for i, file_name in enumerate(result_files):
        if "Seed-OSS-36B-Instruct" in file_name and ("dt" not in file_name):
            key = "Seed-OSS-36B-Instruct"
        elif "Seed-OSS-36B-Instruct" in file_name and ("dt" in file_name):
            key = "Seed-OSS-36B-Instruct-Disable"
        elif "Qwen3-4B" in file_name and ("dt" not in file_name):
            key = "Qwen3-4B"
        elif "Qwen3-4B" in file_name and ("dt" in file_name):
            key = "Qwen3-4B-Disable"
        elif "Qwen3-32B" in file_name and ("dt" not in file_name):
            key = "Qwen3-32B"
        elif "Qwen3-32B" in file_name and ("dt" in file_name):
            key = "Qwen3-32B-Disable"
        elif "Qwen3-30B-A3B" in file_name and ("dt" not in file_name):
            key = "Qwen3-30B-A3B"
        elif "Qwen3-30B-A3B" in file_name and ("dt" in file_name):
            key = "Qwen3-30B-A3B-Disable"
        else:
            assert False, f"Unknown file name: {file_name}"
        temperature = file_name.split("/")[-1].split("_")[1]
        key = f"{key}_{temperature}"
        file_to_metrics[key] = {}
        retained_ids = retained_ids_list[i]
        entropy_list = []
        with open(file_name, "r") as f:
            for line in f:
                item = json.loads(line)
                idx = item["idx"]
                if idx not in retained_ids:
                    continue
                answer_counts = item["answer_counts"]
                distribution = []
                total_count = sum(answer_counts.values())
                for answer, count in answer_counts.items():
                    distribution.append(count / total_count)
                entropy_list.append(entropy(np.array(distribution)))

        accuracy_list = []
        with open(file_name.replace("_counts.jsonl", "_correctness.jsonl"), "r") as f:
            for line in f:
                item = json.loads(line)
                idx = item["idx"]
                if idx not in retained_ids:
                    continue
                correct_count = item["correct_count"]
                total_count = item["total_count"]
                accuracy = correct_count / total_count
                accuracy_list.append(accuracy)
        file_to_metrics[key]["avg_accuracy"] = np.mean(accuracy_list)

        mean = np.mean(entropy_list)
        n = len(entropy_list)
        sem = np.std(entropy_list) / np.sqrt(n)
        ci = stats.t.interval(0.95, n-1, loc=mean, scale=sem)
        file_to_metrics[key]["avg"] = mean
        file_to_metrics[key]["ci"] = ci
        print(key)
        print(f"retained_ids: {len(retained_ids)}")
        print("avg:", f"{file_to_metrics[key]['avg']:.4f}")
        print("ci:", f"{file_to_metrics[key]['ci'][0]:.4f} - {file_to_metrics[key]['ci'][1]:.4f}")
        print("avg_accuracy:", f"{file_to_metrics[key]['avg_accuracy']:.4f}")
        print("-" * 100)

    plot_statistics(file_to_metrics)


if __name__ == "__main__":
    model_name = "Seed-OSS-36B-Instruct"
    save_file = f"outputs/medmcqa/figures/temperature_{model_name}.png"

    retained_ids_list = []
    retained_ids = get_retained_keys([
        f"outputs/medmcqa/processed_results/{model_name}_temp0.0_n1_dt_counts.jsonl",
        f"outputs/medmcqa/processed_results/{model_name}_temp0.0_n1_counts.jsonl",
        f"outputs/medmcqa/processed_results/{model_name}_temp0.3_n50_dt_counts.jsonl",
        f"outputs/medmcqa/processed_results/{model_name}_temp0.3_n50_counts.jsonl",
        f"outputs/medmcqa/processed_results/{model_name}_temp0.6_n50_dt_counts.jsonl",
        f"outputs/medmcqa/processed_results/{model_name}_temp0.6_n50_counts.jsonl",
        f"outputs/medmcqa/processed_results/{model_name}_temp0.9_n50_dt_counts.jsonl",
        f"outputs/medmcqa/processed_results/{model_name}_temp0.9_n50_counts.jsonl",
    ])
    retained_ids_list += [retained_ids] * 6

    get_statistics([
        f"outputs/medmcqa/processed_results/{model_name}_temp0.0_n1_dt_counts.jsonl",
        f"outputs/medmcqa/processed_results/{model_name}_temp0.0_n1_counts.jsonl",

        f"outputs/medmcqa/processed_results/{model_name}_temp0.3_n50_dt_counts.jsonl",
        f"outputs/medmcqa/processed_results/{model_name}_temp0.3_n50_counts.jsonl",

        f"outputs/medmcqa/processed_results/{model_name}_temp0.6_n50_dt_counts.jsonl",
        f"outputs/medmcqa/processed_results/{model_name}_temp0.6_n50_counts.jsonl",

        f"outputs/medmcqa/processed_results/{model_name}_temp0.9_n50_dt_counts.jsonl",
        f"outputs/medmcqa/processed_results/{model_name}_temp0.9_n50_counts.jsonl",
    ], retained_ids_list)
