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
                if "temp0.0" in file_name:
                    if total_count < 1:
                        continue
                else:
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
        f"{model_name}-Disable_temp1.2", f"{model_name}_temp1.2",
    ]
    group_labels = [f"0.3", f"0.6", f"0.9", f"1.2"]
    avg = [file_to_metrics[k]["avg"] for k in keys]
    ci = [file_to_metrics[k]["ci"] for k in keys]
    yerr = [upper - mean for mean, (lower, upper) in zip(avg, ci)]
    diffs = [avg[i*2] - avg[i*2 + 1] for i in range(len(group_labels))]  # Non-R. − R.
    avg_accuracy = [file_to_metrics[k]["avg_accuracy"] for k in keys]
    temp0_disable_avg_accuracy = file_to_metrics[f"{model_name}-Disable_temp0.0"]["avg_accuracy"]
    temp0_avg_accuracy = file_to_metrics[f"{model_name}_temp0.0"]["avg_accuracy"]

    x = np.arange(len(keys))
    bar_width = 0.8

    # Modern color palette
    palette = sns.color_palette("flare", 7)
    colors = [palette[0]] * 2 + [palette[2]] * 2 + [palette[4]] * 2 + [palette[6]] * 2
    group_colors = [palette[0], palette[2], palette[4], palette[6]]

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
        ax.text(center, upper + 0.015, f"{upper:.3f}", ha="center", va="bottom", fontsize=8, color="black", fontweight="bold")

        # CI lower label (bold)
        ax.text(center, lower - 0.025, f"{lower:.3f}", ha="center", va="top", fontsize=8, color="black", fontweight="bold")

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
    ax.text(0.5, -0.15, "Temperature", transform=ax.transAxes, ha="center", va="top", fontsize=11, fontweight="bold")

    ax.set_ylabel("Entropy", fontsize=11, fontweight="bold")
    ax.set_title(f"MedMCQA – Entropy (Mean ± 95% CI) – {model_name}", pad=15, weight="bold")

    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)

    # Use the same group positions you computed for group labels (centers of each pair)
    # If defined later, recompute: group_positions = [0.5 + i * 2 for i in range(len(group_labels))]
    # ax2.plot(group_positions, diffs, marker="o", linewidth=2.0, markersize=6, zorder=5, linestyle="--", label="ΔEntropy (Non-R. − R.)", color="gray")
    # Plot gray dashed connecting line
    ax2.plot(group_positions, diffs, color="black", linestyle=":", linewidth=1.5, zorder=4, alpha=0.7)

    # Plot colored markers (group-colored)
    for pos, diff, color in zip(group_positions, diffs, group_colors):
        ax2.scatter(pos, diff, color=color, s=45, edgecolor="black", linewidth=0.6, zorder=5, label="_nolegend_")

    # Add invisible handle for legend entry
    ax2.plot([], [], color="black", linestyle=":", marker="o", markerfacecolor="white", markeredgecolor="black", label="ΔEntropy (Non-R. − R.)")

    # Right-side y-axis label
    ax2.set_ylabel("ΔEntropy", fontsize=11, fontweight="bold")

    # Optional: make the right spine subtle
    ax2.spines["right"].set_color("gray")
    ax2.spines["right"].set_linewidth(0.8)

    # Optional: keep secondary grid off (primary y-grid already on)
    ax2.grid(False)

    # ====== Third y-axis for Accuracy (hollow blue stars) ======
    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("outward", 65))  # push a second right axis outward
    ax3.spines["right"].set_color("blue")
    ax3.spines["right"].set_linewidth(0.9)
    ax3.tick_params(axis="y", colors="blue")
    ax3.set_ylabel("Accuracy", fontsize=11, fontweight="bold", color="blue")
    ax3.grid(False)
    # Plot accuracy stars aligned with each bar position
    x_all = list(range(len(keys)))
    ax3.scatter(
        x_all, avg_accuracy,
        marker="*", s=80,
        facecolors="blue", edgecolors="blue",
        linewidth=1.0, zorder=7,
        # label="Accuracy"
    )
    # Add value annotations for accuracy stars (blue, right side of marker)
    for xi, acc in zip(x_all, avg_accuracy):
        ax3.text(
            xi + 0.1, acc,                 # slightly to the right of the star
            f"{acc:.3f}",                  # formatted value
            ha="left", va="center",
            fontsize=8, fontweight="bold",
            color="blue"
        )
    # ====== Add horizontal reference lines for temp0 accuracies ======
    ax3.axhline(
        y=temp0_disable_avg_accuracy,
        color="blue", linestyle="--", linewidth=1.2,
        alpha=0.9, label="Accuracy (temp.=0, Non-R.)"
    )
    ax3.axhline(
        y=temp0_avg_accuracy,
        color="blue", linestyle="-", linewidth=1.2,
        alpha=0.9, label="Accuracy (temp.=0, R.)"
    )
    ax3.set_ylim(0.0, max(avg_accuracy + [temp0_disable_avg_accuracy, temp0_avg_accuracy]) + 0.05)

    # ---- Combine legends from ax2 (ΔEntropy) and ax3 (Accuracy + baselines) into ONE legend ----
    handles1, labels1 = ax2.get_legend_handles_labels()
    handles2, labels2 = ax3.get_legend_handles_labels()
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2
    ax.legend(
        all_handles, all_labels,
        loc="best",
        fontsize=9,
        frameon=True, fancybox=True,
        framealpha=0.9, edgecolor="gray"
    )

    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax2.set_axisbelow(True)
    for spine in ["top", "left"]:
        ax2.spines[spine].set_visible(False)
    ax3.set_axisbelow(True)
    for spine in ["top", "left"]:
        ax3.spines[spine].set_visible(False)

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
        f"outputs/medmcqa/processed_results/{model_name}_temp1.2_n50_dt_counts.jsonl",
        f"outputs/medmcqa/processed_results/{model_name}_temp1.2_n50_counts.jsonl",
    ])
    retained_ids_list += [retained_ids] * 10

    get_statistics([
        f"outputs/medmcqa/processed_results/{model_name}_temp0.0_n1_dt_counts.jsonl",
        f"outputs/medmcqa/processed_results/{model_name}_temp0.0_n1_counts.jsonl",
        f"outputs/medmcqa/processed_results/{model_name}_temp0.3_n50_dt_counts.jsonl",
        f"outputs/medmcqa/processed_results/{model_name}_temp0.3_n50_counts.jsonl",
        f"outputs/medmcqa/processed_results/{model_name}_temp0.6_n50_dt_counts.jsonl",
        f"outputs/medmcqa/processed_results/{model_name}_temp0.6_n50_counts.jsonl",
        f"outputs/medmcqa/processed_results/{model_name}_temp0.9_n50_dt_counts.jsonl",
        f"outputs/medmcqa/processed_results/{model_name}_temp0.9_n50_counts.jsonl",
        f"outputs/medmcqa/processed_results/{model_name}_temp1.2_n50_dt_counts.jsonl",
        f"outputs/medmcqa/processed_results/{model_name}_temp1.2_n50_counts.jsonl",
    ], retained_ids_list)
