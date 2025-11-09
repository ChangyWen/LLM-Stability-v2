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
        "Qwen3-4B-Disable", "Qwen3-4B",
        "Qwen3-32B-Disable", "Qwen3-32B",
        "Qwen3-30B-A3B-Disable", "Qwen3-30B-A3B",
        "Seed-36B-Disable", "Seed-36B",
    ]
    group_labels = ["Qwen3-4B", "Qwen3-32B", "Qwen3-30B-A3B", "Seed-36B"]
    avg = [file_to_metrics[k]["avg"] for k in keys]
    ci = [file_to_metrics[k]["ci"] for k in keys]
    yerr = [upper - mean for mean, (lower, upper) in zip(avg, ci)]

    x = np.arange(len(keys))
    bar_width = 0.8

    # Modern color palette
    palette = sns.color_palette("Set2", 4)
    colors = [palette[0]] * 2 + [palette[1]] * 2 + [palette[2]] * 2 + [palette[3]] * 2

    fig, ax = plt.subplots(dpi=1024)

    # Draw bars one by one so we can customize alpha
    bars = []
    for i, (mean, err, key, color) in enumerate(zip(avg, yerr, keys, colors)):
        alpha_val = 0.75 if "Disable" in key else 1.0
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

        # CI upper label
        ax.text(center, upper + 0.015, f"{upper:.2f}", ha="center", va="bottom", fontsize=8, color="dimgray")

        # CI lower label
        ax.text(center, lower - 0.025, f"{lower:.2f}", ha="center", va="top", fontsize=8, color="dimgray")

    # Simplified xtick labels for subconditions
    sublabels = ["Non-R.", "R."] * (len(keys) // 2)
    ax.set_xticks(x)
    ax.set_xticklabels(sublabels, rotation=0, ha="center")

    # Add group labels (centered under every two bars)
    group_positions = [0.5 + i * 2 for i in range(len(group_labels))]
    group_colors = [palette[0], palette[1], palette[2], palette[3]]
    for i, (pos, label) in enumerate(zip(group_positions, group_labels)):
        ax.text(pos, -0.09, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=10, fontweight="bold", color=group_colors[i])

    ax.set_ylabel("Entropy")
    ax.set_title("MMLU Law – Entropy (Mean ± 95% CI)", pad=15, weight="bold")

    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    save_file = "outputs/mmlu-law/figures/entropy.png"
    plt.tight_layout()
    plt.savefig(save_file, bbox_inches="tight")
    plt.close()


def get_statistics(result_files, retained_ids_list):
    file_to_metrics = {}
    for i, file_name in enumerate(result_files):
        if "Seed-OSS-36B-Instruct" in file_name and ("dt" not in file_name):
            key = "Seed-36B"
        elif "Seed-OSS-36B-Instruct" in file_name and ("dt" in file_name):
            key = "Seed-36B-Disable"
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
        print("-" * 100)

    plot_statistics(file_to_metrics)


if __name__ == "__main__":
    retained_ids_list = []
    retained_ids = get_retained_keys([
        "outputs/mmlu-law/processed_results/Qwen3-4B_temp0.6_n50_dt_counts.jsonl",
        "outputs/mmlu-law/processed_results/Qwen3-4B_temp0.6_n50_counts.jsonl",
    ])
    retained_ids_list += [retained_ids] * 2
    retained_ids = get_retained_keys([
        "outputs/mmlu-law/processed_results/Qwen3-32B_temp0.6_n50_dt_counts.jsonl",
        "outputs/mmlu-law/processed_results/Qwen3-32B_temp0.6_n50_counts.jsonl",
    ])
    retained_ids_list += [retained_ids] * 2
    retained_ids = get_retained_keys([
        "outputs/mmlu-law/processed_results/Qwen3-30B-A3B_temp0.6_n50_dt_counts.jsonl",
        "outputs/mmlu-law/processed_results/Qwen3-30B-A3B_temp0.6_n50_counts.jsonl",
    ])
    retained_ids_list += [retained_ids] * 2
    retained_ids = get_retained_keys([
        "outputs/mmlu-law/processed_results/Seed-OSS-36B-Instruct_temp1.1_n50_dt_counts.jsonl",
        "outputs/mmlu-law/processed_results/Seed-OSS-36B-Instruct_temp1.1_n50_counts.jsonl",
    ])
    retained_ids_list += [retained_ids] * 2

    get_statistics([
        "outputs/mmlu-law/processed_results/Qwen3-4B_temp0.6_n50_dt_counts.jsonl",
        "outputs/mmlu-law/processed_results/Qwen3-4B_temp0.6_n50_counts.jsonl",

        "outputs/mmlu-law/processed_results/Qwen3-32B_temp0.6_n50_dt_counts.jsonl",
        "outputs/mmlu-law/processed_results/Qwen3-32B_temp0.6_n50_counts.jsonl",

        "outputs/mmlu-law/processed_results/Qwen3-30B-A3B_temp0.6_n50_dt_counts.jsonl",
        "outputs/mmlu-law/processed_results/Qwen3-30B-A3B_temp0.6_n50_counts.jsonl",

        "outputs/mmlu-law/processed_results/Seed-OSS-36B-Instruct_temp1.1_n50_dt_counts.jsonl",
        "outputs/mmlu-law/processed_results/Seed-OSS-36B-Instruct_temp1.1_n50_counts.jsonl",
    ], retained_ids_list)
