import json
import math
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from scipy.stats import entropy
from scipy import stats
import seaborn as sns
import matplotlib.patches as mpatches
from collections import defaultdict
import os

def get_retained_keys(result_files, dataset_name):
    if dataset_name == "daily_dilemmas":
        retained_ids_list = []
        for file_name in result_files:
            with open(file_name, "r") as f:
                idx_to_options = {}
                for line in f:
                    item = json.loads(line)
                    idx = item["idx"]
                    if idx not in idx_to_options:
                        idx_to_options[idx] = []
                    if item["option"] in ["1", "2", "3"]:
                        idx_to_options[idx].append(item["option"])
                retained_ids = set()
                for idx, options in idx_to_options.items():
                    if len(options) < 35:
                        continue
                    retained_ids.add(idx)
            retained_ids_list.append(retained_ids)
        return set.intersection(*retained_ids_list)
    else:
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

def p_to_stars(p):
    """
    Convert p-value to significance stars.
    """
    if math.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"

def paired_entropy_test_one_sided_category(dict_non, dict_reason):
    """
    One-sided paired Wilcoxon test for categories:
    H1: entropy(with reasoning) < entropy(without reasoning)
    Returns p-value (float).
    """
    common = sorted(set(dict_non.keys()) & set(dict_reason.keys()))
    if len(common) == 0:
        return float("nan")

    x_non = np.array([dict_non[i] for i in common], dtype=float)
    x_reason = np.array([dict_reason[i] for i in common], dtype=float)

    diff = x_reason - x_non  # want diff < 0
    if np.allclose(diff, 0):
        return 1.0

    res = stats.wilcoxon(x_reason, x_non, alternative="less", zero_method="wilcox")
    return float(res.pvalue)

def compute_file_to_metrics(result_files, retained_ids_list, dataset_name):
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
        elif "NVIDIA-Nemotron-Nano-9B-v2" in file_name and ("dt" not in file_name):
            key = "Nemotron-9B"
        elif "NVIDIA-Nemotron-Nano-9B-v2" in file_name and ("dt" in file_name):
            key = "Nemotron-9B-Disable"
        elif "NVIDIA-Nemotron-Nano-12B-v2" in file_name and ("dt" not in file_name):
            key = "Nemotron-12B"
        elif "NVIDIA-Nemotron-Nano-12B-v2" in file_name and ("dt" in file_name):
            key = "Nemotron-12B-Disable"
        else:
            assert False, f"Unknown file name: {file_name}"

        if dataset_name == "daily_dilemmas":
            file_to_metrics[key] = {}
            idx_to_results = {}
            retained_ids = retained_ids_list[i]
            idx_to_entropy = {}
            with open(file_name, "r") as f:
                for line in f:
                    item = json.loads(line)
                    idx = item["idx"]
                    if idx not in retained_ids:
                        continue
                    if idx not in idx_to_results:
                        idx_to_results[idx] = []
                    if item["option"] in ["1", "2", "3"]:
                        idx_to_results[idx].append(item["option"])
            entropy_list = []
            for idx in idx_to_results:
                results = idx_to_results[idx]
                results_counter = dict(Counter(results))
                distribution = []
                for j in ["1", "2", "3"]:
                    distribution.append(results_counter.get(j, 0) / len(results))
                entropy_value = entropy(np.array(distribution))
                idx_to_entropy[idx] = entropy_value
                entropy_list.append(entropy_value)

            file_to_metrics[key]["entropy_list"] = entropy_list
            file_to_metrics[key]["idx_to_entropy"] = idx_to_entropy

        else:
            file_to_metrics[key] = {}
            retained_ids = retained_ids_list[i]
            entropy_list = []
            idx_to_entropy = {}
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
                    entropy_value = entropy(np.array(distribution))
                    idx_to_entropy[idx] = entropy_value
                    entropy_list.append(entropy_value)

            file_to_metrics[key]["entropy_list"] = entropy_list
            file_to_metrics[key]["idx_to_entropy"] = idx_to_entropy

    return file_to_metrics

def draw_type_aggregated_entropy_bars(
    category_to_metrics: dict,
    save_file: str = "figures/entropy-type-aggregated.png",
    title: str = "Ethical vs. Professional"
):
    # --- consistent global styling ---
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 10,
        "axes.edgecolor": "gray",
        "axes.linewidth": 0.8,
    })

    # Adjusted to be narrower and taller based on previous requests
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.8), dpi=1024)

    categories = ["Ethical", "Professional"]

    intra_gap = 1.2  # Gap between Disable/Reasoning
    inter_gap = 2.0  # Gap between Ethical/Professional
    bar_width = 1.0

    x_ticks = []
    star_labels = []
    pos = 0.0

    for cat in categories:
        # 1. Disable Bar
        mean_d = category_to_metrics[cat]["Disable"]["avg"]
        ci_d = category_to_metrics[cat]["Disable"]["ci"]
        yerr_d = ci_d[1] - mean_d

        bar_d = ax.bar(
            pos, mean_d, bar_width,
            yerr=yerr_d, capsize=5,
            color="white", edgecolor="black", linewidth=0.8
        )

        # 2. Reasoning Bar
        mean_r = category_to_metrics[cat]["Reasoning"]["avg"]
        ci_r = category_to_metrics[cat]["Reasoning"]["ci"]
        yerr_r = ci_r[1] - mean_r

        bar_r = ax.bar(
            pos + intra_gap, mean_r, bar_width,
            yerr=yerr_r, capsize=5,
            color="white", edgecolor="black", linewidth=0.8, hatch="//"
        )

        center = pos + intra_gap / 2
        x_ticks.append(center)

        # --- CI labels ---
        ax.text(pos, ci_d[1] + 0.015, f"{ci_d[1]:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.text(pos, ci_d[0] - 0.025, f"{ci_d[0]:.3f}", ha="center", va="top", fontsize=8, fontweight="bold")

        ax.text(pos + intra_gap, ci_r[1] + 0.015, f"{ci_r[1]:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.text(pos + intra_gap, ci_r[0] - 0.025, f"{ci_r[0]:.3f}", ha="center", va="top", fontsize=8, fontweight="bold")

        # --- Calculate Significance Stars (No brackets drawn here anymore) ---
        p_val = paired_entropy_test_one_sided_category(
            category_to_metrics[cat]["Disable"]["idx_to_entropy"],
            category_to_metrics[cat]["Reasoning"]["idx_to_entropy"]
        )
        stars = p_to_stars(p_val)
        star_labels.append(stars)

        pos += (inter_gap + intra_gap)

    # --- axes cosmetics ---
    ax.set_xticks(x_ticks)

    # Append the stars directly below the category name using a newline
    tick_labels = [f"{cat}\n{star}" for cat, star in zip(categories, star_labels)]
    ax.set_xticklabels(tick_labels, fontsize=12, fontweight="bold")

    # Adjust tick params to give the multiline label some breathing room
    ax.tick_params(axis="x", pad=8)
    ax.set_xlabel("")

    ax.set_title(title, pad=10, weight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.supylabel("Entropy (Instability)", fontsize=12, fontweight="bold", x=0.04)

    # --- Legend ---
    style_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", label="Without Reasoning"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", label="With Reasoning"),
    ]

    fig.legend(
        handles=style_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.54, 0.01), # Adjusted slightly lower to account for taller figure/multiline ticks
        fontsize=11,
        handlelength=1.6,
        columnspacing=2.0,
        handletextpad=0.5,
    )

    # Ensure the figure directory exists
    import os
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    plt.tight_layout(rect=[0.05, 0.10, 1.0, 1.0])
    plt.savefig(save_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_file}")


if __name__ == "__main__":
    datasets = [
        "daily_dilemmas",
        "medmcqa",
        "mmlu-accounting",
        "mmlu-law",
    ]

    # New Dictionary Structure
    category_to_entropy = {
        "Ethical": {"Disable": [], "Reasoning": []},
        "Professional": {"Disable": [], "Reasoning": []}
    }
    category_to_idx_to_entropy = {
        "Ethical": {"Disable": {}, "Reasoning": {}},
        "Professional": {"Disable": {}, "Reasoning": {}}
    }

    for dataset_name in datasets:
        # 1. Map dataset to category
        category = "Ethical" if dataset_name == "daily_dilemmas" else "Professional"
        subfix = "_counts" if dataset_name != "daily_dilemmas" else ""

        retained_ids_list = []
        # Qwen3-4B
        retained_ids = get_retained_keys([
            f"outputs/{dataset_name}/processed_results/Qwen3-4B_temp0.6_n50_dt{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/Qwen3-4B_temp0.6_n50{subfix}.jsonl",
        ], dataset_name)
        retained_ids_list += [retained_ids] * 2
        # Qwen3-32B
        retained_ids = get_retained_keys([
            f"outputs/{dataset_name}/processed_results/Qwen3-32B_temp0.6_n50_dt{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/Qwen3-32B_temp0.6_n50{subfix}.jsonl",
        ], dataset_name)
        retained_ids_list += [retained_ids] * 2
        # Qwen3-30B-A3B
        retained_ids = get_retained_keys([
            f"outputs/{dataset_name}/processed_results/Qwen3-30B-A3B_temp0.6_n50_dt{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/Qwen3-30B-A3B_temp0.6_n50{subfix}.jsonl",
        ], dataset_name)
        retained_ids_list += [retained_ids] * 2
        # Seed-OSS-36B
        retained_ids = get_retained_keys([
            f"outputs/{dataset_name}/processed_results/Seed-OSS-36B-Instruct_temp1.1_n50_dt{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/Seed-OSS-36B-Instruct_temp1.1_n50{subfix}.jsonl",
        ], dataset_name)
        retained_ids_list += [retained_ids] * 2
        # Nemotron-9B
        retained_ids = get_retained_keys([
            f"outputs/{dataset_name}/processed_results/NVIDIA-Nemotron-Nano-9B-v2_temp0.6_n50_dt{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/NVIDIA-Nemotron-Nano-9B-v2_temp0.6_n50{subfix}.jsonl",
        ], dataset_name)
        retained_ids_list += [retained_ids] * 2
        # Nemotron-12B
        retained_ids = get_retained_keys([
            f"outputs/{dataset_name}/processed_results/NVIDIA-Nemotron-Nano-12B-v2_temp0.6_n50_dt{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/NVIDIA-Nemotron-Nano-12B-v2_temp0.6_n50{subfix}.jsonl",
        ], dataset_name)
        retained_ids_list += [retained_ids] * 2

        result_files = [
            f"outputs/{dataset_name}/processed_results/Qwen3-4B_temp0.6_n50_dt{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/Qwen3-4B_temp0.6_n50{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/Qwen3-32B_temp0.6_n50_dt{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/Qwen3-32B_temp0.6_n50{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/Qwen3-30B-A3B_temp0.6_n50_dt{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/Qwen3-30B-A3B_temp0.6_n50{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/Seed-OSS-36B-Instruct_temp1.1_n50_dt{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/Seed-OSS-36B-Instruct_temp1.1_n50{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/NVIDIA-Nemotron-Nano-9B-v2_temp0.6_n50_dt{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/NVIDIA-Nemotron-Nano-9B-v2_temp0.6_n50{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/NVIDIA-Nemotron-Nano-12B-v2_temp0.6_n50_dt{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/NVIDIA-Nemotron-Nano-12B-v2_temp0.6_n50{subfix}.jsonl",
        ]

        file_to_metrics = compute_file_to_metrics(
            result_files=result_files,
            retained_ids_list=retained_ids_list,
            dataset_name=dataset_name,
        )

        for key in file_to_metrics:
            is_disable = "Disable" in key
            mode = "Disable" if is_disable else "Reasoning"
            base_model = key.replace("-Disable", "")

            # Append the raw entropies to the pooled lists
            category_to_entropy[category][mode].extend(file_to_metrics[key]["entropy_list"])

            # Save idx-level entropy to calculate significance properly across all pooled answers
            for _idx, _e in file_to_metrics[key]["idx_to_entropy"].items():
                unique_id = f"{base_model}::{dataset_name}::{_idx}"
                category_to_idx_to_entropy[category][mode][unique_id] = _e

    # 3. Calculate Means and Confidence Intervals
    category_to_metrics = {}
    for cat in ["Ethical", "Professional"]:
        category_to_metrics[cat] = {}
        for mode in ["Disable", "Reasoning"]:
            lst = category_to_entropy[cat][mode]
            mean = np.mean(lst)
            n = len(lst)
            sem = stats.sem(lst)
            ci = stats.t.interval(0.95, n-1, loc=mean, scale=sem) if n > 1 else (mean, mean)

            category_to_metrics[cat][mode] = {
                "avg": mean,
                "ci": ci,
                "idx_to_entropy": category_to_idx_to_entropy[cat][mode]
            }

    # 4. Plot
    draw_type_aggregated_entropy_bars(
        category_to_metrics=category_to_metrics,
        save_file="figures/entropy-type-aggregated.png",
        title="All Models' Results Aggregated"
    )