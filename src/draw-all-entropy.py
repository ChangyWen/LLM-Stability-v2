import json
import math
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from scipy.stats import entropy
from scipy import stats
import seaborn as sns


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


def draw_entropy_bars_on_ax(ax, file_to_metrics, dataset_name, show_xlabel=True, show_ylabel=True):
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
        "Nemotron-9B-Disable", "Nemotron-9B",
        "Nemotron-12B-Disable", "Nemotron-12B",
    ]
    group_labels = ["Qwen3-4B", "Qwen3-32B", "Qwen3-30B-A3B", "Seed-36B", "Nemotron-9B", "Nemotron-12B"]

    avg = [file_to_metrics[k]["avg"] for k in keys]
    ci = [file_to_metrics[k]["ci"] for k in keys]
    yerr = [upper - mean for mean, (lower, upper) in zip(avg, ci)]

    x = np.arange(len(keys))
    bar_width = 0.8

    total_color_num = len(group_labels)
    palette = sns.color_palette("Set2", total_color_num)
    color_pool = [palette[i] for i in range(total_color_num)]
    colors = []
    group_colors = []
    for i in range(total_color_num):
        colors += [color_pool[i]] * 2
        group_colors.append(color_pool[i])

    # bars
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

    # CI labels
    for i, bar in enumerate(bars):
        mean = avg[i]
        lower, upper = ci[i]
        center = bar.get_x() + bar.get_width() / 2
        ax.text(center, upper + 0.015, f"{upper:.3f}", ha="center", va="bottom",
                fontsize=8, color="black", fontweight="bold")
        ax.text(center, lower - 0.025, f"{lower:.3f}", ha="center", va="top",
                fontsize=8, color="black", fontweight="bold")

    # xticks
    sublabels = ["Non-R.", "R."] * (len(keys) // 2)
    ax.set_xticks(x)
    ax.set_xticklabels(sublabels, rotation=0, ha="center")

    # group labels under pairs
    group_positions = [0.5 + i * 2 for i in range(len(group_labels))]
    for i, (pos, label) in enumerate(zip(group_positions, group_labels)):
        ax.text(
            pos, -0.09, label,
            transform=ax.get_xaxis_transform(),
            ha="center", va="top",
            fontsize=10, fontweight="bold",
            color=group_colors[i],
            rotation=15
        )

    # axis labels (controlled externally)
    ax.set_xlabel("" if not show_xlabel else "")
    if show_ylabel:
        ax.set_ylabel("Entropy", fontsize=11, fontweight="bold")
    else:
        ax.set_ylabel("")

    dataset_name_to_title = {
        "daily_dilemmas": "DailyDilemmas",
        "medmcqa": "MedMCQA",
        "mmlu-accounting": "MMLU Accounting",
        "mmlu-law": "MMLU Law",
    }
    ax.set_title(f"{dataset_name_to_title[dataset_name]}", pad=10, weight="bold")

    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # If you want consistent y-limits across subplots, uncomment and adjust:
    # ax.set_ylim(0.0, 1.12)

    # Hide x tick labels if this subplot shouldn't show xlabel area
    if not show_xlabel:
        ax.tick_params(axis="x", labelbottom=False)


def plot_all_datasets(metrics_by_dataset, save_file="figures/entropy-all.png"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=1024)

    dataset_order = [
        "daily_dilemmas",
        "medmcqa",
        "mmlu-accounting",
        "mmlu-law",
    ]

    for idx, dataset_name in enumerate(dataset_order):
        r, c = divmod(idx, 2)
        ax = axes[r, c]

        # (2) remove xlabel for first row
        show_xlabel = (r == 1)
        # (3) remove ylabel for second column
        show_ylabel = (c == 0)

        draw_entropy_bars_on_ax(
            ax=ax,
            file_to_metrics=metrics_by_dataset[dataset_name],
            dataset_name=dataset_name,
            show_xlabel=show_xlabel,
            show_ylabel=show_ylabel,
        )

    # shared labels
    fig.supxlabel("Model", fontsize=12, fontweight="bold", y=0.03)
    fig.supylabel("Entropy", fontsize=12, fontweight="bold", x=0.02)

    # leave room for sup labels
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    plt.savefig(save_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_file}")


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
                for i in ["1", "2", "3"]:
                    distribution.append(results_counter.get(i, 0) / len(results))
                entropy_value = entropy(np.array(distribution))
                entropy_list.append(entropy_value)

            mean = np.mean(entropy_list)
            n = len(entropy_list)
            sem = np.std(entropy_list) / np.sqrt(n)
            ci = stats.t.interval(0.95, n-1, loc=mean, scale=sem)
            file_to_metrics[key]["avg"] = mean
            file_to_metrics[key]["ci"] = ci

        else:
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

    return file_to_metrics


if __name__ == "__main__":
    datasets = [
        "daily_dilemmas",
        "medmcqa",
        "mmlu-accounting",
        "mmlu-law",
    ]

    metrics_by_dataset = {}

    for dataset_name in datasets:
        subfix = "_counts" if dataset_name != "daily_dilemmas" else ""

        retained_ids_list = []
        retained_ids = get_retained_keys([
            f"outputs/{dataset_name}/processed_results/Qwen3-4B_temp0.6_n50_dt{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/Qwen3-4B_temp0.6_n50{subfix}.jsonl",
        ], dataset_name)
        retained_ids_list += [retained_ids] * 2
        retained_ids = get_retained_keys([
            f"outputs/{dataset_name}/processed_results/Qwen3-32B_temp0.6_n50_dt{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/Qwen3-32B_temp0.6_n50{subfix}.jsonl",
        ], dataset_name)
        retained_ids_list += [retained_ids] * 2
        retained_ids = get_retained_keys([
            f"outputs/{dataset_name}/processed_results/Qwen3-30B-A3B_temp0.6_n50_dt{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/Qwen3-30B-A3B_temp0.6_n50{subfix}.jsonl",
        ], dataset_name)
        retained_ids_list += [retained_ids] * 2
        retained_ids = get_retained_keys([
            f"outputs/{dataset_name}/processed_results/Seed-OSS-36B-Instruct_temp1.1_n50_dt{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/Seed-OSS-36B-Instruct_temp1.1_n50{subfix}.jsonl",
        ], dataset_name)
        retained_ids_list += [retained_ids] * 2
        retained_ids = get_retained_keys([
            f"outputs/{dataset_name}/processed_results/NVIDIA-Nemotron-Nano-9B-v2_temp0.6_n50_dt{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/NVIDIA-Nemotron-Nano-9B-v2_temp0.6_n50{subfix}.jsonl",
        ], dataset_name)
        retained_ids_list += [retained_ids] * 2
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
        metrics_by_dataset[dataset_name] = file_to_metrics

    plot_all_datasets(metrics_by_dataset, save_file="figures/entropy-all.png")
