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
                for i in ["1", "2", "3"]:
                    distribution.append(results_counter.get(i, 0) / len(results))
                entropy_value = entropy(np.array(distribution))
                idx_to_entropy[idx] = entropy_value
                entropy_list.append(entropy_value)

            mean = np.mean(entropy_list)
            n = len(entropy_list)
            sem = stats.sem(entropy_list)
            ci = stats.t.interval(0.95, n-1, loc=mean, scale=sem)
            file_to_metrics[key]["avg"] = mean
            file_to_metrics[key]["ci"] = ci
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

            mean = np.mean(entropy_list)
            n = len(entropy_list)
            sem = stats.sem(entropy_list)
            ci = stats.t.interval(0.95, n-1, loc=mean, scale=sem)
            file_to_metrics[key]["avg"] = mean
            file_to_metrics[key]["ci"] = ci
            file_to_metrics[key]["entropy_list"] = entropy_list
            file_to_metrics[key]["idx_to_entropy"] = idx_to_entropy

        print(key)
        print(f"retained_ids: {len(retained_ids)}")
        print("avg:", f"{file_to_metrics[key]['avg']:.4f}")
        print("ci:", f"{file_to_metrics[key]['ci'][0]:.4f} - {file_to_metrics[key]['ci'][1]:.4f}")
        print("-" * 100)

    return file_to_metrics


def draw_aggregated_entropy_bars(
    model_to_metrics: dict,
    save_file: str = "figures/entropy-aggregated.png",
    title: str = "Aggregated (All Datasets)"
):
    # --- consistent global styling (same as your per-dataset plot) ---
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.edgecolor": "gray",
        "axes.linewidth": 0.8,
        # "text.usetex": True,
    })

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.8), dpi=1024)

    # Order must match your previous plotting logic
    keys = [
        "Qwen3-4B-Disable", "Qwen3-4B",
        "Qwen3-32B-Disable", "Qwen3-32B",
        "Qwen3-30B-A3B-Disable", "Qwen3-30B-A3B",
        "Seed-36B-Disable", "Seed-36B",
        "Nemotron-9B-Disable", "Nemotron-9B",
        "Nemotron-12B-Disable", "Nemotron-12B",
    ]

    # Model labels used for color mapping (same palette rule)
    model_labels = ["Qwen3-4B", "Qwen3-32B", "Qwen3-30B-A3B", "Seed-36B", "Nemotron-9B", "Nemotron-12B"]

    # --- x positions (same intra/inter gap layout) ---
    n_models = len(model_labels)
    intra_gap = 1.1
    inter_gap = 1.6
    x = []
    pos = 0.0
    for _ in range(n_models):
        x.append(pos)
        x.append(pos + intra_gap)
        pos += (inter_gap + intra_gap)
    x = np.array(x)
    bar_width = 1.0

    # --- same color set ---
    palette = sns.color_palette("Set2", len(model_labels))
    model_to_color = {m: palette[i] for i, m in enumerate(model_labels)}

    def key_to_model(k: str) -> str:
        return k.replace("-Disable", "")

    # --- pull metrics + draw bars ---
    avg = [model_to_metrics[k]["avg"] for k in keys]
    ci = [model_to_metrics[k]["ci"] for k in keys]
    yerr = [upper - mean for mean, (lower, upper) in zip(avg, ci)]

    bars = []
    for i, (mean, err, key) in enumerate(zip(avg, yerr, keys)):
        model = key_to_model(key)
        color = model_to_color[model]

        is_non_reasoning = ("Disable" in key)
        hatch = None if is_non_reasoning else "//"

        bar = ax.bar(
            x[i], mean, bar_width,
            yerr=err, capsize=5,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            alpha=1.0,
            hatch=hatch
        )
        bars.append(bar[0])

    # --- CI labels (same style) ---
    for i, bar in enumerate(bars):
        lower, upper = ci[i]
        center = bar.get_x() + bar.get_width() / 2
        ax.text(center, upper + 0.015, f"{upper:.3f}",
                ha="center", va="bottom", fontsize=8, color="black", fontweight="bold")
        ax.text(center, lower - 0.025, f"{lower:.3f}",
                ha="center", va="top", fontsize=8, color="black", fontweight="bold")

    # --- axes cosmetics (match your per-dataset style) ---
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")  # shared ylabel via fig.supylabel
    ax.set_title(title, pad=10, weight="bold")

    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # --- shared ylabel (same text) ---
    fig.supylabel("Entropy (Decision-making Stability)", fontsize=12, fontweight="bold", x=0.06)

    # --- same legend structure ---
    legend_model_labels = [
        "Qwen3-4B",
        "Qwen3-32B",
        "Qwen3-30B-A3B",
        "Seed-OSS-36B-Instruct",
        "NVIDIA-Nemotron-Nano-9B-v2",
        "NVIDIA-Nemotron-Nano-12B-v2",
    ]
    model_handles = [
        mpatches.Patch(facecolor=palette[i], edgecolor="black", label=legend_model_labels[i])
        for i in range(len(legend_model_labels))
    ]
    style_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", label="Without Reasoning"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", label="With Reasoning"),
    ]
    handles = model_handles + style_handles

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.54, 0.01),
        fontsize=10,
        handlelength=1.6,
        columnspacing=1.6,
        handletextpad=0.5,
    )

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

    model_to_entropy_list = defaultdict(list)
    model_to_idx_to_entropy = defaultdict(dict)

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
        for key in file_to_metrics:
            model_to_entropy_list[key] += file_to_metrics[key]["entropy_list"]
            # update the dict model_to_idx_to_entropy for the key
            model_to_idx_to_entropy[key].update(file_to_metrics[key]["idx_to_entropy"])

    model_to_metrics = {}
    for model in model_to_entropy_list:
        mean = np.mean(model_to_entropy_list[model])
        n = len(model_to_entropy_list[model])
        sem = stats.sem(model_to_entropy_list[model])
        ci = stats.t.interval(0.95, n-1, loc=mean, scale=sem)
        model_to_metrics[model] = {
            "avg": mean,
            "ci": ci,
            "idx_to_entropy": model_to_idx_to_entropy[model],
        }

    draw_aggregated_entropy_bars(
        model_to_metrics=model_to_metrics,
        save_file="figures/entropy-aggregated.png",
        title="All Datasets"
    )
