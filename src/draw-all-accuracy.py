import json
import math
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from scipy.stats import entropy
from scipy import stats
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
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


def file_name_to_key(file_name):
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
    return key


def compute_model_to_entropy(result_files, retained_ids_list):
    model_to_entropy = {}
    for i, file_name in enumerate(result_files):
        key = file_name_to_key(file_name)

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
                entropy_value = entropy(np.array(distribution))
                entropy_list.append(entropy_value)

        mean = np.mean(entropy_list)
        n = len(entropy_list)
        if n > 1:
            sem = stats.sem(entropy_list)
            if sem > 0:
                ci = stats.t.interval(0.95, n-1, loc=mean, scale=sem)
            else:
                ci = (mean, mean)
        else:
            ci = (mean, mean)

        model_to_entropy[key] = {
            "mean": mean,
            "ci": ci
        }

    return model_to_entropy


def compute_model_to_accuracy(result_files, retained_ids_list):
    model_to_accuracy = {}
    for i, file_name in enumerate(result_files):
        key = file_name_to_key(file_name)

        retained_ids = retained_ids_list[i]
        accuracy_list = []

        uuid_to_answer_to_correctness = {}
        with open(file_name.replace("_counts.jsonl", "_correctness-vllm.jsonl"), "r") as f:
            for line in f:
                item = json.loads(line)
                idx = item["idx"]
                if idx not in retained_ids:
                    continue
                uuid = item["uuid"]
                answer = item["answer"]
                correctness = item["correctness"]

                if uuid not in uuid_to_answer_to_correctness:
                    uuid_to_answer_to_correctness[uuid] = {}
                if correctness is None:
                    continue
                uuid_to_answer_to_correctness[uuid][answer] = correctness.strip().upper() == "TRUE"

        with open(file_name, "r") as f:
            for line in f:
                item = json.loads(line)
                idx = item["idx"]
                if idx not in retained_ids:
                    continue
                uuid = item["uuid"]
                answer_counts = item["answer_counts"]
                total_count = sum(answer_counts.values())
                correct_count = 0
                for answer, count in answer_counts.items():
                    if uuid not in uuid_to_answer_to_correctness:
                        continue
                    if answer not in uuid_to_answer_to_correctness[uuid]:
                        continue
                    if uuid_to_answer_to_correctness[uuid][answer]:
                        correct_count += count
                accuracy = correct_count / total_count
                accuracy_list.append(accuracy)

        mean = np.mean(accuracy_list)
        n = len(accuracy_list)
        if n > 1:
            sem = stats.sem(accuracy_list)
            if sem > 0:
                ci = stats.t.interval(0.95, n-1, loc=mean, scale=sem)
            else:
                ci = (mean, mean)
        else:
            ci = (mean, mean)

        model_to_accuracy[key] = {
            "mean": mean,
            "ci": ci
        }

    return model_to_accuracy


def plot_interplay_shift(dataset_to_model_to_entropy, dataset_to_model_to_accuracy, models):
    """
    Generates the trajectory plot visualizing the shift in accuracy and entropy
    from Standard (Disable) to Reasoning mode, including 95% Confidence Intervals.
    """
    sns.set_theme(style="white")
    datasets = list(dataset_to_model_to_entropy.keys())

    # Map model names to letters A, B, C...
    model_to_letter = {m: chr(65 + i) for i, m in enumerate(models)}

    fig, axes = plt.subplots(1, len(datasets), figsize=(18, 6), sharey=False, sharex=False)

    std_color = "#4C72B0" # Standard Blue
    rsn_color = "#DD8452" # Reasoning Orange

    for i, dataset in enumerate(datasets):
        ax = axes[i]

        # 1. Use dashed-line grid
        ax.grid(True, linestyle='--', alpha=0.6, zorder=0)

        for model in models:
            std_key = f"{model}-Disable"
            rsn_key = model

            # Skip if the model data wasn't successfully extracted for this dataset
            if std_key not in dataset_to_model_to_entropy[dataset] or rsn_key not in dataset_to_model_to_entropy[dataset]:
                continue

            # Extract means and symmetric errors for Standard Mode
            ent_std_data = dataset_to_model_to_entropy[dataset][std_key]
            acc_std_data = dataset_to_model_to_accuracy[dataset][std_key]

            ent_std = ent_std_data["mean"]
            ent_std_err = ent_std - ent_std_data["ci"][0]

            acc_std = acc_std_data["mean"] * 100
            acc_std_err = (acc_std_data["mean"] - acc_std_data["ci"][0]) * 100

            # Extract means and symmetric errors for Reasoning Mode
            ent_rsn_data = dataset_to_model_to_entropy[dataset][rsn_key]
            acc_rsn_data = dataset_to_model_to_accuracy[dataset][rsn_key]

            ent_rsn = ent_rsn_data["mean"]
            ent_rsn_err = ent_rsn - ent_rsn_data["ci"][0]

            acc_rsn = acc_rsn_data["mean"] * 100
            acc_rsn_err = (acc_rsn_data["mean"] - acc_rsn_data["ci"][0]) * 100

            # 1. Plot error bars (drawn first so points stay on top)
            ax.errorbar(ent_std, acc_std, xerr=ent_std_err, yerr=acc_std_err,
                        fmt='none', ecolor=std_color, alpha=0.4, capsize=3, zorder=1)
            ax.errorbar(ent_rsn, acc_rsn, xerr=ent_rsn_err, yerr=acc_rsn_err,
                        fmt='none', ecolor=rsn_color, alpha=0.4, capsize=3, zorder=1)

            # 2. Draw the vector arrow
            ax.annotate("", xy=(ent_rsn, acc_rsn), xytext=(ent_std, acc_std),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, shrinkA=6, shrinkB=6),
                        zorder=2)

            # 3. Plot Standard point
            ax.scatter(ent_std, acc_std, color=std_color, s=80, zorder=3)

            # 4. Plot Reasoning point
            ax.scatter(ent_rsn, acc_rsn, color=rsn_color, s=80, zorder=3)

            # 5. Label the Reasoning point with the corresponding letter
            label = model_to_letter[model]
            ax.text(ent_rsn, acc_rsn + 1.5, label, fontsize=16, ha='center', va='bottom',
                    fontweight='bold', color="#333333", zorder=4)

        # Formatting
        ax.set_title(dataset.replace("-", " ").title(), fontsize=16, fontweight="bold")

        if i == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=16, fontweight="bold")

    fig.supxlabel("Entropy (Decision-making Stability)", fontsize=16, fontweight="bold")

    # 2. Add a legend region below the figure
    # Create custom handles for modes
    mode_handles = [
        mlines.Line2D([], [], color='w', marker='o', markerfacecolor=std_color, markersize=10, label='Non-reasoning'),
        mlines.Line2D([], [], color='w', marker='o', markerfacecolor=rsn_color, markersize=10, label='Reasoning')
    ]

    # Create custom handles for models using the letters
    model_handles = [
        mlines.Line2D([], [], color='w', marker='', label=f"{model_to_letter[m]}: {m}") for m in models
    ]

    # Combine handles and plot below the subplots
    all_handles = mode_handles + model_handles
    fig.legend(handles=all_handles,
               loc='lower center',
               bbox_to_anchor=(0.5, -0.15),
               ncol=4,
               frameon=False,
               fontsize=16)

    plt.tight_layout()
    plt.savefig("figures/interplay_shift_plot_ci.png", bbox_inches="tight", dpi=150)


if __name__ == "__main__":
    datasets = [
        "medmcqa",
        "mmlu-accounting",
        "mmlu-law",
    ]

    dataset_to_model_to_entropy = {}
    dataset_to_model_to_accuracy = {}

    for dataset_name in datasets:
        retained_ids_list = []
        retained_ids = get_retained_keys([
            f"outputs/{dataset_name}/processed_results/Qwen3-4B_temp0.6_n50_dt_counts.jsonl",
            f"outputs/{dataset_name}/processed_results/Qwen3-4B_temp0.6_n50_counts.jsonl",
        ], dataset_name)
        retained_ids_list += [retained_ids] * 2
        retained_ids = get_retained_keys([
            f"outputs/{dataset_name}/processed_results/Qwen3-32B_temp0.6_n50_dt_counts.jsonl",
            f"outputs/{dataset_name}/processed_results/Qwen3-32B_temp0.6_n50_counts.jsonl",
        ], dataset_name)
        retained_ids_list += [retained_ids] * 2
        retained_ids = get_retained_keys([
            f"outputs/{dataset_name}/processed_results/Qwen3-30B-A3B_temp0.6_n50_dt_counts.jsonl",
            f"outputs/{dataset_name}/processed_results/Qwen3-30B-A3B_temp0.6_n50_counts.jsonl",
        ], dataset_name)
        retained_ids_list += [retained_ids] * 2
        retained_ids = get_retained_keys([
            f"outputs/{dataset_name}/processed_results/Seed-OSS-36B-Instruct_temp1.1_n50_dt_counts.jsonl",
            f"outputs/{dataset_name}/processed_results/Seed-OSS-36B-Instruct_temp1.1_n50_counts.jsonl",
        ], dataset_name)
        retained_ids_list += [retained_ids] * 2
        retained_ids = get_retained_keys([
            f"outputs/{dataset_name}/processed_results/NVIDIA-Nemotron-Nano-9B-v2_temp0.6_n50_dt_counts.jsonl",
            f"outputs/{dataset_name}/processed_results/NVIDIA-Nemotron-Nano-9B-v2_temp0.6_n50_counts.jsonl",
        ], dataset_name)
        retained_ids_list += [retained_ids] * 2
        retained_ids = get_retained_keys([
            f"outputs/{dataset_name}/processed_results/NVIDIA-Nemotron-Nano-12B-v2_temp0.6_n50_dt_counts.jsonl",
            f"outputs/{dataset_name}/processed_results/NVIDIA-Nemotron-Nano-12B-v2_temp0.6_n50_counts.jsonl",
        ], dataset_name)
        retained_ids_list += [retained_ids] * 2

        result_files = [
            f"outputs/{dataset_name}/processed_results/Qwen3-4B_temp0.6_n50_dt_counts.jsonl",
            f"outputs/{dataset_name}/processed_results/Qwen3-4B_temp0.6_n50_counts.jsonl",

            f"outputs/{dataset_name}/processed_results/Qwen3-32B_temp0.6_n50_dt_counts.jsonl",
            f"outputs/{dataset_name}/processed_results/Qwen3-32B_temp0.6_n50_counts.jsonl",

            f"outputs/{dataset_name}/processed_results/Qwen3-30B-A3B_temp0.6_n50_dt_counts.jsonl",
            f"outputs/{dataset_name}/processed_results/Qwen3-30B-A3B_temp0.6_n50_counts.jsonl",

            f"outputs/{dataset_name}/processed_results/Seed-OSS-36B-Instruct_temp1.1_n50_dt_counts.jsonl",
            f"outputs/{dataset_name}/processed_results/Seed-OSS-36B-Instruct_temp1.1_n50_counts.jsonl",

            f"outputs/{dataset_name}/processed_results/NVIDIA-Nemotron-Nano-9B-v2_temp0.6_n50_dt_counts.jsonl",
            f"outputs/{dataset_name}/processed_results/NVIDIA-Nemotron-Nano-9B-v2_temp0.6_n50_counts.jsonl",

            f"outputs/{dataset_name}/processed_results/NVIDIA-Nemotron-Nano-12B-v2_temp0.6_n50_dt_counts.jsonl",
            f"outputs/{dataset_name}/processed_results/NVIDIA-Nemotron-Nano-12B-v2_temp0.6_n50_counts.jsonl",
        ]

        model_to_entropy = compute_model_to_entropy(
            result_files=result_files,
            retained_ids_list=retained_ids_list
        )
        dataset_to_model_to_entropy[dataset_name] = model_to_entropy

        model_to_accuracy = compute_model_to_accuracy(
            result_files=result_files,
            retained_ids_list=retained_ids_list
        )
        dataset_to_model_to_accuracy[dataset_name] = model_to_accuracy

    models = [
        "Qwen3-4B",
        "Qwen3-32B",
        "Qwen3-30B-A3B",
        "Seed-36B",
        "Nemotron-9B",
        "Nemotron-12B",
    ]

    # Generate the visualization
    plot_interplay_shift(dataset_to_model_to_entropy, dataset_to_model_to_accuracy, models)