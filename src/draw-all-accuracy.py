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
        model_to_entropy[key] = mean

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
        model_to_accuracy[key] = mean

    return model_to_accuracy


def plot_interplay_shift(dataset_to_model_to_entropy, dataset_to_model_to_accuracy, models):
    """
    Generates the trajectory plot visualizing the shift in accuracy and entropy
    from Standard (Disable) to Reasoning mode.
    """
    sns.set_theme(style="whitegrid")
    datasets = list(dataset_to_model_to_entropy.keys())

    # Map model names to letters A, B, C...
    model_to_letter = {m: chr(65 + i) for i, m in enumerate(models)}

    fig, axes = plt.subplots(1, len(datasets), figsize=(18, 6), sharey=True, sharex=False)

    std_color = "#4C72B0" # Standard Blue
    rsn_color = "#DD8452" # Reasoning Orange

    for i, dataset in enumerate(datasets):
        ax = axes[i]

        for model in models:
            std_key = f"{model}-Disable"
            rsn_key = model

            # Skip if the model data wasn't successfully extracted for this dataset
            if std_key not in dataset_to_model_to_entropy[dataset] or rsn_key not in dataset_to_model_to_entropy[dataset]:
                continue

            # Extract values (converting accuracy to percentage)
            ent_std = dataset_to_model_to_entropy[dataset][std_key]
            acc_std = dataset_to_model_to_accuracy[dataset][std_key] * 100

            ent_rsn = dataset_to_model_to_entropy[dataset][rsn_key]
            acc_rsn = dataset_to_model_to_accuracy[dataset][rsn_key] * 100

            # 1. Draw the vector arrow
            ax.annotate("", xy=(ent_rsn, acc_rsn), xytext=(ent_std, acc_std),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, shrinkA=6, shrinkB=6))

            # 2. Plot Standard point
            ax.scatter(ent_std, acc_std, color=std_color, s=80, zorder=3,
                       label="Standard" if (i==0 and model==models[0]) else "")

            # 3. Plot Reasoning point
            ax.scatter(ent_rsn, acc_rsn, color=rsn_color, s=80, zorder=3,
                       label="Reasoning" if (i==0 and model==models[0]) else "")

            # 4. Label the Reasoning point with the corresponding letter
            label = model_to_letter[model]
            ax.text(ent_rsn, acc_rsn + 1.5, label, fontsize=10, ha='center', va='bottom',
                    fontweight='bold', color="#333333")

        # Formatting
        ax.set_title(dataset.replace("-", " ").title(), fontsize=14)

        # [Removed] ax.set_xlabel("Entropy...")

        if i == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=14)
            ax.legend(title="Inference Mode")

    # [Added] A single, shared X-axis label for the entire figure
    fig.supxlabel("Entropy (Lower = More Confident)", fontsize=14)

    plt.tight_layout()
    plt.savefig("figures/interplay_shift_plot.png", bbox_inches="tight", dpi=150)


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

    # print(dataset_to_model_to_entropy)
    # print(dataset_to_model_to_accuracy)

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