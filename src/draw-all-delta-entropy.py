import json
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import matplotlib.patches as mpatches
from collections import defaultdict

def get_retained_keys(result_files, dataset_name):
    # Same as your original logic
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
    if "Seed-OSS-36B-Instruct" in file_name and ("dt" not in file_name): return "Seed-36B"
    elif "Seed-OSS-36B-Instruct" in file_name and ("dt" in file_name): return "Seed-36B-Disable"
    elif "Qwen3-4B" in file_name and ("dt" not in file_name): return "Qwen3-4B"
    elif "Qwen3-4B" in file_name and ("dt" in file_name): return "Qwen3-4B-Disable"
    elif "Qwen3-32B" in file_name and ("dt" not in file_name): return "Qwen3-32B"
    elif "Qwen3-32B" in file_name and ("dt" in file_name): return "Qwen3-32B-Disable"
    elif "Qwen3-30B-A3B" in file_name and ("dt" not in file_name): return "Qwen3-30B-A3B"
    elif "Qwen3-30B-A3B" in file_name and ("dt" in file_name): return "Qwen3-30B-A3B-Disable"
    elif "NVIDIA-Nemotron-Nano-9B-v2" in file_name and ("dt" not in file_name): return "Nemotron-9B"
    elif "NVIDIA-Nemotron-Nano-9B-v2" in file_name and ("dt" in file_name): return "Nemotron-9B-Disable"
    elif "NVIDIA-Nemotron-Nano-12B-v2" in file_name and ("dt" not in file_name): return "Nemotron-12B"
    elif "NVIDIA-Nemotron-Nano-12B-v2" in file_name and ("dt" in file_name): return "Nemotron-12B-Disable"
    else: assert False, f"Unknown file name: {file_name}"

def compute_model_entropy_components(result_files, retained_ids_list):
    """
    Computes the decomposed entropy components for each model/mode:
    1. H(C) : Entropy of the binary correctness variable
    2. P(C=0)H(Y|C=0) : Entropy of the incorrect answers weighted by error rate
    """
    model_to_components = {}

    for i, file_name in enumerate(result_files):
        key = file_name_to_key(file_name)
        retained_ids = retained_ids_list[i]

        # 1. Load correctness map for this file
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

        # 2. Compute components for each question
        h_c_list = []
        h_y_given_c0_weighted_list = []

        with open(file_name, "r") as f:
            for line in f:
                item = json.loads(line)
                idx = item["idx"]
                if idx not in retained_ids:
                    continue
                uuid = item["uuid"]
                answer_counts = item["answer_counts"]

                if uuid not in uuid_to_answer_to_correctness:
                    continue
                correctness_map = uuid_to_answer_to_correctness[uuid]

                total_count = sum(answer_counts.values())
                if total_count == 0:
                    continue

                correct_count = 0
                wrong_counts = []

                for answer, count in answer_counts.items():
                    if answer in correctness_map and correctness_map[answer]:
                        correct_count += count
                    else:
                        wrong_counts.append(count)

                p_c1 = correct_count / total_count
                p_c0 = 1.0 - p_c1

                # Component 1: H(C)
                h_c = entropy([p_c1, p_c0]) if 0 < p_c1 < 1 else 0.0
                h_c_list.append(h_c)

                # Component 2: P(C=0) * H(Y|C=0)
                if p_c0 > 0 and sum(wrong_counts) > 0:
                    wrong_dist = [c / sum(wrong_counts) for c in wrong_counts]
                    h_y_c0 = entropy(wrong_dist)
                    h_y_given_c0_weighted_list.append(p_c0 * h_y_c0)
                else:
                    h_y_given_c0_weighted_list.append(0.0)

        # Average across all valid questions
        model_to_components[key] = {
            "h_c_mean": np.mean(h_c_list),
            "h_y_given_c0_weighted_mean": np.mean(h_y_given_c0_weighted_list)
        }

    return model_to_components

def prepare_model_to_color():
    model_labels = ["Qwen3-4B", "Qwen3-32B", "Qwen3-30B-A3B", "Seed-OSS-36B-Instruct", "NVIDIA-Nemotron-Nano-9B-v2", "NVIDIA-Nemotron-Nano-12B-v2"]
    family_palettes = {"Qwen": "Blues", "Seed": "mako", "NVIDIA-Nemotron": "Purples"}
    family_groups = defaultdict(list)
    for model in model_labels:
        for family in family_palettes.keys():
            if model.startswith(family):
                family_groups[family].append(model)
                break

    model_to_color = {}
    for family, models in family_groups.items():
        palette_name = family_palettes[family]
        colors = sns.color_palette(palette_name, n_colors=len(models) + 1)[1:]
        for i, model in enumerate(models):
            model_to_color[model] = colors[i]

    model_to_color["Seed-36B"] = model_to_color["Seed-OSS-36B-Instruct"]
    model_to_color["Nemotron-9B"] = model_to_color["NVIDIA-Nemotron-Nano-9B-v2"]
    model_to_color["Nemotron-12B"] = model_to_color["NVIDIA-Nemotron-Nano-12B-v2"]
    return model_to_color


def plot_delta_decomposition(dataset_to_model_components, models, model_to_color):
    """
    Plots the stacked bar charts showing the decomposition of \Delta H(Y).
    Delta is defined as (Non-Reasoning - Reasoning) so that positive values
    represent a reduction in entropy (a gain in stability).
    """
    sns.set_theme(style="whitegrid")
    datasets = list(dataset_to_model_components.keys())

    fig, axes = plt.subplots(1, len(datasets), figsize=(18, 6), sharey=True)
    x_positions = np.arange(len(models))
    bar_width = 0.55

    for i, dataset in enumerate(datasets):
        ax = axes[i]

        dataset_name = {"medmcqa": "MedMCQA", "mmlu-accounting": "MMLU-Accounting", "mmlu-law": "MMLU-Law"}.get(dataset, dataset)
        ax.set_title(dataset_name, fontsize=16, fontweight="bold")

        # Draw a bold zero line
        ax.axhline(0, color='black', linewidth=1.5, zorder=3)
        ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

        for j, model in enumerate(models):
            std_key = f"{model}-Disable"
            rsn_key = model

            if std_key not in dataset_to_model_components[dataset] or rsn_key not in dataset_to_model_components[dataset]:
                continue

            std_comp = dataset_to_model_components[dataset][std_key]
            rsn_comp = dataset_to_model_components[dataset][rsn_key]

            # Calculate Deltas: Non-Reasoning (std) - Reasoning (rsn)
            # A positive delta means entropy went down (stability improved)
            delta_h_c = std_comp["h_c_mean"] - rsn_comp["h_c_mean"]
            delta_beyond = std_comp["h_y_given_c0_weighted_mean"] - rsn_comp["h_y_given_c0_weighted_mean"]

            m_color = model_to_color[model]

            # Bar 1: delta_h_c (Accuracy-related reduction) - Neutral Grey
            ax.bar(x_positions[j], delta_h_c, width=bar_width, color='#D3D3D3',
                   edgecolor='#A9A9A9', linewidth=1, zorder=2)

            # Bar 2: delta_beyond (Beyond-accuracy reduction) - Model specific color, stacked
            ax.bar(x_positions[j], delta_beyond, width=bar_width, bottom=delta_h_c,
                   color=m_color, edgecolor='white', linewidth=0.5, zorder=2)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=12)
        ax.tick_params(axis='y', labelsize=12)

        if i == 0:
            ax.set_ylabel(r"$\Delta$ Entropy", fontsize=16, fontweight="bold")

    # Custom Legends reflecting the new positive framing
    acc_patch = mpatches.Patch(color='#D3D3D3', ec='#A9A9A9', label=r'Accuracy-driven reduction: $\Delta H(C)$')
    beyond_patch = mpatches.Patch(facecolor='#4A4A4A', label=r'Beyond-accuracy reduction: $\Delta [ P(C=0)H(Y|C=0) ]$')

    fig.legend(handles=[acc_patch, beyond_patch],
               loc='lower center',
               bbox_to_anchor=(0.5, -0.2),
               ncol=2,
               frameon=False,
               fontsize=14)

    plt.tight_layout()
    plt.savefig("figures/delta_entropy_decomposition.png", bbox_inches="tight", dpi=150)


if __name__ == "__main__":
    datasets = ["medmcqa", "mmlu-accounting", "mmlu-law"]
    dataset_to_model_components = {}

    for dataset_name in datasets:
        # Re-using your exact file listing and extraction logic
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

        # Get retained IDs exactly like the original code
        retained_ids_list = [get_retained_keys(result_files[i:i+2], dataset_name) for i in range(0, len(result_files), 2)]
        flattened_retained_ids = [ids for ids in retained_ids_list for _ in range(2)]

        # Compute the specific mathematical components
        model_components = compute_model_entropy_components(result_files, flattened_retained_ids)
        dataset_to_model_components[dataset_name] = model_components

    models = ["Qwen3-4B", "Qwen3-32B", "Qwen3-30B-A3B", "Seed-36B", "Nemotron-9B", "Nemotron-12B"]
    model_to_color = prepare_model_to_color()

    # Generate the decomposed stacked bar chart
    plot_delta_decomposition(dataset_to_model_components, models, model_to_color)