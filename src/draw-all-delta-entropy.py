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


def load_correctness_map(file_name, retained_ids):
    """Helper to load the correctness map for a specific file."""
    uuid_to_answer_to_correctness = {}
    corr_file = file_name.replace("_counts.jsonl", "_correctness-vllm.jsonl")
    try:
        with open(corr_file, "r") as f:
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
                if correctness is not None:
                    uuid_to_answer_to_correctness[uuid][answer] = correctness.strip().upper() == "TRUE"
    except FileNotFoundError:
        pass
    return uuid_to_answer_to_correctness


def get_question_components(file_name, retained_ids, correctness_map):
    """Calculates H(C) and P(C=0)H(Y|C=0) for each question in a single file."""
    idx_to_components = {}
    with open(file_name, "r") as f:
        for line in f:
            item = json.loads(line)
            idx = item["idx"]
            if idx not in retained_ids:
                continue

            uuid = item["uuid"]
            answer_counts = item["answer_counts"]

            if uuid not in correctness_map:
                continue
            cmap = correctness_map[uuid]

            total_count = sum(answer_counts.values())
            if total_count == 0:
                continue

            correct_counts = []
            wrong_counts = []

            for answer, count in answer_counts.items():
                if answer in cmap and cmap[answer]:
                    correct_counts.append(count)
                else:
                    wrong_counts.append(count)

            # if len(correct_counts) > 1:
                # # check if "mmol/L" in any of the answers
                # if any("mmol/L" in answer for answer in answer_counts.keys()):
                #     print(f"idx: {idx}")
                #     print(json.dumps(answer_counts, indent=4))
                #     print(json.dumps(cmap, indent=4))
                #     print("--------------------------------")
                # continue

            p_c1 = sum(correct_counts) / total_count
            p_c0 = sum(wrong_counts) / total_count

            # 1. Component: H(C)
            h_c = entropy([p_c1, p_c0]) if 0 < p_c1 < 1 else 0.0

            # 2. Component: P(C=0) * H(Y|C=0)
            if p_c0 > 0 and sum(wrong_counts) > 0:
                wrong_dist = [c / sum(wrong_counts) for c in wrong_counts]
                h_y_c0_weighted = p_c0 * entropy(wrong_dist)
            else:
                h_y_c0_weighted = 0.0

            # 3. Component: P(C=1) * H(Y|C=1)
            # (Usually 0, but calculated safely in case multiple distinct strings evaluate to True)
            if p_c1 > 0 and sum(correct_counts) > 0:
                correct_dist = [c / sum(correct_counts) for c in correct_counts]
                h_y_c1_weighted = p_c1 * entropy(correct_dist)
            else:
                h_y_c1_weighted = 0.0

            # Mathematical reconstruction based on the Entropy Chain Rule
            calculated_h_y = h_c + h_y_c0_weighted + h_y_c1_weighted

            all_dist = [c / total_count for c in answer_counts.values()]
            h_y_total = entropy(all_dist)

            assert math.isclose(h_y_total, calculated_h_y, abs_tol=1e-7), \
                f"Entropy verification failed! Direct H(Y): {h_y_total}, Decomposed: {calculated_h_y}"

            idx_to_components[idx] = {
                "h_c": h_c,
                "h_y_c0_weighted": h_y_c0_weighted
            }
    return idx_to_components


def compute_paired_deltas_per_model(std_file, rsn_file, retained_ids):
    """
    1. Computes components per question for Standard mode.
    2. Computes components per question for Reasoning mode.
    3. Calculates the delta (Standard - Reasoning) for EACH question.
    4. Returns the average of those paired deltas.
    """
    # Load correctness maps
    std_cmap = load_correctness_map(std_file, retained_ids)
    rsn_cmap = load_correctness_map(rsn_file, retained_ids)

    # Get components per question index
    std_components = get_question_components(std_file, retained_ids, std_cmap)
    rsn_components = get_question_components(rsn_file, retained_ids, rsn_cmap)

    delta_h_c_list = []
    delta_beyond_list = []

    # Calculate paired deltas for questions that exist in both sets
    common_idx = set(std_components.keys()).intersection(set(rsn_components.keys()))

    for idx in common_idx:
        std_vals = std_components[idx]
        rsn_vals = rsn_components[idx]

        # Delta = Non-Reasoning - Reasoning (Positive = Stability Gained)
        d_h_c = std_vals["h_c"] - rsn_vals["h_c"]
        d_beyond = std_vals["h_y_c0_weighted"] - rsn_vals["h_y_c0_weighted"]

        delta_h_c_list.append(d_h_c)
        delta_beyond_list.append(d_beyond)

    return {
        "delta_h_c_mean": np.mean(delta_h_c_list) if delta_h_c_list else 0.0,
        "delta_beyond_mean": np.mean(delta_beyond_list) if delta_beyond_list else 0.0,
        # You now have access to the raw lists if you ever need to calculate p-values!
        "raw_deltas": {
            "h_c": delta_h_c_list,
            "beyond": delta_beyond_list
        }
    }


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


def plot_delta_decomposition(dataset_to_model_deltas, models, model_to_color):
    # Apply the target style parameters
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 10,
        "axes.edgecolor": "gray",
        "axes.linewidth": 0.8,
    })

    datasets = list(dataset_to_model_deltas.keys())

    fig, axes = plt.subplots(1, len(datasets), figsize=(16, 5), sharey=True, dpi=1024)
    x_positions = np.arange(len(models))
    bar_width = 0.7

    dataset_name_to_title = {
        "medmcqa": r"Medicine ($\it{MedMCQA}$)",
        "mmlu-accounting": r"Finance ($\it{MMLU\!-\!Accounting}$)",
        "mmlu-law": r"Law ($\it{MMLU\!-\!Law}$)",
    }

    for i, dataset in enumerate(datasets):
        ax = axes[i]

        # Style Title
        title = dataset_name_to_title.get(dataset, dataset)
        ax.set_title(title, pad=10, weight="bold")

        # Style Grid and Spines
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        # Bold zero line
        ax.axhline(0, color='black', linewidth=1.2, zorder=3)

        for j, model in enumerate(models):
            if model not in dataset_to_model_deltas[dataset]:
                continue

            deltas = dataset_to_model_deltas[dataset][model]
            delta_h_c = deltas["delta_h_c_mean"]
            delta_beyond = deltas["delta_beyond_mean"]
            m_color = model_to_color[model]

            # Bar 1: Accuracy-related reduction
            # Applied: edgecolor="black", linewidth=0.6, alpha=1.0
            ax.bar(x_positions[j], delta_h_c, width=bar_width, color='#E0E0E0',
                   edgecolor="black", linewidth=0.6, zorder=4)

            # Bar 2: Beyond-accuracy reduction
            ax.bar(x_positions[j], delta_beyond, width=bar_width, bottom=delta_h_c,
                   color=m_color, edgecolor="black", linewidth=0.6, zorder=4)

        # Style X-ticks
        ax.set_xticks(x_positions)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.tick_params(axis="x", length=0, pad=6) # Removing tick marks, matching pad=6

        if i == 0:
            ax.set_ylabel(r"Entropy Reduction ($\Delta$ Entropy)", fontsize=12, fontweight="bold")

    # Custom Legends matching the styling conventions
    acc_patch = mpatches.Patch(
        facecolor='#E0E0E0', edgecolor='black', linewidth=0.6,
        label=r'Accuracy-driven reduction: $\Delta H(C)$'
    )
    beyond_patch = mpatches.Patch(
        facecolor='#4A4A4A', edgecolor='black', linewidth=0.6,
        label=r'Beyond-accuracy reduction: $\Delta [ P(C=0)H(Y|C=0) ]$'
    )

    fig.legend(handles=[acc_patch, beyond_patch],
               loc='lower center',
               bbox_to_anchor=(0.5, -0.22),
               ncol=2,
               frameon=False,
               fontsize=12,
               handlelength=1.6,
               columnspacing=2.0,
               handletextpad=0.5)

    plt.tight_layout()
    plt.savefig("figures/delta_entropy_decomposition.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    datasets = ["medmcqa", "mmlu-accounting", "mmlu-law"]

    # Map each model key to its (Standard/Disable File, Reasoning File) tuple
    model_to_files = {
        "Qwen3-4B": (
            "Qwen3-4B_temp0.6_n50_dt_counts.jsonl",
            "Qwen3-4B_temp0.6_n50_counts.jsonl"
        ),
        "Qwen3-32B": (
            "Qwen3-32B_temp0.6_n50_dt_counts.jsonl",
            "Qwen3-32B_temp0.6_n50_counts.jsonl"
        ),
        "Qwen3-30B-A3B": (
            "Qwen3-30B-A3B_temp0.6_n50_dt_counts.jsonl",
            "Qwen3-30B-A3B_temp0.6_n50_counts.jsonl"
        ),
        "Seed-36B": (
            "Seed-OSS-36B-Instruct_temp1.1_n50_dt_counts.jsonl",
            "Seed-OSS-36B-Instruct_temp1.1_n50_counts.jsonl"
        ),
        "Nemotron-9B": (
            "NVIDIA-Nemotron-Nano-9B-v2_temp0.6_n50_dt_counts.jsonl",
            "NVIDIA-Nemotron-Nano-9B-v2_temp0.6_n50_counts.jsonl"
        ),
        "Nemotron-12B": (
            "NVIDIA-Nemotron-Nano-12B-v2_temp0.6_n50_dt_counts.jsonl",
            "NVIDIA-Nemotron-Nano-12B-v2_temp0.6_n50_counts.jsonl"
        ),
    }

    dataset_to_model_deltas = {}

    for dataset_name in datasets:
        model_deltas = {}

        for model_key, (std_filename, rsn_filename) in model_to_files.items():
            std_file = f"outputs/{dataset_name}/processed_results/{std_filename}"
            rsn_file = f"outputs/{dataset_name}/processed_results/{rsn_filename}"

            try:
                # 1. Get the questions that meet the >35 response threshold for BOTH modes
                retained_ids = get_retained_keys([std_file, rsn_file], dataset_name)

                # 2. Compute the paired deltas for this specific model on this dataset
                deltas = compute_paired_deltas_per_model(std_file, rsn_file, retained_ids)

                # 3. Store the results
                model_deltas[model_key] = deltas

            except FileNotFoundError:
                print(f"Warning: Missing files for {model_key} on {dataset_name}. Skipping.")
                continue

        dataset_to_model_deltas[dataset_name] = model_deltas

    for dataset_name, model_deltas in dataset_to_model_deltas.items():
        for model_key, deltas in model_deltas.items():
            print(f"{dataset_name} {model_key} {deltas['delta_h_c_mean']} {deltas['delta_beyond_mean']}")

    # Define the order of models for the x-axis
    models = [
        "Qwen3-4B",
        "Qwen3-32B",
        "Qwen3-30B-A3B",
        "Seed-36B",
        "Nemotron-9B",
        "Nemotron-12B"
    ]

    # Generate colors using your existing function
    model_to_color = prepare_model_to_color()

    # Generate the decomposed stacked bar chart
    plot_delta_decomposition(dataset_to_model_deltas, models, model_to_color)
    print("Plot successfully saved to figures/delta_entropy_decomposition.png")