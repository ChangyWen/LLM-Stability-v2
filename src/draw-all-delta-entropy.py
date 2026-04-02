import json
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from collections import defaultdict


def get_retained_keys(result_files, dataset_name):
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
    uuid_to_answer_to_correctness = {}
    corr_file = file_name.replace("_counts.jsonl", "_correctness-vllm-v2.jsonl")
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

            wrong_counts = []
            for answer, count in answer_counts.items():
                if answer in cmap and (not cmap[answer]):
                    wrong_counts.append(count)

            p_c0 = sum(wrong_counts) / total_count

            if p_c0 > 0 and sum(wrong_counts) > 0:
                wrong_dist = [c / sum(wrong_counts) for c in wrong_counts]
                h_y_c0_unweighted = entropy(wrong_dist)
            else:
                h_y_c0_unweighted = None

            idx_to_components[idx] = {
                "h_y_c0_unweighted": h_y_c0_unweighted
            }
    return idx_to_components


def compute_paired_deltas_per_model(std_file, rsn_file, retained_ids):
    """
    Computes both the absolute means and the delta of the pure unweighted error-path entropy.
    """
    std_cmap = load_correctness_map(std_file, retained_ids)
    rsn_cmap = load_correctness_map(rsn_file, retained_ids)

    std_components = get_question_components(std_file, retained_ids, std_cmap)
    rsn_components = get_question_components(rsn_file, retained_ids, rsn_cmap)

    std_unweighted_list = []
    rsn_unweighted_list = []

    common_idx = set(std_components.keys()).intersection(set(rsn_components.keys()))

    for idx in common_idx:
        std_vals = std_components[idx]
        rsn_vals = rsn_components[idx]

        if std_vals["h_y_c0_unweighted"] is not None and rsn_vals["h_y_c0_unweighted"] is not None:
            std_unweighted_list.append(std_vals["h_y_c0_unweighted"])
            rsn_unweighted_list.append(rsn_vals["h_y_c0_unweighted"])

    delta_unweighted_list = np.array(std_unweighted_list) - np.array(rsn_unweighted_list) if std_unweighted_list else []

    return {
        "std_unweighted_mean": np.mean(std_unweighted_list) if std_unweighted_list else 0.0,
        "rsn_unweighted_mean": np.mean(rsn_unweighted_list) if rsn_unweighted_list else 0.0,
        "delta_unweighted_mean": np.mean(delta_unweighted_list) if len(delta_unweighted_list) > 0 else 0.0,
        "valid_pairs_count": len(std_unweighted_list)
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


def plot_unweighted_dumbbell(dataset_to_model_deltas, models, model_to_color):
    import matplotlib.lines as mlines # Ensure this is imported

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.edgecolor": "gray",
        "axes.linewidth": 0.8,
    })

    datasets = list(dataset_to_model_deltas.keys())

    fig, axes = plt.subplots(1, len(datasets), figsize=(16, 5), sharey=True, dpi=1024)
    y_positions = np.arange(len(models))

    dataset_name_to_title = {
        "medmcqa": r"Medicine ($\it{MedMCQA}$)",
        "mmlu-accounting": r"Finance ($\it{MMLU\!-\!Accounting}$)",
        "mmlu-law": r"Law ($\it{MMLU\!-\!Law}$)",
    }

    display_names = {
        "Qwen3-4B": "Qwen3-4B",
        "Qwen3-32B": "Qwen3-32B",
        "Qwen3-30B-A3B": "Qwen3-30B-A3B",
        "Seed-36B": "Seed-OSS-36B-Instruct",
        "Nemotron-9B": "NVIDIA-Nemotron-Nano-9B-v2",
        "Nemotron-12B": "NVIDIA-Nemotron-Nano-12B-v2"
    }

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        title = dataset_name_to_title.get(dataset, dataset)
        ax.set_title(title, pad=10, weight="bold")

        ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        for j, model in enumerate(models):
            if model not in dataset_to_model_deltas[dataset]:
                continue

            deltas = dataset_to_model_deltas[dataset][model]
            std_val = deltas["std_unweighted_mean"]
            rsn_val = deltas["rsn_unweighted_mean"]
            m_color = model_to_color[model]

            # 1. Connecting Line
            ax.plot([std_val, rsn_val], [j, j], color="#CCCCCC", linewidth=3.5, zorder=1)

            # 2. Standard Dot (Grey)
            ax.scatter(std_val, j, color="#A0A0A0", edgecolor="white", linewidth=0.8, s=150, zorder=2)

            # 3. Reasoning Dot (Model Color)
            ax.scatter(rsn_val, j, color=m_color, edgecolor="black", linewidth=0.8, s=150, zorder=3)

        # Apply y-labels only to the first panel
        if i == 0:
            ax.set_yticks(y_positions)
            ax.set_yticklabels([display_names[m] for m in models], fontweight="bold")
            # Invert Y-axis so the first model appears at the top
            ax.invert_yaxis()

    # --- Global X-Axis Label ---
    # Centered horizontally across the entire figure
    fig.supxlabel(r"Conditional (Error-Path) Entropy $H(Y|C=0)$", fontsize=12, fontweight="bold", y=0.05)

    # --- Clean Legend ---
    std_patch = mlines.Line2D([], [], color='white', marker='o', markerfacecolor='#A0A0A0',
                              markeredgecolor='white', markersize=12, label='Without Reasoning')
    rsn_patch = mlines.Line2D([], [], color='white', marker='o', markerfacecolor='#4A4A4A',
                              markeredgecolor='black', markersize=12, label='With Reasoning')

    # Lowered bbox_to_anchor to ensure it sits safely below the supxlabel
    fig.legend(handles=[std_patch, rsn_patch],
               loc='lower center',
               bbox_to_anchor=(0.5, -0.1),
               ncol=2,
               frameon=False,
               fontsize=12,
               handletextpad=0.5,
               columnspacing=3.0)

    # Expanded bottom rect from 0.05 to 0.12 to give breathing room for both supxlabel and legend
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig("figures/delta_unweighted_entropy_dumbbell.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    datasets = ["medmcqa", "mmlu-accounting", "mmlu-law"]

    model_to_files = {
        "Qwen3-4B": ("Qwen3-4B_temp0.6_n50_dt_counts.jsonl", "Qwen3-4B_temp0.6_n50_counts.jsonl"),
        "Qwen3-32B": ("Qwen3-32B_temp0.6_n50_dt_counts.jsonl", "Qwen3-32B_temp0.6_n50_counts.jsonl"),
        "Qwen3-30B-A3B": ("Qwen3-30B-A3B_temp0.6_n50_dt_counts.jsonl", "Qwen3-30B-A3B_temp0.6_n50_counts.jsonl"),
        "Seed-36B": ("Seed-OSS-36B-Instruct_temp1.1_n50_dt_counts.jsonl", "Seed-OSS-36B-Instruct_temp1.1_n50_counts.jsonl"),
        "Nemotron-9B": ("NVIDIA-Nemotron-Nano-9B-v2_temp0.6_n50_dt_counts.jsonl", "NVIDIA-Nemotron-Nano-9B-v2_temp0.6_n50_counts.jsonl"),
        "Nemotron-12B": ("NVIDIA-Nemotron-Nano-12B-v2_temp0.6_n50_dt_counts.jsonl", "NVIDIA-Nemotron-Nano-12B-v2_temp0.6_n50_counts.jsonl"),
    }

    dataset_to_model_deltas = {}

    for dataset_name in datasets:
        model_deltas = {}

        for model_key, (std_filename, rsn_filename) in model_to_files.items():
            std_file = f"outputs/{dataset_name}/processed_results/{std_filename}"
            rsn_file = f"outputs/{dataset_name}/processed_results/{rsn_filename}"

            try:
                retained_ids = get_retained_keys([std_file, rsn_file], dataset_name)
                deltas = compute_paired_deltas_per_model(std_file, rsn_file, retained_ids)
                model_deltas[model_key] = deltas
            except FileNotFoundError:
                print(f"Warning: Missing files for {model_key} on {dataset_name}. Skipping.")
                continue

        dataset_to_model_deltas[dataset_name] = model_deltas

    for dataset_name, model_deltas in dataset_to_model_deltas.items():
        for model_key, deltas in model_deltas.items():
            print(f"{dataset_name} {model_key} | Std: {deltas['std_unweighted_mean']:.4f} -> Rsn: {deltas['rsn_unweighted_mean']:.4f} (over {deltas['valid_pairs_count']} paired error questions)")

    models = [
        "Qwen3-4B", "Qwen3-32B", "Qwen3-30B-A3B",
        "Seed-36B", "Nemotron-9B", "Nemotron-12B"
    ]

    model_to_color = prepare_model_to_color()

    plot_unweighted_dumbbell(dataset_to_model_deltas, models, model_to_color)
    print("Plot successfully saved to figures/delta_unweighted_entropy_dumbbell.png")