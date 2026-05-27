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
        with open(file_name.replace("_counts.jsonl", "_correctness-vllm-v2.jsonl"), "r") as f:
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


def plot_interplay_shift(dataset_to_model_to_entropy, dataset_to_model_to_accuracy, models, model_to_color):
    """
    Generates the trajectory plot visualizing the shift in accuracy and entropy
    from Standard (Disable) to Reasoning mode, including 95% Confidence Intervals.
    """
    # 1. Apply the target style parameters matching previous plots
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.edgecolor": "gray",
        "axes.linewidth": 0.8,
    })

    datasets = list(dataset_to_model_to_entropy.keys())

    # Set DPI to 1024 and Figure size to match the other plots
    fig, axes = plt.subplots(1, len(datasets), figsize=(16, 5), sharey=False, sharex=False, dpi=1024)

    # 2. Define distinct markers (matching the Dumbbell Plot grammar)
    std_marker = "X" # Cross for Non-Reasoning
    rsn_marker = "o" # Circle for Reasoning

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

        # Style Title
        title = dataset_name_to_title.get(dataset, dataset)
        ax.set_title(title, pad=10, weight="bold")

        # Style Grid and Spines (Remove top and right boundaries)
        ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        for model in models:
            std_key = f"{model}-Disable"
            rsn_key = model

            if std_key not in dataset_to_model_to_entropy[dataset] or rsn_key not in dataset_to_model_to_entropy[dataset]:
                continue

            m_color = model_to_color[model]

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

            # Plot error bars (slightly transparent to keep focus on markers)
            error_bars1 = ax.errorbar(acc_std, ent_std, xerr=acc_std_err, yerr=ent_std_err,
                        fmt='none', ecolor=m_color, alpha=0.8, capsize=3, zorder=1)
            error_bars2 = ax.errorbar(acc_rsn, ent_rsn, xerr=acc_rsn_err, yerr=ent_rsn_err,
                        fmt='none', ecolor=m_color, alpha=0.8, capsize=3, zorder=1)

            for bar in error_bars1[2] + error_bars2[2]:
                bar.set_linestyle(':')

            # Draw the vector arrow indicating the shift
            ax.annotate("", xy=(acc_rsn, ent_rsn), xytext=(acc_std, ent_std),
                        arrowprops=dict(arrowstyle="->", color=m_color, lw=1.5, shrinkA=8, shrinkB=8, mutation_scale=15),
                        zorder=2)

            # 3. Plot Standard Point (Cross 'X', white border to match dumbbell)
            ax.scatter(acc_std, ent_std, color=m_color, marker=std_marker, edgecolor="black", linewidth=0.8, s=150, zorder=3)

            # 4. Plot Reasoning Point (Circle 'o', black border to match dumbbell)
            ax.scatter(acc_rsn, ent_rsn, color=m_color, marker=rsn_marker, edgecolor="black", linewidth=0.8, s=130, zorder=4)

        if i == 0:
            ax.set_ylabel("Total Entropy (Instability)", fontsize=12, fontweight="bold")

    # --- Global X-Axis Label ---
    fig.supxlabel("Accuracy (%)", fontsize=12, fontweight="bold", y=0.08)

    # --- Create Unified Legend ---
    # Legend for Modes (Using neutral dark grey to show shape represents state)
    mode_handles = [
        mlines.Line2D([], [], color='white', marker=std_marker, markerfacecolor='#4A4A4A',
                      markeredgecolor='black', markersize=12, label='Without Reasoning'),
        mlines.Line2D([], [], color='white', marker=rsn_marker, markerfacecolor='#4A4A4A',
                      markeredgecolor='black', markersize=11, label='With Reasoning')
    ]

    # Legend for Models (Color patches)
    model_handles = [
        mpatches.Patch(facecolor=model_to_color[m], edgecolor='black', linewidth=0.6,
                       label=display_names[m]) for m in models
    ]

    # Combine handles and plot below subplots in a clean layout (4 items per row)
    all_handles = model_handles + mode_handles
    fig.legend(handles=all_handles,
               loc='lower center',
               bbox_to_anchor=(0.5, -0.08),
               ncol=4,
               frameon=False,
               fontsize=11,
               handletextpad=0.5,
               columnspacing=2.0)

    # Squeeze layout up to make room for supxlabel and legend
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig("figures/interplay_shift_plot_ci.pdf", bbox_inches="tight")
    plt.close()


def prepare_model_to_color():
    # 1. Assign a base sequential palette to each model family.
    model_labels = ["Qwen3-4B", "Qwen3-32B", "Qwen3-30B-A3B", "Seed-OSS-36B-Instruct", "NVIDIA-Nemotron-Nano-9B-v2", "NVIDIA-Nemotron-Nano-12B-v2"]
    family_palettes = {
        "Qwen": "Blues",
        "Seed": "mako",
        "NVIDIA-Nemotron": "Purples"
    }
    # 2. Group the exact model labels into their respective families
    family_groups = defaultdict(list)
    for model in model_labels:
        for family in family_palettes.keys():
            if model.startswith(family):
                family_groups[family].append(model)
                break
    # 3. Generate the model_to_color dictionary
    model_to_color = {}
    for family, models in family_groups.items():
        palette_name = family_palettes[family]
        # Drop the first shade, which is often too light/white
        colors = sns.color_palette(palette_name, n_colors=len(models) + 1)[1:]
        for i, model in enumerate(models):
            model_to_color[model] = colors[i]

    model_to_color["Seed-36B"] = model_to_color["Seed-OSS-36B-Instruct"]
    model_to_color["Nemotron-9B"] = model_to_color["NVIDIA-Nemotron-Nano-9B-v2"]
    model_to_color["Nemotron-12B"] = model_to_color["NVIDIA-Nemotron-Nano-12B-v2"]

    return model_to_color


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

    model_to_color = prepare_model_to_color()

    # Generate the visualization
    print(json.dumps(dataset_to_model_to_accuracy, indent=4))
    plot_interplay_shift(dataset_to_model_to_entropy, dataset_to_model_to_accuracy, models, model_to_color)