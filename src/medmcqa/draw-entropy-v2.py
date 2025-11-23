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


def get_masks_retained_keys(result_file):
    partial_response_label_to_idx_set = {}
    with open(result_file, "r") as f:
        for line in f:
            item = json.loads(line)
            idx = item["idx"]
            partial_response_label = item["partial_response_label"]
            answer_counts = item["answer_counts"]
            if answer_counts is None:
                continue
            total_count = sum(answer_counts.values())
            if total_count < 35:
                continue
            if partial_response_label not in partial_response_label_to_idx_set:
                partial_response_label_to_idx_set[partial_response_label] = set()
            partial_response_label_to_idx_set[partial_response_label].add(idx)
    return set.intersection(*partial_response_label_to_idx_set.values())


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
        "Qwen3-4B-Disable", "Qwen3-4B (1)", "Qwen3-4B (1-2)",
        "Qwen3-4B (1-3)", "Qwen3-4B (1-4)",
    ]

    # Extract metrics
    avg = [file_to_metrics[k]["avg"] for k in keys]
    ci = [file_to_metrics[k]["ci"] for k in keys]

    lower_err = [mean - lower for mean, (lower, upper) in zip(avg, ci)]
    upper_err = [upper - mean for mean, (lower, upper) in zip(avg, ci)]
    yerr = [lower_err, upper_err]

    x = np.arange(len(keys))

    # Color for both points and connecting line
    point_color = sns.color_palette("Set2", 1)[0]

    # ---- Create figure ----
    fig, ax = plt.subplots(dpi=1024)

    # ---- Draw line connecting points ----
    ax.plot(
        x, avg,
        linestyle="-",
        linewidth=1.5,
        color=point_color,
        alpha=0.9,
        zorder=2
    )

    # ---- Draw CI + points ----
    ax.errorbar(
        x, avg,
        yerr=yerr,
        fmt="o",
        markersize=6,
        capsize=5,
        elinewidth=1.2,
        color=point_color,
        markeredgecolor="black",
        markeredgewidth=0.6,
        alpha=1.0,
        zorder=3
    )

    # ---- CI numeric labels ----
    for i, (mean, (lower, upper)) in enumerate(zip(avg, ci)):
        ax.text(
            x[i], upper + 0.005, f"{upper:.3f}",
            ha="center", va="bottom",
            fontsize=8, fontweight="bold"
        )
        ax.text(
            x[i], lower - 0.005, f"{lower:.3f}",
            ha="center", va="top",
            fontsize=8, fontweight="bold"
        )

    # ---- X-labels: remove the prefix “Qwen3-4B-” ----
    clean_labels = ["None", "Step 1", "Steps 1-2", "Steps 1-3", "Steps 1-4"]

    ax.set_xticks(x)
    ax.set_xticklabels(clean_labels, rotation=0, ha="center", fontweight="bold")

    # ---- Custom bold xlabel in same color ----
    ax.text(
        0.5, -0.1,
        "Qwen3-4B",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=11, fontweight="bold",
        color=point_color
    )

    # ---- Y and title ----
    ax.set_ylabel("Entropy", fontsize=11, fontweight="bold")
    ax.set_title("MedMCQA – Entropy (Mean ± 95% CI)", pad=15, weight="bold")

    # ---- Grid + styling ----
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    save_file = "outputs/medmcqa/figures/entropy-masks.png"
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
            key = "Qwen3-4B (1-4)"
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

        accuracy_list = []
        with open(file_name.replace("_counts.jsonl", "_correctness.jsonl"), "r") as f:
            for line in f:
                item = json.loads(line)
                idx = item["idx"]
                if idx not in retained_ids:
                    continue
                correct_count = item["correct_count"]
                total_count = item["total_count"]
                accuracy = correct_count / total_count
                accuracy_list.append(accuracy)
        file_to_metrics[key]["avg_accuracy"] = np.mean(accuracy_list)

        print(key)
        print(f"retained_ids: {len(retained_ids)}")
        print("avg:", f"{file_to_metrics[key]['avg']:.4f}")
        print("ci:", f"{file_to_metrics[key]['ci'][0]:.4f} - {file_to_metrics[key]['ci'][1]:.4f}")
        print("avg_accuracy:", f"{file_to_metrics[key]['avg_accuracy']:.4f}")
        print("-" * 100)

    return file_to_metrics

def get_masks_statistics(result_file, retained_ids):
    file_to_metrics = {}
    for partial_response_label in ["step_1", "step_1,step_2", "step_1,step_2,step_3"]:
        if partial_response_label == "step_1":
            key = "Qwen3-4B (1)"
        elif partial_response_label == "step_1,step_2":
            key = "Qwen3-4B (1-2)"
        elif partial_response_label == "step_1,step_2,step_3":
            key = "Qwen3-4B (1-3)"
        else:
            assert False, f"Unknown partial response label: {partial_response_label}"
        file_to_metrics[key] = {}
        entropy_list = []
        with open(result_file, "r") as f:
            for line in f:
                item = json.loads(line)
                idx = item["idx"]
                if idx not in retained_ids:
                    continue
                cur_partial_response_label = item["partial_response_label"]
                if cur_partial_response_label != partial_response_label:
                    continue
                answer_counts = item["answer_counts"]
                total_count = sum(answer_counts.values())
                if total_count < 35:
                    continue
                distribution = []
                for answer, count in answer_counts.items():
                    distribution.append(count / total_count)
                entropy_list.append(entropy(np.array(distribution)))
        mean = np.mean(entropy_list)
        n = len(entropy_list)
        sem = np.std(entropy_list) / np.sqrt(n)
        ci = stats.t.interval(0.95, n-1, loc=mean, scale=sem)
        file_to_metrics[key]["avg"] = mean
        file_to_metrics[key]["ci"] = ci

        accuracy_list = []
        with open(result_file.replace("_counts.jsonl", "_correctness.jsonl"), "r") as f:
            for line in f:
                item = json.loads(line)
                idx = item["idx"]
                if idx not in retained_ids:
                    continue
                cur_partial_response_label = item["partial_response_label"]
                if cur_partial_response_label != partial_response_label:
                    continue
                correct_count = item["correct_count"]
                total_count = item["total_count"]
                if total_count < 35:
                    continue
                accuracy = correct_count / total_count
                accuracy_list.append(accuracy)
        file_to_metrics[key]["avg_accuracy"] = np.mean(accuracy_list)

        print(key)
        print(f"retained_ids: {len(retained_ids)}")
        print("avg:", f"{file_to_metrics[key]['avg']:.4f}")
        print("ci:", f"{file_to_metrics[key]['ci'][0]:.4f} - {file_to_metrics[key]['ci'][1]:.4f}")
        print("avg_accuracy:", f"{file_to_metrics[key]['avg_accuracy']:.4f}")
        print("-" * 100)

    return file_to_metrics

if __name__ == "__main__":
    retained_ids_list = []
    retained_ids = get_retained_keys([
        "outputs/medmcqa/processed_results/Qwen3-4B_temp0.6_n50_dt_counts.jsonl",
        "outputs/medmcqa/processed_results/Qwen3-4B_temp0.6_n50_counts.jsonl",
    ])
    print(len(retained_ids))
    # masks_retained_ids = get_masks_retained_keys("outputs/medmcqa/processed_results/Qwen3-4B_temp0.6_n50_masks_completion_counts.jsonl")
    # print(len(masks_retained_ids))
    # retained_ids = set.intersection(retained_ids, masks_retained_ids)
    # print(len(retained_ids))


    res1 = get_statistics([
        "outputs/medmcqa/processed_results/Qwen3-4B_temp0.6_n50_dt_counts.jsonl",
        "outputs/medmcqa/processed_results/Qwen3-4B_temp0.6_n50_counts.jsonl",
    ], [retained_ids, retained_ids])

    res2 = get_masks_statistics("outputs/medmcqa/processed_results/Qwen3-4B_temp0.6_n50_masks_completion_counts.jsonl", retained_ids)

    res = {**res1, **res2}
    plot_statistics(res)
