import json
import math
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from scipy.stats import fisher_exact, MonteCarloMethod
from scipy.stats import entropy
from scipy import stats


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


def plot_statistics(file_to_metrics):
    plt.rc('font', weight='normal', size=10)
    keys = [
        "Qwen3-4B-Disable", "Qwen3-4B",
        "Qwen3-32B-Disable", "Qwen3-32B",
        "Qwen3-30B-A3B-Disable", "Qwen3-30B-A3B",
        "Seed-36B-Disable", "Seed-36B",
    ]
    avg = [file_to_metrics[k]["avg"] for k in keys]
    ci = [file_to_metrics[k]["ci"] for k in keys]
    yerr = [upper - mean for mean, (lower, upper) in zip(avg, ci)]

    x = np.arange(len(keys))
    bar_width = 0.8

    colors = ["red"] * 2 + ["blue"] * 2 + ["brown"] * 2 + ["green"] * 2

    fig, ax = plt.subplots(dpi=1024)
    bars = ax.bar(
        x, avg, bar_width,
        yerr=yerr, capsize=5,
        color=colors, edgecolor="black"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=80, ha="center")  # center alignment
    ax.set_ylabel("Entropy")
    ax.set_title("MedMCQA - Entropy (Mean with 95% CI)")
    plt.grid(True)  # Add grid lines
    save_file = "outputs/medmcqa/figures/medmcqa_entropy.png"
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
        print(key)
        print(f"retained_ids: {len(retained_ids)}")
        print("avg:", f"{file_to_metrics[key]['avg']:.4f}")
        print("ci:", f"{file_to_metrics[key]['ci'][0]:.4f} - {file_to_metrics[key]['ci'][1]:.4f}")
        print("-" * 100)

    plot_statistics(file_to_metrics)


if __name__ == "__main__":
    retained_ids_list = []
    retained_ids = get_retained_keys([
        "outputs/medmcqa/processed_results/Qwen3-4B_temp0.6_n50_dt_counts.jsonl",
        "outputs/medmcqa/processed_results/Qwen3-4B_temp0.6_n50_counts.jsonl",
    ])
    retained_ids_list += [retained_ids] * 2
    retained_ids = get_retained_keys([
        "outputs/medmcqa/processed_results/Qwen3-32B_temp0.6_n50_dt_counts.jsonl",
        "outputs/medmcqa/processed_results/Qwen3-32B_temp0.6_n50_counts.jsonl",
    ])
    retained_ids_list += [retained_ids] * 2
    retained_ids = get_retained_keys([
        "outputs/medmcqa/processed_results/Qwen3-30B-A3B_temp0.6_n50_dt_counts.jsonl",
        "outputs/medmcqa/processed_results/Qwen3-30B-A3B_temp0.6_n50_counts.jsonl",
    ])
    retained_ids_list += [retained_ids] * 2
    retained_ids = get_retained_keys([
        "outputs/medmcqa/processed_results/Seed-OSS-36B-Instruct_temp1.1_n50_dt_counts.jsonl",
        "outputs/medmcqa/processed_results/Seed-OSS-36B-Instruct_temp1.1_n50_counts.jsonl",
    ])
    retained_ids_list += [retained_ids] * 2

    get_statistics([
        "outputs/medmcqa/processed_results/Qwen3-4B_temp0.6_n50_dt_counts.jsonl",
        "outputs/medmcqa/processed_results/Qwen3-4B_temp0.6_n50_counts.jsonl",

        "outputs/medmcqa/processed_results/Qwen3-32B_temp0.6_n50_dt_counts.jsonl",
        "outputs/medmcqa/processed_results/Qwen3-32B_temp0.6_n50_counts.jsonl",

        "outputs/medmcqa/processed_results/Qwen3-30B-A3B_temp0.6_n50_dt_counts.jsonl",
        "outputs/medmcqa/processed_results/Qwen3-30B-A3B_temp0.6_n50_counts.jsonl",

        "outputs/medmcqa/processed_results/Seed-OSS-36B-Instruct_temp1.1_n50_dt_counts.jsonl",
        "outputs/medmcqa/processed_results/Seed-OSS-36B-Instruct_temp1.1_n50_counts.jsonl",
    ], retained_ids_list)
