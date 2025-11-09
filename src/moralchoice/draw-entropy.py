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
    ax.set_title("MoralChoice - Entropy (Mean with 95% CI)")
    plt.grid(True)  # Add grid lines
    save_file = "outputs/moralchoice/figures/moralchoice_entropy.png"
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
        print(key)
        print(f"retained_ids: {len(retained_ids)}")
        print("avg:", f"{file_to_metrics[key]['avg']:.4f}")
        print("ci:", f"{file_to_metrics[key]['ci'][0]:.4f} - {file_to_metrics[key]['ci'][1]:.4f}")
        print("-" * 100)

    plot_statistics(file_to_metrics)


if __name__ == "__main__":
    retained_ids_list = []
    retained_ids = get_retained_keys([
        "outputs/moralchoice/processed_results/Qwen3-4B_temp0.6_n50_dt.jsonl",
        "outputs/moralchoice/processed_results/Qwen3-4B_temp0.6_n50.jsonl",
    ])
    retained_ids_list += [retained_ids] * 2
    retained_ids = get_retained_keys([
        "outputs/moralchoice/processed_results/Qwen3-32B_temp0.6_n50_dt.jsonl",
        "outputs/moralchoice/processed_results/Qwen3-32B_temp0.6_n50.jsonl",
    ])
    retained_ids_list += [retained_ids] * 2
    retained_ids = get_retained_keys([
        "outputs/moralchoice/processed_results/Qwen3-30B-A3B_temp0.6_n50_dt.jsonl",
        "outputs/moralchoice/processed_results/Qwen3-30B-A3B_temp0.6_n50.jsonl",
    ])
    retained_ids_list += [retained_ids] * 2
    # retained_ids = get_retained_keys([
    #     "outputs/moralchoice/processed_results/Seed-OSS-36B-Instruct_temp1.1_n50_dt.jsonl",
    #     "outputs/moralchoice/processed_results/Seed-OSS-36B-Instruct_temp1.1_n50.jsonl",
    # ])
    # retained_ids_list += [retained_ids] * 2

    get_statistics([
        "outputs/moralchoice/processed_results/Qwen3-4B_temp0.6_n50_dt.jsonl",
        "outputs/moralchoice/processed_results/Qwen3-4B_temp0.6_n50.jsonl",

        "outputs/moralchoice/processed_results/Qwen3-32B_temp0.6_n50_dt.jsonl",
        "outputs/moralchoice/processed_results/Qwen3-32B_temp0.6_n50.jsonl",

        "outputs/moralchoice/processed_results/Qwen3-30B-A3B_temp0.6_n50_dt.jsonl",
        "outputs/moralchoice/processed_results/Qwen3-30B-A3B_temp0.6_n50.jsonl",

        # "outputs/moralchoice/processed_results/Seed-OSS-36B-Instruct_temp1.1_n50_dt.jsonl",
        # "outputs/moralchoice/processed_results/Seed-OSS-36B-Instruct_temp1.1_n50.jsonl",
    ], retained_ids_list)
