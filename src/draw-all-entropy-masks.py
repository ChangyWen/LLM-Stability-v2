import json
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
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
            partial_response_label_to_idx_set.setdefault(partial_response_label, set()).add(idx)
    return set.intersection(*partial_response_label_to_idx_set.values())


def plot_combined_statistics(all_results):
    save_file = "figures/entropy-masks_combined.png"
    # --- Global style (modern, consistent) ---
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.edgecolor": "gray",
        "axes.linewidth": 0.8,
    })

    # Create 1 row, 2 columns layout with shared Y-axis
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=1024, sharey=True)

    # One main color (Set2) like your recent figs
    point_color = sns.color_palette("Set2", 1)[0]

    for ax, (file_to_metrics, dataset, model) in zip(axes, all_results):
        # Order you want to show
        keys = [
            f"{model}-Disable",
            f"{model} (1)",
            f"{model} (1-2)",
            f"{model} (1-3)",
            f"{model} (1-4)",
        ]

        # Labels shown on x-axis
        x_labels = ["None", "Step 1", "Steps 1-2", "Steps 1-3", "Steps 1-4"]

        # Extract metrics
        avg = [file_to_metrics[k]["avg"] for k in keys]
        ci = [file_to_metrics[k]["ci"] for k in keys]

        # asymmetric CI errors
        lower_err = [mean - lower for mean, (lower, upper) in zip(avg, ci)]
        upper_err = [upper - mean for mean, (lower, upper) in zip(avg, ci)]
        yerr = [lower_err, upper_err]

        x = np.arange(len(keys))

        # ---- Line behind points (subtle) ----
        ax.plot(
            x, avg,
            linestyle="--",
            linewidth=2.2,
            color=point_color,
            alpha=0.85,
            zorder=2
        )

        # ---- Errorbar + points (on top) ----
        eb = ax.errorbar(
            x, avg,
            yerr=yerr,
            fmt="o",
            markersize=12.5,
            capsize=10,
            elinewidth=2.2,
            color=point_color,
            markerfacecolor=point_color,
            markeredgecolor="black",
            markeredgewidth=2.2,
            alpha=1.0,
            zorder=3
        )

        # ---- CI numeric labels (bold) ----
        for i, (mean, (lower, upper)) in enumerate(zip(avg, ci)):
            ax.text(
                x[i], upper + 0.006, f"{upper:.3f}",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold"
            )
            ax.text(
                x[i], lower - 0.006, f"{lower:.3f}",
                ha="center", va="top",
                fontsize=8, fontweight="bold"
            )

        # ---- X axis ----
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=0, ha="center")

        # ---- Titles ----
        dataset_formatted = "MedMCQA" if dataset == "medmcqa" else "MMLU-Accounting"
        ax.set_title(f"{dataset_formatted} – {model}", pad=12, weight="bold")

        # ---- Grid + spines ----
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    # ---- Sup Labels for the entire figure ----
    fig.supylabel("Entropy (Decision-making Stability)", fontsize=13, fontweight="bold")
    fig.supxlabel("Reasoning Step(s)", fontsize=13, fontweight="bold")

    # Tighten margins so labels don’t float too far
    plt.tight_layout()
    plt.savefig(save_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_file}")


def get_statistics(result_files, retained_ids_list, dataset, model):
    file_to_metrics = {}

    for i, file_name in enumerate(result_files):
        if model in file_name and ("masks_completion" in file_name):
            # handled elsewhere
            continue

        if model in file_name and ("dt" in file_name):
            key = f"{model}-Disable"
        elif model in file_name and ("dt" not in file_name):
            key = f"{model} (1-4)"
        else:
            raise ValueError(f"Unexpected file name in this script: {file_name}")

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
                if answer_counts is None:
                    continue
                total_count = sum(answer_counts.values())
                if total_count <= 0:
                    continue
                distribution = [count / total_count for count in answer_counts.values()]
                entropy_list.append(entropy(np.array(distribution)))

        entropy_list = np.array(entropy_list, dtype=float)
        mean = float(np.mean(entropy_list))
        n = int(len(entropy_list))
        sem = float(np.std(entropy_list, ddof=1) / np.sqrt(n)) if n >= 2 else 0.0
        ci = stats.t.interval(0.95, n - 1, loc=mean, scale=sem) if n >= 2 else (mean, mean)

        file_to_metrics[key]["avg"] = mean
        file_to_metrics[key]["ci"] = ci

        print(key)
        print(f"retained_ids: {len(retained_ids)}")
        print("avg:", f"{mean:.4f}")
        print("ci:", f"{ci[0]:.4f} - {ci[1]:.4f}")
        print("-" * 100)

    return file_to_metrics


def get_masks_statistics(result_file, retained_ids, model):
    file_to_metrics = {}
    mapping = {
        "step_1": f"{model} (1)",
        "step_1,step_2": f"{model} (1-2)",
        "step_1,step_2,step_3": f"{model} (1-3)",
    }

    for partial_response_label, key in mapping.items():
        entropy_list = []
        with open(result_file, "r") as f:
            for line in f:
                item = json.loads(line)
                idx = item["idx"]
                if idx not in retained_ids:
                    continue
                if item["partial_response_label"] != partial_response_label:
                    continue
                answer_counts = item["answer_counts"]
                if answer_counts is None:
                    continue
                total_count = sum(answer_counts.values())
                if total_count < 35:
                    continue
                distribution = [count / total_count for count in answer_counts.values()]
                entropy_list.append(entropy(np.array(distribution)))

        entropy_list = np.array(entropy_list, dtype=float)
        mean = float(np.mean(entropy_list))
        n = int(len(entropy_list))
        sem = float(np.std(entropy_list, ddof=1) / np.sqrt(n)) if n >= 2 else 0.0
        ci = stats.t.interval(0.95, n - 1, loc=mean, scale=sem) if n >= 2 else (mean, mean)

        file_to_metrics[key] = {"avg": mean, "ci": ci}

        print(key)
        print(f"retained_ids: {len(retained_ids)}")
        print("avg:", f"{mean:.4f}")
        print("ci:", f"{ci[0]:.4f} - {ci[1]:.4f}")
        print("-" * 100)

    return file_to_metrics


if __name__ == "__main__":
    datasets_models = [
        ("medmcqa", "Qwen3-4B"),
        ("mmlu-accounting", "NVIDIA-Nemotron-Nano-9B-v2"),
    ]

    all_results = []

    for dataset, model in datasets_models:
        retained_ids = get_retained_keys([
            f"outputs/{dataset}/processed_results/{model}_temp0.6_n50_dt_counts.jsonl",
            f"outputs/{dataset}/processed_results/{model}_temp0.6_n50_counts.jsonl",
        ])
        print("Retained:", len(retained_ids))

        res1 = get_statistics(
            [
                f"outputs/{dataset}/processed_results/{model}_temp0.6_n50_dt_counts.jsonl",
                f"outputs/{dataset}/processed_results/{model}_temp0.6_n50_counts.jsonl",
            ],
            [retained_ids, retained_ids],
            dataset,
            model,
        )

        res2 = get_masks_statistics(
            f"outputs/{dataset}/processed_results/{model}_temp0.6_n50_masks_completion_counts.jsonl",
            retained_ids,
            model,
        )

        res = {**res1, **res2}

        # Append to our list of results rather than plotting immediately
        all_results.append((res, dataset, model))

    # Plot everything in one big 1x2 figure
    plot_combined_statistics(all_results)