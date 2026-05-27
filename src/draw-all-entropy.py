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


def p_to_stars(p):
    """
    Convert p-value to significance stars.
    """
    if math.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


def paired_entropy_test_one_sided(file_to_metrics, model_base):
    """
    One-sided paired Wilcoxon test:
    H1: entropy(with reasoning) < entropy(without reasoning)
    Returns p-value (float).
    """
    k_non = f"{model_base}-Disable"
    k_reason = f"{model_base}"

    a = file_to_metrics[k_non]["idx_to_entropy"]
    b = file_to_metrics[k_reason]["idx_to_entropy"]

    common = sorted(set(a.keys()) & set(b.keys()))
    if len(common) == 0:
        return float("nan")

    x_non = np.array([a[i] for i in common], dtype=float)
    x_reason = np.array([b[i] for i in common], dtype=float)

    diff = x_reason - x_non  # want diff < 0
    if np.allclose(diff, 0):
        return 1.0

    res = stats.wilcoxon(x_reason, x_non, alternative="less", zero_method="wilcox")
    return float(res.pvalue)


def add_sig_bracket(ax, x1, x2, y, h, text, fontsize=11):
    # bracket line
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            lw=1.0, c="black", clip_on=False)
    # stars
    ax.text((x1 + x2) / 2, y + h, text,
            ha="center", va="bottom", fontsize=fontsize, fontweight="bold",
            color="black", clip_on=False)


def draw_entropy_bars_on_ax(ax, file_to_metrics, dataset_name, show_xlabel=True, show_ylabel=True, model_to_color=None):
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.edgecolor": "gray",
        "axes.linewidth": 0.8,
        # "text.usetex": True,
    })

    keys = [
        "Qwen3-4B-Disable", "Qwen3-4B",
        "Qwen3-32B-Disable", "Qwen3-32B",
        "Qwen3-30B-A3B-Disable", "Qwen3-30B-A3B",
        "Seed-36B-Disable", "Seed-36B",
        "Nemotron-9B-Disable", "Nemotron-9B",
        "Nemotron-12B-Disable", "Nemotron-12B",
    ]
    model_labels = ["Qwen3-4B", "Qwen3-32B", "Qwen3-30B-A3B", "Seed-36B", "Nemotron-9B", "Nemotron-12B"]

    avg = [file_to_metrics[k]["avg"] for k in keys]
    ci = [file_to_metrics[k]["ci"] for k in keys]
    yerr = [upper - mean for mean, (lower, upper) in zip(avg, ci)]

    # x = np.arange(len(keys))
    # bar_width = 0.8
    n_models = len(model_labels)
    intra_gap = 1.1
    inter_gap = 1.6
    x = []
    pos = 0.0
    for _ in range(n_models):
        x.append(pos)
        x.append(pos + intra_gap)
        pos += (inter_gap + intra_gap)
    x = np.array(x)
    bar_width = 1.0

    # 6 colors for 6 models
    # palette = sns.color_palette("Set2", len(model_labels))
    # model_to_color = {m: palette[i] for i, m in enumerate(model_labels)}

    # map each key -> its model label
    def key_to_model(k: str) -> str:
        # keys are like "Qwen3-4B-Disable" or "Qwen3-4B"
        # strip "-Disable" if present to recover model name
        return k.replace("-Disable", "")

    bars = []
    for i, (mean, err, key) in enumerate(zip(avg, yerr, keys)):
        model = key_to_model(key)
        color = model_to_color[model]

        is_non_reasoning = ("Disable" in key)     # empty style
        hatch = None if is_non_reasoning else "//" # reasoning: hatch

        bar = ax.bar(
            x[i], mean, bar_width,
            yerr=err, capsize=5,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            alpha=1.0,
            hatch=hatch
        )
        bars.append(bar[0])

    # CI labels
    for i, bar in enumerate(bars):
        lower, upper = ci[i]
        center = bar.get_x() + bar.get_width() / 2
        ax.text(center, upper + 0.015, f"{upper:.3f}",
                ha="center", va="bottom", fontsize=8, color="black", fontweight="bold")
        ax.text(center, lower - 0.025, f"{lower:.3f}",
                ha="center", va="top", fontsize=8, color="black", fontweight="bold")

    # ---------- p-values as x-tick labels (at pair centers) ----------
    # one-sided paired Wilcoxon: H_with < H_without
    pvals = [paired_entropy_test_one_sided(file_to_metrics, m) for m in model_labels]
    p_texts = [p_to_stars(p) for p in pvals]

    # pair centers (center between the two bars in each model)
    pair_centers = []
    for j in range(len(model_labels)):
        left_bar = bars[2 * j]
        right_bar = bars[2 * j + 1]
        c1 = left_bar.get_x() + left_bar.get_width() / 2
        c2 = right_bar.get_x() + right_bar.get_width() / 2
        pair_centers.append((c1 + c2) / 2)

    if show_xlabel:
        ax.set_xticks(pair_centers)
        ax.set_xticklabels(p_texts, fontsize=15)
        ax.tick_params(axis="x", length=0, pad=6)  # length=0 removes tick marks
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])

    # no per-axes labels (you want shared labels only)
    ax.set_xlabel("")
    ax.set_ylabel("")

    dataset_name_to_title = {
        "daily_dilemmas": r"Ethic ($\it{DailyDilemmas}$)",
        "medmcqa": r"Medicine ($\it{MedMCQA}$)",
        "mmlu-accounting": r"Finance ($\it{MMLU\!-\!Accounting}$)",
        "mmlu-law": r"Law ($\it{MMLU\!-\!Law}$)",
    }
    ax.set_title(f"{dataset_name_to_title[dataset_name]}", pad=10, weight="bold")

    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_all_datasets(metrics_by_dataset, save_file="figures/entropy-all.png", model_to_color=None):
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), dpi=1024)

    dataset_order = [
        "daily_dilemmas",
        "medmcqa",
        "mmlu-accounting",
        "mmlu-law",
    ]

    for idx, dataset_name in enumerate(dataset_order):
        r, c = divmod(idx, 2)
        ax = axes[r, c]
        draw_entropy_bars_on_ax(
            ax=ax,
            file_to_metrics=metrics_by_dataset[dataset_name],
            dataset_name=dataset_name,
            show_xlabel=True,
            show_ylabel=(c == 0),
            model_to_color=model_to_color,
        )

    # --- Shared labels (tight spacing) ---
    # fig.supxlabel("Model", fontsize=12, fontweight="bold", y=0.02)
    fig.supylabel("Entropy (Instability)", fontsize=12, fontweight="bold", x=0.06)

    # --- Legend: 6 model colors + 2 style boxes (empty vs hatched) ---
    model_labels = ["Qwen3-4B", "Qwen3-32B", "Qwen3-30B-A3B", "Seed-OSS-36B-Instruct", "NVIDIA-Nemotron-Nano-9B-v2", "NVIDIA-Nemotron-Nano-12B-v2"]

    model_handles = [
        mpatches.Patch(facecolor=model_to_color[model_labels[i]], edgecolor="black", label=model_labels[i])
        for i in range(len(model_labels))
    ]

    style_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", label="Without Reasoning"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", label="With Reasoning"),
    ]

    # Put legend at the bottom, two rows: models then styles (or vice versa)
    handles = model_handles + style_handles
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,                  # adjust to taste (e.g., 4 or 5)
        frameon=False,
        bbox_to_anchor=(0.53, 0.01),
        fontsize=12,
        handlelength=1.6,
        columnspacing=1.6,
        handletextpad=0.5,
    )

    # leave room for legend + sup labels
    plt.tight_layout(rect=[0.05, 0.10, 1.0, 1.0])
    plt.savefig(save_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_file}")


def compute_file_to_metrics(result_files, retained_ids_list, dataset_name):
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

        if dataset_name == "daily_dilemmas":
            file_to_metrics[key] = {}
            idx_to_results = {}
            retained_ids = retained_ids_list[i]
            idx_to_entropy = {}
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
                idx_to_entropy[idx] = entropy_value
                entropy_list.append(entropy_value)

            mean = np.mean(entropy_list)
            n = len(entropy_list)
            sem = stats.sem(entropy_list)
            ci = stats.t.interval(0.95, n-1, loc=mean, scale=sem)
            file_to_metrics[key]["avg"] = mean
            file_to_metrics[key]["ci"] = ci
            file_to_metrics[key]["entropy_list"] = entropy_list
            file_to_metrics[key]["idx_to_entropy"] = idx_to_entropy

        else:
            file_to_metrics[key] = {}
            retained_ids = retained_ids_list[i]
            entropy_list = []
            idx_to_entropy = {}
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
                    idx_to_entropy[idx] = entropy_value
                    entropy_list.append(entropy_value)

            mean = np.mean(entropy_list)
            n = len(entropy_list)
            sem = stats.sem(entropy_list)
            ci = stats.t.interval(0.95, n-1, loc=mean, scale=sem)
            file_to_metrics[key]["avg"] = mean
            file_to_metrics[key]["ci"] = ci
            file_to_metrics[key]["entropy_list"] = entropy_list
            file_to_metrics[key]["idx_to_entropy"] = idx_to_entropy

        print(key)
        print(f"retained_ids: {len(retained_ids)}")
        print("avg:", f"{file_to_metrics[key]['avg']:.4f}")
        print("ci:", f"{file_to_metrics[key]['ci'][0]:.4f} - {file_to_metrics[key]['ci'][1]:.4f}")
        print("-" * 100)

    return file_to_metrics


if __name__ == "__main__":
    datasets = [
        "daily_dilemmas",
        "medmcqa",
        "mmlu-accounting",
        "mmlu-law",
    ]

    metrics_by_dataset = {}

    for dataset_name in datasets:
        subfix = "_counts" if dataset_name != "daily_dilemmas" else ""

        retained_ids_list = []
        retained_ids = get_retained_keys([
            f"outputs/{dataset_name}/processed_results/Qwen3-4B_temp0.6_n50_dt{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/Qwen3-4B_temp0.6_n50{subfix}.jsonl",
        ], dataset_name)
        retained_ids_list += [retained_ids] * 2
        retained_ids = get_retained_keys([
            f"outputs/{dataset_name}/processed_results/Qwen3-32B_temp0.6_n50_dt{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/Qwen3-32B_temp0.6_n50{subfix}.jsonl",
        ], dataset_name)
        retained_ids_list += [retained_ids] * 2
        retained_ids = get_retained_keys([
            f"outputs/{dataset_name}/processed_results/Qwen3-30B-A3B_temp0.6_n50_dt{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/Qwen3-30B-A3B_temp0.6_n50{subfix}.jsonl",
        ], dataset_name)
        retained_ids_list += [retained_ids] * 2
        retained_ids = get_retained_keys([
            f"outputs/{dataset_name}/processed_results/Seed-OSS-36B-Instruct_temp1.1_n50_dt{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/Seed-OSS-36B-Instruct_temp1.1_n50{subfix}.jsonl",
        ], dataset_name)
        retained_ids_list += [retained_ids] * 2
        retained_ids = get_retained_keys([
            f"outputs/{dataset_name}/processed_results/NVIDIA-Nemotron-Nano-9B-v2_temp0.6_n50_dt{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/NVIDIA-Nemotron-Nano-9B-v2_temp0.6_n50{subfix}.jsonl",
        ], dataset_name)
        retained_ids_list += [retained_ids] * 2
        retained_ids = get_retained_keys([
            f"outputs/{dataset_name}/processed_results/NVIDIA-Nemotron-Nano-12B-v2_temp0.6_n50_dt{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/NVIDIA-Nemotron-Nano-12B-v2_temp0.6_n50{subfix}.jsonl",
        ], dataset_name)
        retained_ids_list += [retained_ids] * 2

        result_files = [
            f"outputs/{dataset_name}/processed_results/Qwen3-4B_temp0.6_n50_dt{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/Qwen3-4B_temp0.6_n50{subfix}.jsonl",

            f"outputs/{dataset_name}/processed_results/Qwen3-32B_temp0.6_n50_dt{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/Qwen3-32B_temp0.6_n50{subfix}.jsonl",

            f"outputs/{dataset_name}/processed_results/Qwen3-30B-A3B_temp0.6_n50_dt{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/Qwen3-30B-A3B_temp0.6_n50{subfix}.jsonl",

            f"outputs/{dataset_name}/processed_results/Seed-OSS-36B-Instruct_temp1.1_n50_dt{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/Seed-OSS-36B-Instruct_temp1.1_n50{subfix}.jsonl",

            f"outputs/{dataset_name}/processed_results/NVIDIA-Nemotron-Nano-9B-v2_temp0.6_n50_dt{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/NVIDIA-Nemotron-Nano-9B-v2_temp0.6_n50{subfix}.jsonl",

            f"outputs/{dataset_name}/processed_results/NVIDIA-Nemotron-Nano-12B-v2_temp0.6_n50_dt{subfix}.jsonl",
            f"outputs/{dataset_name}/processed_results/NVIDIA-Nemotron-Nano-12B-v2_temp0.6_n50{subfix}.jsonl",
        ]

        file_to_metrics = compute_file_to_metrics(
            result_files=result_files,
            retained_ids_list=retained_ids_list,
            dataset_name=dataset_name,
        )
        metrics_by_dataset[dataset_name] = file_to_metrics

    for dataset in datasets:
        print(f"Dataset: {dataset}")
        for m in ["Qwen3-4B", "Qwen3-32B", "Qwen3-30B-A3B", "Seed-36B", "Nemotron-9B", "Nemotron-12B"]:
            p = paired_entropy_test_one_sided(metrics_by_dataset[dataset], m)
            print(f"{dataset} {m}: {p:.8f}")
        print("-" * 100)


    # 1. Assign a base sequential palette to each model family.
    # You can change these to "Purples", "Oranges", "mako", "flare", etc.
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
        # We ask for `len(models) + 1` colors and slice `[1:]` to drop the very
        # first shade, which is often too light/white to see clearly on a white background.
        colors = sns.color_palette(palette_name, n_colors=len(models) + 1)[1:]
        for i, model in enumerate(models):
            model_to_color[model] = colors[i]

    model_to_color["Seed-36B"] = model_to_color["Seed-OSS-36B-Instruct"]
    model_to_color["Nemotron-9B"] = model_to_color["NVIDIA-Nemotron-Nano-9B-v2"]
    model_to_color["Nemotron-12B"] = model_to_color["NVIDIA-Nemotron-Nano-12B-v2"]

    plot_all_datasets(metrics_by_dataset, save_file="figures/entropy-all.png", model_to_color=model_to_color)
