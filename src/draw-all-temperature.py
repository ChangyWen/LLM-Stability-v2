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


# ----------------------------
# Statistical Functions
# ----------------------------
def p_to_stars(p):
    """Convert p-value to significance stars."""
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

def paired_entropy_test_one_sided(file_to_metrics, k_non, k_reason):
    """
    One-sided paired Wilcoxon test:
    H1: entropy(with reasoning) < entropy(without reasoning)
    Returns p-value (float).
    """
    if k_non not in file_to_metrics or k_reason not in file_to_metrics:
        return float("nan")

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


# ----------------------------
# Retained Keys Extraction
# ----------------------------
def get_retained_keys(result_files, dataset_name):
    if dataset_name == "daily_dilemmas":
        retained_ids_list = []
        for file_name in result_files:
            with open(file_name, "r") as f:
                idx_to_options = {}
                for line in f:
                    item = json.loads(line)
                    idx = item["idx"]
                    idx_to_options.setdefault(idx, [])
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

                    if "temp0.0" in file_name:
                        if total_count < 1:
                            continue
                    else:
                        if total_count < 35:
                            continue

                    idx_set.add(idx)
                retained_ids_list.append(idx_set)
        return set.intersection(*retained_ids_list)


# --------------------------------------------------------
# Compute Metrics (Updated to retain idx_to_entropy)
# --------------------------------------------------------
def compute_file_to_metrics_for_model(dataset_name, model_name, temperatures, base_dir="outputs"):
    subfix = "_counts" if dataset_name != "daily_dilemmas" else ""

    result_files = []
    for t in temperatures:
        result_files += [
            f"{base_dir}/{dataset_name}/processed_results/{model_name}_temp{t}_n50_dt{subfix}.jsonl",
            f"{base_dir}/{dataset_name}/processed_results/{model_name}_temp{t}_n50{subfix}.jsonl",
        ]

    retained_ids = get_retained_keys(result_files, dataset_name)
    retained_ids_list = [retained_ids] * len(result_files)

    file_to_metrics = {}

    for i, file_name in enumerate(result_files):
        is_non_reasoning = ("_dt" in file_name)
        fname = file_name.split("/")[-1]
        temp_part = [p for p in fname.split("_") if p.startswith("temp")][0]
        temp_str = temp_part.replace("temp", "")

        key = f"{model_name}-Disable_temp{temp_str}" if is_non_reasoning else f"{model_name}_temp{temp_str}"
        file_to_metrics[key] = {}

        retained = retained_ids_list[i]
        entropy_list = []
        idx_to_entropy = {}

        if dataset_name == "daily_dilemmas":
            idx_to_results = {}
            with open(file_name, "r") as f:
                for line in f:
                    item = json.loads(line)
                    idx = item["idx"]
                    if idx not in retained:
                        continue
                    idx_to_results.setdefault(idx, [])
                    if item["option"] in ["1", "2", "3"]:
                        idx_to_results[idx].append(item["option"])

            for idx, results in idx_to_results.items():
                cnt = Counter(results)
                dist = np.array([
                    cnt.get("1", 0) / len(results),
                    cnt.get("2", 0) / len(results),
                    cnt.get("3", 0) / len(results),
                ])
                e = entropy(dist)
                entropy_list.append(e)
                idx_to_entropy[idx] = e

        else:
            with open(file_name, "r") as f:
                for line in f:
                    item = json.loads(line)
                    idx = item["idx"]
                    if idx not in retained:
                        continue
                    answer_counts = item["answer_counts"]
                    if answer_counts is None:
                        continue
                    total = sum(answer_counts.values())
                    if total <= 0:
                        continue
                    dist = np.array([c / total for c in answer_counts.values()])
                    e = entropy(dist)
                    entropy_list.append(e)
                    idx_to_entropy[idx] = e

        entropy_list = np.array(entropy_list, dtype=float)
        mean = float(np.mean(entropy_list))
        n = len(entropy_list)
        if n >= 2:
            sem = float(np.std(entropy_list, ddof=1) / np.sqrt(n))
            ci = stats.t.interval(0.95, n - 1, loc=mean, scale=sem)
        else:
            ci = (mean, mean)

        file_to_metrics[key]["avg"] = mean
        file_to_metrics[key]["ci"] = ci
        file_to_metrics[key]["idx_to_entropy"] = idx_to_entropy  # <-- Essential for paired test

    return file_to_metrics


# --------------------------------------------------------
# Draw Subplot (Updated with shared x-labels and stars as ticks)
# --------------------------------------------------------
def draw_temperature_subplot(ax, file_to_metrics, dataset_name, model_name, temperatures, model_color, show_xlabel=True, show_temperature=True):
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.edgecolor": "gray",
        "axes.linewidth": 0.8,
    })

    keys = []
    for t in temperatures:
        keys += [f"{model_name}-Disable_temp{t}", f"{model_name}_temp{t}"]
    temp_palette = sns.color_palette("flare", len(temperatures))

    avg = [file_to_metrics[k]["avg"] for k in keys]
    ci  = [file_to_metrics[k]["ci"] for k in keys]
    yerr = [upper - mean for mean, (lower, upper) in zip(avg, ci)]

    diffs = []
    for j in range(len(temperatures)):
        non_r = avg[2*j]
        r     = avg[2*j + 1]
        diffs.append(non_r - r)

    group_gap = 0.9
    x = []
    for j in range(len(temperatures)):
        base = j * (2 + group_gap)
        x += [base + 0, base + 1]
    x = np.array(x, dtype=float)

    bar_width = 0.78

    bars = []
    for i, (mean, err, key) in enumerate(zip(avg, yerr, keys)):
        is_non_reasoning = ("-Disable_" in key)
        temp_idx = i // 2
        bar_color = temp_palette[temp_idx]
        hatch = None if is_non_reasoning else "//"

        b = ax.bar(
            x[i], mean, bar_width,
            yerr=err, capsize=4,
            color=bar_color,
            edgecolor="black",
            linewidth=1.1,
            hatch=hatch,
            zorder=3
        )
        bars.append(b[0])

    for i, bar in enumerate(bars):
        lower, upper = ci[i]
        center = bar.get_x() + bar.get_width() / 2
        ax.text(center, upper + 0.008, f"{upper:.3f}", ha="center", va="bottom",
                fontsize=7.5, color="black", fontweight="bold")
        ax.text(center, lower - 0.010, f"{lower:.3f}", ha="center", va="top",
                fontsize=7.5, color="black", fontweight="bold")

    group_positions = np.array([j * (2 + group_gap) + 0.5 for j in range(len(temperatures))], dtype=float)
    ax.set_xticks(group_positions)

    # ----------------------------------------------------
    # Calculate Significance Stars and set x-tick labels
    # ----------------------------------------------------
    xtick_labels = []
    for j, t in enumerate(temperatures):
        k_non = f"{model_name}-Disable_temp{t}"
        k_reason = f"{model_name}_temp{t}"

        pval = paired_entropy_test_one_sided(file_to_metrics, k_non, k_reason)
        star = p_to_stars(pval)

        # Format as requested
        if show_temperature:
            xtick_labels.append(f"{t}\n{star}")
        else:
            xtick_labels.append(f"{star}")

    if show_xlabel:
        ax.set_xticklabels(xtick_labels)
    else:
        # Hide labels for the upper rows, but keep ticks for alignment
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)

    # grid & spines
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.set_title(model_name, pad=8, weight="bold")
    ax.set_ylabel("")

    # --- Twin axis for ΔEntropy ---
    ax2 = ax.twinx()
    group_colors = temp_palette

    ax2.plot(group_positions, diffs, color="black", linestyle=":", linewidth=1.5, zorder=4, alpha=0.7)

    for pos, diff, c in zip(group_positions, diffs, group_colors):
        ax2.scatter(pos, diff, color=c, s=45, edgecolor="black", linewidth=0.6, zorder=5, label="_nolegend_")

    ax2.set_ylabel("")
    ax2.spines["right"].set_color("gray")
    ax2.spines["right"].set_linewidth(0.8)
    for spine in ["top", "left"]:
        ax2.spines[spine].set_visible(False)
    ax2.grid(False)

    delta_handle = mlines.Line2D(
        [], [], color="black", linestyle=":", marker="o",
        markerfacecolor="white", markeredgecolor="black",
        label="ΔEntropy"
    )
    return delta_handle


# --------------------------------------------------------
# Main Plotting Wrapper
# --------------------------------------------------------
def plot_dataset_temperature_all_models(dataset_name, model_names, temperatures, save_dir="figures"):
    model_palette = sns.color_palette("Set2", len(model_names))
    model_to_color = {m: model_palette[i] for i, m in enumerate(model_names)}

    model_to_metrics = {}
    for m in model_names:
        model_to_metrics[m] = compute_file_to_metrics_for_model(
            dataset_name=dataset_name,
            model_name=m,
            temperatures=temperatures
        )

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), dpi=1024)
    axes = axes.flatten()

    delta_handle = None
    for i, m in enumerate(model_names):
        ax = axes[i]

        # Only show the x-label (temp + stars) on the bottom row (indices 4 and 5)
        # show_x = True if i >= 4 else False
        show_x = True
        # show_temperature = True if i >= 4 else False
        show_temperature = True

        h = draw_temperature_subplot(
            ax=ax,
            file_to_metrics=model_to_metrics[m],
            dataset_name=dataset_name,
            model_name=m,
            temperatures=temperatures,
            model_color=model_to_color[m],
            show_xlabel=show_x,
            show_temperature=show_temperature
        )
        delta_handle = h

    fig.supxlabel("Temperature", fontsize=12, fontweight="bold", y=0.1)
    fig.supylabel("Entropy (Instability)", fontsize=12, fontweight="bold", x=0.06)

    fig.text(
        1.01, 0.5,
        "ΔEntropy",
        va="center",
        ha="center",
        rotation=-90,
        fontsize=12,
        fontweight="bold"
    )

    dataset_name_to_title = {
        "daily_dilemmas": r"Ethic ($\it{DailyDilemmas}$)",
        "medmcqa": r"Medicine ($\it{MedMCQA}$)",
        "mmlu-accounting": r"Finance ($\it{MMLU\!-\!Accounting}$)",
        "mmlu-law": r"Law ($\it{MMLU\!-\!Law}$)",
    }
    fig.suptitle(dataset_name_to_title.get(dataset_name, dataset_name), fontsize=14, fontweight="bold", y=0.98)

    style_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", label="Without Reasoning"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", label="With Reasoning"),
    ]

    handles = style_handles
    if delta_handle is not None:
        handles = handles + [delta_handle]

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
        fontsize=12,
        handlelength=1.6,
        columnspacing=1.6,
        handletextpad=0.5,
    )

    plt.tight_layout(rect=[0.05, 0.08, 1.0, 0.95])

    save_file = f"{save_dir}/temperature_{dataset_name}_all_models.png"
    plt.savefig(save_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_file}")


if __name__ == "__main__":
    import os
    if not os.path.exists("figures"):
        os.makedirs("figures")

    model_names = [
        "Qwen3-4B",
        "Qwen3-32B",
        "Qwen3-30B-A3B",
        "Seed-OSS-36B-Instruct",
        "NVIDIA-Nemotron-Nano-9B-v2",
        "NVIDIA-Nemotron-Nano-12B-v2",
    ]
    datasets = [
        "daily_dilemmas",
        "medmcqa",
        "mmlu-accounting",
        "mmlu-law",
    ]
    temperatures = ["0.3", "0.6", "0.9", "1.2"]

    for dataset_name in datasets:
        plot_dataset_temperature_all_models(
            dataset_name=dataset_name,
            model_names=model_names,
            temperatures=temperatures,
            save_dir="figures"
        )