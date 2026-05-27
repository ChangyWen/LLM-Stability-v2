import json
import math
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from collections import Counter
import numpy as np
from scipy.stats import entropy
from scipy import stats
import seaborn as sns
from collections import defaultdict


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


def p_to_stars(p):
    """
    Convert p-value to significance stars.
    Returns 'N.S.' if not significant to clarify the bracket.
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
        return "N.S."


def add_sig_bracket(ax, x1, x2, y, h, text):
    """
    Draws a significance bracket between x1 and x2 at height y.
    """
    fontsize = 12 if text != "N.S." else 10
    fontweight = "bold" if text != "N.S." else "normal"

    # Bracket line
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="black", clip_on=False)

    # Stars or n.s. text
    ax.text((x1 + x2) / 2, y + h + (h * 0.2), text,
            ha="center", va="bottom", fontsize=fontsize, fontweight=fontweight,
            color="black", clip_on=False)


def paired_entropy_test_one_sided(file_to_metrics, key_baseline, key_compare):
    """
    One-sided paired Wilcoxon test:
    H1: Entropy_{baseline} > Entropy_{compare}
    Returns p-value (float).
    """
    a = file_to_metrics[key_baseline]["idx_to_entropy"]
    b = file_to_metrics[key_compare]["idx_to_entropy"]

    common = sorted(set(a.keys()) & set(b.keys()))
    if len(common) == 0:
        return float("nan")

    x_base = np.array([a[i] for i in common], dtype=float)
    x_comp = np.array([b[i] for i in common], dtype=float)

    diff = x_base - x_comp
    if np.allclose(diff, 0):
        return 1.0

    res = stats.wilcoxon(x_base, x_comp, alternative="greater", zero_method="wilcox")
    return float(res.pvalue)


def plot_combined_statistics(all_results):
    save_file = "figures/entropy-masks_combined.png"
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.edgecolor": "gray",
        "axes.linewidth": 0.8,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=1024, sharey=True)
    # point_color = sns.color_palette("Set2", 1)[0]

    # --- Pre-calculate Global Y Bounds for consistent bracket scaling ---
    all_upper_bounds = []
    all_lower_bounds = []
    for res, _, _ in all_results:
        for k in res:
            all_lower_bounds.append(res[k]["ci"][0])
            all_upper_bounds.append(res[k]["ci"][1])

    global_min_y = min(all_lower_bounds)
    global_max_y = max(all_upper_bounds)
    y_range = global_max_y - global_min_y

    # Dynamically scale bracket height and spacing
    y_start = global_max_y + (y_range * 0.1)
    y_step = y_range * 0.12
    h = y_range * 0.025

    for ax, (file_to_metrics, dataset, model) in zip(axes, all_results):
        keys = [
            f"{model}-Disable",
            f"{model} (1)",
            f"{model} (1-2)",
            f"{model} (1-3)",
            f"{model} (1-4)",
        ]
        x_labels = ["None", "Step 1", "Steps 1-2", "Steps 1-3", "Steps 1-4"]

        avg = [file_to_metrics[k]["avg"] for k in keys]
        ci = [file_to_metrics[k]["ci"] for k in keys]
        lower_bound = [lower for lower, upper in ci]
        upper_bound = [upper for lower, upper in ci]
        x = np.arange(len(keys))

        point_color = model_to_color[model]

        # ---- 1. Line connecting means (subtle) ----
        ax.plot(x, avg, linestyle="-", linewidth=2.2, color=point_color, alpha=0.85, zorder=1)

        # ---- 2. 95% CI Shaded Area ----
        ax.fill_between(x, lower_bound, upper_bound, color=point_color, alpha=0.2, zorder=0, linewidth=0)

        # ---- 3. Mean Markers (on top) ----
        ax.plot(x, avg, marker="o", markersize=10, color=point_color, markeredgecolor=point_color,
                markeredgewidth=1.5, linestyle="None", alpha=1.0, zorder=4)

        # ---- Significance Brackets ----
        margin = 0.06

        # Adjacent pairs
        p_01 = paired_entropy_test_one_sided(file_to_metrics, keys[0], keys[1])
        add_sig_bracket(ax, 0 + margin, 1 - margin, y_start, h, p_to_stars(p_01))

        p_12 = paired_entropy_test_one_sided(file_to_metrics, keys[1], keys[2])
        add_sig_bracket(ax, 1 + margin, 2 - margin, y_start, h, p_to_stars(p_12))

        p_23 = paired_entropy_test_one_sided(file_to_metrics, keys[2], keys[3])
        add_sig_bracket(ax, 2 + margin, 3 - margin, y_start, h, p_to_stars(p_23))

        p_34 = paired_entropy_test_one_sided(file_to_metrics, keys[3], keys[4])
        add_sig_bracket(ax, 3 + margin, 4 - margin, y_start, h, p_to_stars(p_34))

        # Baseline vs remaining
        for i in range(2, 5):
            p_val = paired_entropy_test_one_sided(file_to_metrics, keys[0], keys[i])
            stars = p_to_stars(p_val)
            stack_level = y_start + (i - 1) * y_step
            add_sig_bracket(ax, 0, i, stack_level, h, stars)

        # ---- Plot formatting ----
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=0, ha="center")
        dataset_formatted = "MedMCQA" if dataset == "medmcqa" else "MMLU-Accounting"
        ax.set_title(f"{dataset_formatted} ({model})", pad=20, weight="bold")
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        max_bracket_height = y_start + 3 * y_step + (h * 3)
        ax.set_ylim(global_min_y - (y_range * 0.1), max_bracket_height)

    # ---- Sup Labels and Legend ----
    fig.supylabel("Entropy (Instability)", fontsize=13, fontweight="bold", x=0.02)
    fig.supxlabel("Reasoning Step(s)", fontsize=13, fontweight="bold", y=0.08)

    # Custom Legend to explain the visual elements
    mean_line = mlines.Line2D([], [], color=point_color, marker='o', linestyle='--',
                              markersize=9, markeredgecolor='black', label='Mean')
    ci_patch = mpatches.Patch(color=point_color, alpha=0.2, label='95% CI (Shaded)')
    iqr_patch = mpatches.Patch(color=point_color, alpha=0.3, label='IQR (25th - 75th)')
    median_marker = mlines.Line2D([], [], color='black', marker='_', linestyle='None',
                                  markersize=12, markeredgewidth=2.5, label='Median')

    # fig.legend(handles=[mean_line, ci_patch, iqr_patch, median_marker],
    #            loc="lower center", ncol=4, bbox_to_anchor=(0.53, -0.02),
    #            frameon=False, fontsize=11, columnspacing=1.5)

    # Adjust layout to make room for the legend at the bottom
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(save_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_file}")


def get_statistics(result_files, retained_ids_list, dataset, model):
    file_to_metrics = {}

    for i, file_name in enumerate(result_files):
        if model in file_name and ("masks_completion" in file_name):
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
        idx_to_entropy = {}
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

                ent_val = entropy(np.array(distribution))
                entropy_list.append(ent_val)
                idx_to_entropy[idx] = ent_val

        entropy_list = np.array(entropy_list, dtype=float)
        mean = float(np.mean(entropy_list))
        n = int(len(entropy_list))
        sem = float(np.std(entropy_list, ddof=1) / np.sqrt(n)) if n >= 2 else 0.0
        ci = stats.t.interval(0.95, n - 1, loc=mean, scale=sem) if n >= 2 else (mean, mean)

        file_to_metrics[key]["avg"] = mean
        file_to_metrics[key]["ci"] = ci
        file_to_metrics[key]["idx_to_entropy"] = idx_to_entropy

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
        idx_to_entropy = {}
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

                ent_val = entropy(np.array(distribution))
                entropy_list.append(ent_val)
                idx_to_entropy[idx] = ent_val

        entropy_list = np.array(entropy_list, dtype=float)
        mean = float(np.mean(entropy_list))
        n = int(len(entropy_list))
        sem = float(np.std(entropy_list, ddof=1) / np.sqrt(n)) if n >= 2 else 0.0
        ci = stats.t.interval(0.95, n - 1, loc=mean, scale=sem) if n >= 2 else (mean, mean)

        file_to_metrics[key] = {
            "avg": mean,
            "ci": ci,
            "idx_to_entropy": idx_to_entropy
        }

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
        all_results.append((res, dataset, model))

    plot_combined_statistics(all_results)