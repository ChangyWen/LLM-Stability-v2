import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from scipy.stats import entropy
from scipy import stats
import seaborn as sns


# ----------------------------
# 1) Retained keys
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

                    # your special rule
                    if "temp0.0" in file_name:
                        if total_count < 1:
                            continue
                    else:
                        if total_count < 35:
                            continue
                    idx_set.add(idx)

            retained_ids_list.append(idx_set)
        return set.intersection(*retained_ids_list)


# ----------------------------
# 2) Compute metrics for a single (dataset, model, temp, mode)
# ----------------------------
def compute_entropy_list(file_name, retained_ids, dataset_name):
    ent_list = []

    if dataset_name == "daily_dilemmas":
        idx_to_results = {}
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

        for idx, results in idx_to_results.items():
            c = dict(Counter(results))
            dist = np.array([
                c.get("1", 0) / len(results),
                c.get("2", 0) / len(results),
                c.get("3", 0) / len(results),
            ])
            ent_list.append(entropy(dist))

    else:
        with open(file_name, "r") as f:
            for line in f:
                item = json.loads(line)
                idx = item["idx"]
                if idx not in retained_ids:
                    continue
                answer_counts = item["answer_counts"]
                if answer_counts is None:
                    continue
                total = sum(answer_counts.values())
                if total <= 0:
                    continue
                dist = np.array([v / total for v in answer_counts.values()])
                ent_list.append(entropy(dist))

    return np.array(ent_list, dtype=float)


def mean_ci_t(entropy_list):
    n = len(entropy_list)
    mean = float(np.mean(entropy_list)) if n else float("nan")
    if n >= 2:
        sem = float(np.std(entropy_list, ddof=1) / np.sqrt(n))
        ci = stats.t.interval(0.95, n - 1, loc=mean, scale=sem)
    elif n == 1:
        ci = (mean, mean)
    else:
        ci = (float("nan"), float("nan"))
    return mean, ci


# ----------------------------
# 3) Build all metrics for one dataset across models
# ----------------------------
def build_metrics_for_dataset(dataset_name, model_names, temps):
    subfix = "_counts" if dataset_name != "daily_dilemmas" else ""

    # metrics[model][temp]["non"/"reason"] = {"avg":..., "ci":...}
    metrics = {m: {} for m in model_names}

    for model in model_names:
        # (A) compute retained ids once per model across all 8 files (4 temps × 2 modes)
        all_files = []
        for t in temps:
            all_files += [
                f"outputs/{dataset_name}/processed_results/{model}_temp{t}_n50_dt{subfix}.jsonl",  # non-reasoning
                f"outputs/{dataset_name}/processed_results/{model}_temp{t}_n50{subfix}.jsonl",     # reasoning
            ]
        retained_ids = get_retained_keys(all_files, dataset_name)

        # (B) compute stats per file
        for t in temps:
            non_file = f"outputs/{dataset_name}/processed_results/{model}_temp{t}_n50_dt{subfix}.jsonl"
            rea_file = f"outputs/{dataset_name}/processed_results/{model}_temp{t}_n50{subfix}.jsonl"

            ent_non = compute_entropy_list(non_file, retained_ids, dataset_name)
            ent_rea = compute_entropy_list(rea_file, retained_ids, dataset_name)

            non_mean, non_ci = mean_ci_t(ent_non)
            rea_mean, rea_ci = mean_ci_t(ent_rea)

            metrics[model][t] = {
                "non": {"avg": non_mean, "ci": non_ci},
                "reason": {"avg": rea_mean, "ci": rea_ci},
                "retained_n": len(retained_ids),
            }

            print(f"[{dataset_name}] {model} temp{t} retained={len(retained_ids)}")
            print(f"  non:   {non_mean:.4f}  CI {non_ci[0]:.4f}-{non_ci[1]:.4f}")
            print(f"  reason:{rea_mean:.4f}  CI {rea_ci[0]:.4f}-{rea_ci[1]:.4f}")

    return metrics


# ----------------------------
# 4) Drawing: one subplot (one model)
# ----------------------------
def draw_temperature_bars_on_ax(
    ax,
    model_metrics_for_one_model,   # dict: temp -> {non:{avg,ci}, reason:{avg,ci}}
    model_title: str,
    temps,
    temp_to_color,
    group_gap=0.8,                 # <-- increase this to add more space between temperature groups
    bar_width=0.38,
):
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.edgecolor": "gray",
        "axes.linewidth": 0.8,
    })

    # positions with gaps: each temp group has 2 bars, and groups are separated by `group_gap`
    x_positions = []
    bar_specs = []  # (x, mean, err_up, color, hatch, ci_low, ci_up)
    cursor = 0.0

    for t in temps:
        # group center is cursor, two bars are left/right of center
        x_non = cursor - bar_width / 2
        x_rea = cursor + bar_width / 2

        non = model_metrics_for_one_model[t]["non"]
        rea = model_metrics_for_one_model[t]["reason"]

        non_mean = non["avg"]
        non_ci = non["ci"]
        non_err = non_ci[1] - non_mean

        rea_mean = rea["avg"]
        rea_ci = rea["ci"]
        rea_err = rea_ci[1] - rea_mean

        color = temp_to_color[t]

        bar_specs.append((x_non, non_mean, non_err, color, None, non_ci[0], non_ci[1]))     # non-reasoning: no hatch
        bar_specs.append((x_rea, rea_mean, rea_err, color, "//", rea_ci[0], rea_ci[1]))     # reasoning: hatch

        x_positions += [x_non, x_rea]
        cursor += 1.0 + group_gap

    # draw bars
    bars = []
    for x, mean, err, color, hatch, lo, up in bar_specs:
        b = ax.bar(
            x, mean, bar_width,
            yerr=err, capsize=5,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            alpha=1.0,
            hatch=hatch
        )
        bars.append((b[0], lo, up))

    # CI labels
    for bar, lo, up in bars:
        center = bar.get_x() + bar.get_width() / 2
        ax.text(center, up + 0.015, f"{up:.3f}",
                ha="center", va="bottom", fontsize=8, color="black", fontweight="bold")
        ax.text(center, lo - 0.025, f"{lo:.3f}",
                ha="center", va="top", fontsize=8, color="black", fontweight="bold")

    # match entropy_all style: no xticks/labels
    ax.set_xticks([])
    ax.set_xticklabels([])

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_title(model_title, pad=10, weight="bold")

    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


# ----------------------------
# 5) Drawing: one big figure per dataset (2×2 models)
# ----------------------------
def plot_dataset_temperature_2x2(
    dataset_name,
    metrics_for_dataset,   # metrics[model][temp]...
    model_names_2x2,
    temps,
    save_file,
    group_gap=0.8,         # increase to widen between temp groups
):
    # 4 temperature colors (one per temperature), like you used a "flare" feel earlier
    temp_palette = sns.color_palette("flare", len(temps) * 2)  # more resolution
    pick_idx = np.linspace(0, len(temp_palette) - 1, len(temps)).astype(int)
    temp_colors = [temp_palette[i] for i in pick_idx]
    temp_to_color = {t: temp_colors[i] for i, t in enumerate(temps)}

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=1024)
    axes = axes.flatten()

    dataset_title_map = {
        "daily_dilemmas": "Ethical Dilemmas (DailyDilemmas)",
        "medmcqa": "Medicine (MedMCQA)",
        "mmlu-accounting": "Finance (MMLU Accounting)",
        "mmlu-law": "Law (MMLU Law)",
    }

    for i, model in enumerate(model_names_2x2):
        ax = axes[i]
        draw_temperature_bars_on_ax(
            ax=ax,
            model_metrics_for_one_model=metrics_for_dataset[model],
            model_title=model,
            temps=temps,
            temp_to_color=temp_to_color,
            group_gap=group_gap,
            bar_width=0.38,
        )

    # one shared y-label (entropy_all style)
    fig.supylabel("Entropy", fontsize=12, fontweight="bold", x=0.045)

    # ---- Legend: temperatures (colored boxes) + 2 style boxes ----
    temp_handles = [
        mpatches.Patch(facecolor=temp_to_color[t], edgecolor="black", label=f"temp={t}")
        for t in temps
    ]
    style_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", label="Without Reasoning"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", label="With Reasoning"),
    ]

    handles = temp_handles + style_handles
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,                    # 4 temps + 2 styles -> 2 rows nicely with ncol=3
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),  # closer to figure
        fontsize=10,
        handlelength=1.6,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    # dataset title as suptitle (optional; consistent with your previous style)
    fig.suptitle(dataset_title_map.get(dataset_name, dataset_name), y=0.98, fontsize=14, fontweight="bold")

    # leave room for legend
    plt.tight_layout(rect=[0.05, 0.10, 1.0, 0.95])
    plt.savefig(save_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_file}")


# ----------------------------
# 6) Main
# ----------------------------
if __name__ == "__main__":
    datasets = [
        "daily_dilemmas",
        "medmcqa",
        # "mmlu-accounting",
        # "mmlu-law",
    ]

    # Choose exactly 4 models for 2x2
    model_names_2x2 = [
        "Qwen3-4B",
        "Qwen3-32B",
        "Qwen3-30B-A3B",
        "Seed-OSS-36B-Instruct",
    ]

    # temperatures you used
    temps = ["0.3", "0.6", "0.9", "1.2"]

    for dataset_name in datasets:
        metrics = build_metrics_for_dataset(dataset_name, model_names_2x2, temps)

        plot_dataset_temperature_2x2(
            dataset_name=dataset_name,
            metrics_for_dataset=metrics,
            model_names_2x2=model_names_2x2,
            temps=temps,
            save_file=f"figures/temperature_{dataset_name}_all_models.png",
            group_gap=0.9,   # <-- increase this to add more space between temperature groups
        )
