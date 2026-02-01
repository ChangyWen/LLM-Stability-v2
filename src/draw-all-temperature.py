import json
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from scipy.stats import entropy
from scipy import stats
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


# ----------------------------
# retained keys (same as yours)
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

                    # your special rule for temp0.0
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
# compute metrics for ONE (dataset, model) across temps
# returns: dict with keys like "<model>-Disable_temp0.3" etc
# --------------------------------------------------------
def compute_file_to_metrics_for_model(dataset_name, model_name, temperatures, base_dir="outputs"):
    subfix = "_counts" if dataset_name != "daily_dilemmas" else ""

    # build all files (dt = non-reasoning in your naming)
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
        # detect non-reasoning vs reasoning
        is_non_reasoning = ("_dt" in file_name)

        # parse temperature from filename: ".../<model>_temp0.6_n50..."
        # safer than split("_")[1] if model contains underscores
        fname = file_name.split("/")[-1]
        # find substring "tempX"
        temp_part = [p for p in fname.split("_") if p.startswith("temp")][0]  # e.g., "temp0.6"
        temp_str = temp_part.replace("temp", "")                              # e.g., "0.6"

        key = f"{model_name}-Disable_temp{temp_str}" if is_non_reasoning else f"{model_name}_temp{temp_str}"
        file_to_metrics[key] = {}

        retained = retained_ids_list[i]
        entropy_list = []

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
                entropy_list.append(entropy(dist))

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
                    entropy_list.append(entropy(dist))

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

    return file_to_metrics


# --------------------------------------------------------
# draw ONE model subplot (bars + twin axis ΔEntropy)
# --------------------------------------------------------
def draw_temperature_subplot(ax, file_to_metrics, dataset_name, model_name, temperatures, model_color):
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.edgecolor": "gray",
        "axes.linewidth": 0.8,
    })

    # order: for each temp: Non-R then R
    keys = []
    for t in temperatures:
        keys += [f"{model_name}-Disable_temp{t}", f"{model_name}_temp{t}"]
    temp_palette = sns.color_palette("flare", len(temperatures))

    avg = [file_to_metrics[k]["avg"] for k in keys]
    ci  = [file_to_metrics[k]["ci"] for k in keys]
    yerr = [upper - mean for mean, (lower, upper) in zip(avg, ci)]

    # ΔEntropy per temp (Non-R − R)
    diffs = []
    for j in range(len(temperatures)):
        non_r = avg[2*j]
        r     = avg[2*j + 1]
        diffs.append(non_r - r)

    # x positions with extra gap between temperatures (group spacing)
    group_gap = 0.9   # increase this to add more space between temperature groups
    x = []
    for j in range(len(temperatures)):
        base = j * (2 + group_gap)
        x += [base + 0, base + 1]
    x = np.array(x, dtype=float)

    bar_width = 0.78

    # Bars: non-reasoning = white face; reasoning = hatched
    bars = []
    for i, (mean, err, key) in enumerate(zip(avg, yerr, keys)):
        is_non_reasoning = ("-Disable_" in key)

        # temperature index: every two bars share one temperature
        temp_idx = i // 2
        bar_color = temp_palette[temp_idx]

        hatch = None if is_non_reasoning else "//"

        b = ax.bar(
            x[i], mean, bar_width,
            yerr=err, capsize=4,
            color=bar_color,          # 🔴 solid temperature color
            edgecolor="black",        # clean contrast
            linewidth=1.1,
            hatch=hatch,
            zorder=3
        )
        bars.append(b[0])

    # CI labels
    for i, bar in enumerate(bars):
        lower, upper = ci[i]
        center = bar.get_x() + bar.get_width() / 2
        ax.text(center, upper + 0.008, f"{upper:.3f}", ha="center", va="bottom",
                fontsize=7.5, color="black", fontweight="bold")
        ax.text(center, lower - 0.010, f"{lower:.3f}", ha="center", va="top",
                fontsize=7.5, color="black", fontweight="bold")

    # x ticks: show only temperatures at group centers
    group_positions = np.array([j * (2 + group_gap) + 0.5 for j in range(len(temperatures))], dtype=float)
    ax.set_xticks(group_positions)
    ax.set_xticklabels([str(t) for t in temperatures])

    # grid & spines
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.set_title(model_name, pad=8, weight="bold")
    ax.set_ylabel("")  # shared at figure level

    # --- Twin axis for ΔEntropy ---
    ax2 = ax.twinx()

    # temperature colors for Δ markers (like your origin: flare palette)
    # temp_palette = sns.color_palette("flare", len(temperatures))
    group_colors = temp_palette

    ax2.plot(group_positions, diffs, color="black", linestyle=":", linewidth=1.5, zorder=4, alpha=0.7)

    for pos, diff, c in zip(group_positions, diffs, group_colors):
        ax2.scatter(pos, diff, color=c, s=45, edgecolor="black", linewidth=0.6, zorder=5, label="_nolegend_")

    # style ax2
    ax2.set_ylabel("")  # shared via legend; keep clean
    ax2.spines["right"].set_color("gray")
    ax2.spines["right"].set_linewidth(0.8)
    for spine in ["top", "left"]:
        ax2.spines[spine].set_visible(False)
    ax2.grid(False)

    # return an invisible handle for Δ legend entry (so we only add once per figure)
    delta_handle = mlines.Line2D(
        [], [], color="black", linestyle=":", marker="o",
        markerfacecolor="white", markeredgecolor="black",
        label="ΔEntropy"
    )
    return delta_handle


# --------------------------------------------------------
# big figure per dataset: 2x2 subplots (one per model)
# --------------------------------------------------------
def plot_dataset_temperature_all_models(dataset_name, model_names, temperatures, save_dir="figures"):
    # style palette for models (Set2 like entropy_all.png)
    model_palette = sns.color_palette("Set2", len(model_names))
    model_to_color = {m: model_palette[i] for i, m in enumerate(model_names)}

    # compute metrics for each model
    model_to_metrics = {}
    for m in model_names:
        model_to_metrics[m] = compute_file_to_metrics_for_model(
            dataset_name=dataset_name,
            model_name=m,
            temperatures=temperatures
        )

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=1024)
    axes = axes.flatten()

    delta_handle = None
    for i, m in enumerate(model_names):
        ax = axes[i]
        h = draw_temperature_subplot(
            ax=ax,
            file_to_metrics=model_to_metrics[m],
            dataset_name=dataset_name,
            model_name=m,
            temperatures=temperatures,
            model_color=model_to_color[m]
        )
        delta_handle = h  # same style for all

    # shared left y label (match entropy_all.png style)
    fig.supxlabel("Temperature", fontsize=12, fontweight="bold", y=0.1)
    fig.supylabel("Output Entropy (Decision-making Stability)", fontsize=12, fontweight="bold", x=0.06)
    # shared right y label for twin axis (ΔEntropy)
    fig.text(
        1.01, 0.5,                      # x, y in figure coordinates
        "ΔEntropy",
        va="center",
        ha="center",
        rotation=-90,
        fontsize=12,
        fontweight="bold"
    )

    # dataset title (optional; remove if you don’t want)
    dataset_name_to_title = {
        "daily_dilemmas": r"Ethic ($\it{DailyDilemmas}$)",
        "medmcqa": r"Medicine ($\it{MedMCQA}$)",
        "mmlu-accounting": r"Finance ($\it{MMLU\!-\!Accounting}$)",
        "mmlu-law": r"Law ($\it{MMLU\!-\!Law}$)",
    }
    fig.suptitle(dataset_name_to_title.get(dataset_name, dataset_name), fontsize=14, fontweight="bold", y=0.98)

    # 2 style handles: non-reasoning empty, reasoning hatched (no color)
    style_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", label="Without Reasoning"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", label="With Reasoning"),
    ]

    handles = style_handles
    if delta_handle is not None:
        handles = handles + [delta_handle]

    # ✅ Legend ABOVE subplots and BELOW title
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),  # between title (0.98) and axes
        fontsize=12,
        handlelength=1.6,
        columnspacing=1.6,
        handletextpad=0.5,
    )

    # layout: leave less room at bottom so legend sits closer
    plt.tight_layout(rect=[0.05, 0.08, 1.0, 0.95])

    save_file = f"{save_dir}/temperature_{dataset_name}_all_models.png"
    plt.savefig(save_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_file}")


if __name__ == "__main__":
    model_names = [
        "Qwen3-4B",
        "Qwen3-32B",
        "Qwen3-30B-A3B",
        "Seed-OSS-36B-Instruct",
    ]
    datasets = [
        "daily_dilemmas",
        "medmcqa",
        # add more if you have them:
        # "mmlu-accounting",
        # "mmlu-law",
    ]
    temperatures = ["0.3", "0.6", "0.9", "1.2"]

    for dataset_name in datasets:
        plot_dataset_temperature_all_models(
            dataset_name=dataset_name,
            model_names=model_names,
            temperatures=temperatures,
            save_dir="figures"
        )
