"""Generate report-ready plots from benchmark CSV files.

The script reads ``reports/summary.csv`` and ``reports/*_pairs.csv`` and writes
PNG figures that match the P2 speaker-verification tasks:

* baseline score distribution and ROC curve,
* EER/accuracy comparison against the baseline,
* task-specific plots for amplitude scaling, downsampling, noise, codecs,
  and reverberation.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

Path("/tmp/matplotlib-cache").mkdir(parents=True, exist_ok=True)
Path("/tmp/fontconfig-cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fontconfig-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


TASK_LABELS = {
    "task1_baseline": "Baseline",
    "task7_reverberation": "Reverberation",
}


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "figure.constrained_layout.use": True,
        }
    )


def _clean_summary(summary_path: Path) -> pd.DataFrame:
    summary = pd.read_csv(summary_path)
    summary = summary.dropna(subset=["eer", "accuracy"]).copy()
    summary = summary.drop_duplicates(subset=["task"], keep="last")
    summary["eer_pct"] = summary["eer"] * 100.0
    summary["accuracy_pct"] = summary["accuracy"] * 100.0
    return summary.sort_values("task").reset_index(drop=True)


def _task_label(task: str) -> str:
    if task in TASK_LABELS:
        return TASK_LABELS[task]
    label = task
    label = label.replace("task2_amplitude_", "Amplitude ")
    label = label.replace("task3_naive_step", "Naive x")
    label = label.replace("task3_interp_factor", "Interpolated x")
    label = label.replace("task4_gaussian_snr", "Gaussian ")
    label = label.replace("task5_env_snr", "Background ")
    label = label.replace("task6_", "")
    label = label.replace("dB", " dB")
    label = label.replace("kbps", " kbps")
    return label.replace("_", " ")


def _save(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{filename}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def _baseline(summary: pd.DataFrame) -> pd.Series:
    baseline = summary.loc[summary["task"] == "task1_baseline"]
    if baseline.empty:
        raise ValueError("Missing task1_baseline in summary.csv")
    return baseline.iloc[0]


def _plot_overall_metrics(summary: pd.DataFrame, output_dir: Path) -> None:
    data = summary.copy()
    data["label"] = data["task"].map(_task_label)
    data = data.sort_values(["eer", "task"], ascending=[False, True])

    fig, axes = plt.subplots(1, 2, figsize=(12, max(5, 0.28 * len(data))))

    axes[0].barh(data["label"], data["eer_pct"], color="#bd4f4f")
    axes[0].set_title("Equal Error Rate by experiment")
    axes[0].set_xlabel("EER [%]")
    axes[0].invert_yaxis()

    axes[1].barh(data["label"], data["accuracy_pct"], color="#3d7f72")
    axes[1].set_title("Accuracy by experiment")
    axes[1].set_xlabel("Accuracy [%]")
    axes[1].set_xlim(max(0, data["accuracy_pct"].min() - 5), 100)
    axes[1].invert_yaxis()

    _save(fig, output_dir, "overall_eer_accuracy")


def _plot_delta_vs_baseline(summary: pd.DataFrame, output_dir: Path) -> None:
    base = _baseline(summary)
    data = summary.loc[summary["task"] != "task1_baseline"].copy()
    data["delta_eer_pp"] = (data["eer"] - base["eer"]) * 100.0
    data["delta_acc_pp"] = (data["accuracy"] - base["accuracy"]) * 100.0
    data["label"] = data["task"].map(_task_label)
    data = data.sort_values("delta_eer_pp", ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.26 * len(data))))
    colors = np.where(data["delta_eer_pp"] >= 0, "#bd4f4f", "#3d7f72")
    ax.barh(data["label"], data["delta_eer_pp"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title("Change in EER relative to baseline")
    ax.set_xlabel("Delta EER [percentage points]")
    ax.invert_yaxis()

    _save(fig, output_dir, "delta_eer_vs_baseline")


def _plot_task2_amplitude(summary: pd.DataFrame, output_dir: Path) -> None:
    data = summary[summary["task"].str.startswith("task2_amplitude_")].copy()
    if data.empty:
        return
    data["amplitude"] = data["task"].str.extract(r"amplitude_([0-9.]+)").astype(float)
    data = data.sort_values("amplitude")
    base = _baseline(summary)

    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax2 = ax1.twinx()
    ax1.plot(data["amplitude"], data["eer_pct"], marker="o", color="#bd4f4f", label="EER")
    ax2.plot(
        data["amplitude"],
        data["accuracy_pct"],
        marker="s",
        color="#3d7f72",
        label="Accuracy",
    )
    ax1.axhline(base["eer_pct"], color="#bd4f4f", linestyle="--", linewidth=1, alpha=0.55)
    ax2.axhline(
        base["accuracy_pct"], color="#3d7f72", linestyle="--", linewidth=1, alpha=0.55
    )
    ax1.set_xscale("log")
    ax1.set_title("Task 2: amplitude scaling")
    ax1.set_xlabel("Amplitude multiplier")
    ax1.set_ylabel("EER [%]")
    ax2.set_ylabel("Accuracy [%]")
    ax2.set_ylim(max(0, data["accuracy_pct"].min() - 5), 100)
    lines = ax1.get_lines()[:1] + ax2.get_lines()[:1]
    ax1.legend(lines, [line.get_label() for line in lines], loc="center right")

    _save(fig, output_dir, "task2_amplitude")


def _plot_task3_downsampling(summary: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for _, row in summary.iterrows():
        match = re.match(r"task3_(naive_step|interp_factor)(\d+)", row["task"])
        if match:
            rows.append(
                {
                    "method": "Naive" if match.group(1) == "naive_step" else "Interpolated",
                    "factor": int(match.group(2)),
                    "eer_pct": row["eer_pct"],
                    "accuracy_pct": row["accuracy_pct"],
                }
            )
    data = pd.DataFrame(rows)
    if data.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    for method, group in data.groupby("method"):
        group = group.sort_values("factor")
        axes[0].plot(group["factor"], group["eer_pct"], marker="o", label=method)
        axes[1].plot(group["factor"], group["accuracy_pct"], marker="o", label=method)
    axes[0].set_title("Task 3: EER")
    axes[0].set_ylabel("EER [%]")
    axes[1].set_title("Task 3: accuracy")
    axes[1].set_ylabel("Accuracy [%]")
    axes[1].set_ylim(max(0, data["accuracy_pct"].min() - 5), 100)
    for ax in axes:
        ax.set_xlabel("Downsampling factor")
        ax.set_xticks(sorted(data["factor"].unique()))
        ax.legend()

    _save(fig, output_dir, "task3_downsampling")


def _plot_snr_task(
    summary: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    title: str,
    filename: str,
) -> None:
    data = summary[summary["task"].str.startswith(prefix)].copy()
    if data.empty:
        return
    data["snr"] = data["task"].str.extract(r"snr([0-9]+)dB").astype(int)
    data = data.sort_values("snr", ascending=False)
    base = _baseline(summary)

    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax2 = ax1.twinx()
    ax1.plot(data["snr"], data["eer_pct"], marker="o", color="#bd4f4f", label="EER")
    ax2.plot(
        data["snr"],
        data["accuracy_pct"],
        marker="s",
        color="#3d7f72",
        label="Accuracy",
    )
    ax1.axhline(base["eer_pct"], color="#bd4f4f", linestyle="--", linewidth=1, alpha=0.55)
    ax2.axhline(
        base["accuracy_pct"], color="#3d7f72", linestyle="--", linewidth=1, alpha=0.55
    )
    ax1.set_title(title)
    ax1.set_xlabel("SNR [dB]")
    ax1.set_ylabel("EER [%]")
    ax2.set_ylabel("Accuracy [%]")
    ax2.set_ylim(max(0, data["accuracy_pct"].min() - 5), 100)
    ax1.invert_xaxis()
    lines = ax1.get_lines()[:1] + ax2.get_lines()[:1]
    ax1.legend(lines, [line.get_label() for line in lines], loc="center left")

    _save(fig, output_dir, filename)


def _plot_task6_codecs(summary: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for _, row in summary.iterrows():
        match = re.match(r"task6_([a-z0-9]+)_([0-9]+)kbps", row["task"])
        if match:
            rows.append(
                {
                    "codec": match.group(1).upper(),
                    "bitrate": int(match.group(2)),
                    "eer_pct": row["eer_pct"],
                    "accuracy_pct": row["accuracy_pct"],
                }
            )
    data = pd.DataFrame(rows)
    if data.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=False)
    for codec, group in data.groupby("codec"):
        group = group.sort_values("bitrate")
        axes[0].plot(group["bitrate"], group["eer_pct"], marker="o", label=codec)
        axes[1].plot(group["bitrate"], group["accuracy_pct"], marker="o", label=codec)
    axes[0].set_title("Task 6: codec compression EER")
    axes[0].set_ylabel("EER [%]")
    axes[1].set_title("Task 6: codec compression accuracy")
    axes[1].set_ylabel("Accuracy [%]")
    axes[1].set_ylim(max(0, data["accuracy_pct"].min() - 5), 100)
    for ax in axes:
        ax.set_xlabel("Bitrate [kbps]")
        ax.set_xscale("log")
        ax.legend()

    _save(fig, output_dir, "task6_codecs")


def _plot_task7_reverberation(summary: pd.DataFrame, output_dir: Path) -> None:
    data = summary[summary["task"].isin(["task1_baseline", "task7_reverberation"])].copy()
    if len(data) < 2:
        return
    data["label"] = data["task"].map(_task_label)

    fig, axes = plt.subplots(1, 2, figsize=(7, 4.2))
    axes[0].bar(data["label"], data["eer_pct"], color=["#607d8b", "#bd4f4f"])
    axes[0].set_title("Task 7: EER")
    axes[0].set_ylabel("EER [%]")
    axes[1].bar(data["label"], data["accuracy_pct"], color=["#607d8b", "#3d7f72"])
    axes[1].set_title("Task 7: accuracy")
    axes[1].set_ylabel("Accuracy [%]")
    axes[1].set_ylim(max(0, data["accuracy_pct"].min() - 5), 100)

    _save(fig, output_dir, "task7_reverberation")


def _read_pairs(reports_dir: Path, task: str) -> pd.DataFrame | None:
    path = reports_dir / f"{task}_pairs.csv"
    if not path.exists():
        return None
    pairs = pd.read_csv(path)
    if not {"score", "label"}.issubset(pairs.columns):
        return None
    return pairs.dropna(subset=["score", "label"]).copy()


def _plot_score_distribution(reports_dir: Path, output_dir: Path) -> None:
    pairs = _read_pairs(reports_dir, "task1_baseline")
    if pairs is None or pairs.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4.2))
    impostor = pairs.loc[pairs["label"] == 0, "score"]
    genuine = pairs.loc[pairs["label"] == 1, "score"]
    bins = np.linspace(pairs["score"].min(), pairs["score"].max(), 28)
    ax.hist(impostor, bins=bins, alpha=0.72, density=True, label="Impostor", color="#bd4f4f")
    ax.hist(genuine, bins=bins, alpha=0.72, density=True, label="Genuine", color="#3d7f72")
    ax.set_title("Baseline score distribution")
    ax.set_xlabel("Cosine similarity score")
    ax.set_ylabel("Density")
    ax.legend()

    _save(fig, output_dir, "baseline_score_distribution")


def _plot_roc_curves(summary: pd.DataFrame, reports_dir: Path, output_dir: Path) -> None:
    tasks = [
        "task1_baseline",
        "task3_naive_step10",
        "task3_interp_factor10",
        "task4_gaussian_snr10dB",
        "task5_env_snr0dB",
        "task7_reverberation",
    ]
    existing_tasks = [task for task in tasks if task in set(summary["task"])]
    if not existing_tasks:
        return

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for task in existing_tasks:
        pairs = _read_pairs(reports_dir, task)
        if pairs is None or pairs["label"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(pairs["label"].astype(int), pairs["score"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=1.8, label=f"{_task_label(task)} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    ax.set_title("ROC curves for selected experiments")
    ax.set_xlabel("False Acceptance Rate")
    ax.set_ylabel("True Acceptance Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8)

    _save(fig, output_dir, "selected_roc_curves")


def generate_plots(reports_dir: Path, output_dir: Path) -> None:
    summary = _clean_summary(reports_dir / "summary.csv")
    _plot_overall_metrics(summary, output_dir)
    _plot_delta_vs_baseline(summary, output_dir)
    _plot_task2_amplitude(summary, output_dir)
    _plot_task3_downsampling(summary, output_dir)
    _plot_snr_task(
        summary,
        output_dir,
        "task4_gaussian_",
        "Task 4: Gaussian noise",
        "task4_gaussian_noise",
    )
    _plot_snr_task(
        summary,
        output_dir,
        "task5_env_",
        "Task 5: background noise",
        "task5_background_noise",
    )
    _plot_task6_codecs(summary, output_dir)
    _plot_task7_reverberation(summary, output_dir)
    _plot_score_distribution(reports_dir, output_dir)
    _plot_roc_curves(summary, reports_dir, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory with summary.csv and *_pairs.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/figures"),
        help="Directory where PNG/PDF plots will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    _set_style()
    args = parse_args()
    generate_plots(args.reports_dir, args.output_dir)
    print(f"Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
