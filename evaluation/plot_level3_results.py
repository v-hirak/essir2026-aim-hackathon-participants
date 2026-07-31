#!/usr/bin/env python3
"""Plot the Level-3 method progression and measured results."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = ROOT / "results" / "evaluation"

RUNS = [
    ("baseline-level3.json", "Page baseline", "Baseline"),
    ("chunking-level3.json", "Fixed chunks", "Chunking"),
    ("twohop-level3.json", "Fixed chunks + 2-hop", "Chunking + multi-hop"),
    ("original-chunking-twohop-level3.json", "Page + 2-hop", "Multi-hop"),
    ("aspect-multiquery-level3.json", "Aspect multi-query", "Query decomposition"),
    ("hybrid-original-aspects-level3.json", "Original + aspects", "Hybrid queries"),
    (
        "hierarchical-page-paragraph-level3.json",
        "Page → paragraph",
        "Hierarchical",
    ),
    ("semantic-single-level3.json", "Semantic chunks", "Semantic chunking"),
    (
        "semantic-twohop-level3.json",
        "Semantic + 2-hop",
        "Final method",
    ),
    ("page-to-semantic-level3.json", "Page → semantic", "Hierarchical semantic"),
]


def load_runs(results_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for filename, label, idea in RUNS:
        path = results_dir / filename
        if not path.is_file():
            missing.append(filename)
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
        rows.append(
            {
                "filename": filename,
                "label": label,
                "idea": idea,
                "page_recall": summary["mean_gold_page_recall"],
                "bertscore": summary["mean_bertscore_f1"],
                "latency_s": summary["mean_latency_ms"] / 1000,
            }
        )
    if missing:
        print("Skipped missing runs: " + ", ".join(missing))
    if not rows:
        raise FileNotFoundError(f"no evaluation runs found in {results_dir}")
    return rows


def method_color(label: str) -> str:
    if label == "Semantic + 2-hop":
        return "#2a9d8f"
    if label == "Page baseline":
        return "#6c757d"
    return "#8fb9d9"


def plot_metrics(rows: list[dict[str, object]], output: Path) -> None:
    labels = [str(row["label"]) for row in rows]
    positions = list(range(len(rows)))
    colors = [method_color(label) for label in labels]
    baseline = rows[0]

    fig, axes = plt.subplots(1, 3, figsize=(16, 8), sharey=True)
    panels = [
        ("page_recall", "Gold-page recall", 0, 0.70, 3),
        ("bertscore", "BERTScore F1", 0.80, 0.90, 3),
        ("latency_s", "Mean latency (seconds)", 0, 40, 1),
    ]

    for axis, (key, title, lower, upper, decimals) in zip(axes, panels):
        values = [float(row[key]) for row in rows]
        bars = axis.barh(positions, values, color=colors, height=0.68)
        axis.set_title(title, fontweight="bold")
        axis.set_xlim(lower, upper)
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)
        axis.axvline(
            float(baseline[key]),
            color="#495057",
            linewidth=1.2,
            linestyle="--",
            alpha=0.75,
        )
        for bar, value in zip(bars, values):
            axis.text(
                value + (upper - lower) * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.{decimals}f}",
                va="center",
                fontsize=9,
            )
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)

    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    for tick, label in zip(axes[0].get_yticklabels(), labels):
        if label == "Semantic + 2-hop":
            tick.set_fontweight("bold")

    fig.suptitle(
        "Level 3 retrieval experiments — Q7 to Q9",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.02,
        "Dashed line = page-level baseline. Green = selected final method.",
        ha="center",
        color="#495057",
    )
    fig.subplots_adjust(left=0.24, right=0.97, top=0.90, bottom=0.09, wspace=0.25)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def add_box(axis, x: float, y: float, title: str, detail: str, kind: str = "normal"):
    colors = {
        "baseline": ("#e9ecef", "#343a40"),
        "normal": ("#d9eaf7", "#1f2933"),
        "negative": ("#f8d7da", "#54232a"),
        "positive": ("#d1e7dd", "#18392f"),
        "final": ("#2a9d8f", "white"),
    }
    face, text_color = colors[kind]
    content = f"{title}\n{detail}"
    axis.text(
        x,
        y,
        content,
        ha="center",
        va="center",
        fontsize=8.5,
        color=text_color,
        fontweight="bold" if kind in {"baseline", "final"} else "normal",
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": face,
            "edgecolor": "#6c757d",
            "linewidth": 1,
        },
    )


def arrow(axis, start: tuple[float, float], end: tuple[float, float]):
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "->", "color": "#6c757d", "linewidth": 1.4},
    )


def plot_progression(output: Path) -> None:
    fig, axis = plt.subplots(figsize=(17, 8.5))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(0.02, 0.82, "Chunk boundaries", fontweight="bold", color="#495057")
    axis.text(0.02, 0.60, "Query strategy", fontweight="bold", color="#495057")
    axis.text(0.02, 0.20, "Hierarchy", fontweight="bold", color="#495057")

    add_box(axis, 0.13, 0.51, "Page baseline", "R=.439 · B=.867 · 25.5s", "baseline")

    add_box(axis, 0.30, 0.82, "Fixed chunks", "R=.328\nduplicate-page crowding", "negative")
    add_box(axis, 0.48, 0.82, "Fixed + 2-hop", "R=.328\nno coverage gain", "negative")
    add_box(axis, 0.67, 0.82, "Semantic chunks", "R=.567 · 18.7s\ncoherent boundaries", "positive")
    add_box(axis, 0.88, 0.82, "Semantic + 2-hop", "R=.633 · B=.865 · 26.6s", "final")

    add_box(axis, 0.31, 0.51, "Page + 2-hop", "R=.439\nearly stopping", "negative")
    add_box(axis, 0.51, 0.51, "Aspect queries", "R=.394\nQ8 improved; unstable", "normal")
    add_box(axis, 0.71, 0.51, "Original + aspects", "R=.439\npreserved baseline", "normal")

    add_box(axis, 0.34, 0.20, "Page → paragraph", "R=.439 · B=.870 · 19.4s", "positive")
    add_box(axis, 0.59, 0.20, "Page → semantic", "R=.439\nfirst-stage recall ceiling", "normal")

    arrow(axis, (0.18, 0.55), (0.26, 0.78))
    arrow(axis, (0.35, 0.82), (0.43, 0.82))
    arrow(axis, (0.53, 0.82), (0.61, 0.82))
    arrow(axis, (0.73, 0.82), (0.82, 0.82))

    arrow(axis, (0.19, 0.51), (0.25, 0.51))
    arrow(axis, (0.37, 0.51), (0.44, 0.51))
    arrow(axis, (0.58, 0.51), (0.64, 0.51))

    arrow(axis, (0.18, 0.47), (0.28, 0.24))
    arrow(axis, (0.42, 0.20), (0.52, 0.20))

    axis.text(
        0.5,
        0.96,
        "Level 3 method progression and measured diagnosis",
        ha="center",
        fontsize=16,
        fontweight="bold",
    )
    axis.text(
        0.5,
        0.06,
        textwrap.fill(
            "Main finding: semantic boundaries improved evidence coverage; a bounded "
            "second hop recovered an additional missing page while keeping answer "
            "similarity close to the baseline.",
            width=125,
        ),
        ha="center",
        fontsize=10,
        color="#495057",
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir or results_dir / "figures").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_runs(results_dir)
    metrics_file = output_dir / "level3-metrics.png"
    progression_file = output_dir / "level3-method-progression.png"
    plot_metrics(rows, metrics_file)
    plot_progression(progression_file)
    print(f"Saved: {metrics_file}")
    print(f"Saved: {progression_file}")


if __name__ == "__main__":
    main()
