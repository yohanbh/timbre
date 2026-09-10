"""Plot a completed Phase 2 sweep (requires matplotlib).

    python3 tests/plot_hnsw.py store/hnsw_phase2/results.json docs/hnsw_frontier.svg
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "timbre-hnsw"})
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    report = json.loads(args.results.read_text())
    fig, ax = plt.subplots(figsize=(8, 4.8))
    groups = sorted({(p["M"], p["ef_construction"]) for p in report["points"]})
    for M, ef in groups:
        points = sorted([p for p in report["points"]
                         if (p["M"], p["ef_construction"]) == (M, ef)],
                        key=lambda p: p["ef_search"])
        ax.plot([p["p50_ms"] for p in points], [p["recall_at_10"] for p in points],
                marker="o", markersize=4, label=f"M={M}, efConstruction={ef}")
    ax.axhline(0.95, color="#666666", linestyle="--", linewidth=1, label="Phase 2 target: 0.95")
    ax.set(xscale="log", xlabel="Median query latency (ms, warm cache)", ylabel="Recall@10",
           ylim=(min(p["recall_at_10"] for p in report["points"]) - 0.025, 1.005))
    ax.set_title(f"HNSW: {report['vectors']:,} segments, {report['queries']:,} held-out queries\n"
                 "Python + NumPy, one CPU thread", loc="left", fontsize=12)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, metadata={"Date": None})
    plt.close(fig)
    if args.out.suffix.lower() == ".svg":
        args.out.write_text("\n".join(line.rstrip() for line in args.out.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
