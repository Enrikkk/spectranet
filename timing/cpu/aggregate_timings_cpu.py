"""CPU companion to aggregate_timings.py.

Reads results_cpu/*_timing.csv, joins with the hard-coded test L2 leaderboard,
writes results_cpu/leaderboard_speed.csv, and emits plots into plots_cpu/.
"""
import csv
import glob
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from paths_cpu import RESULTS_DIR, PLOTS_DIR  # noqa: E402

os.makedirs(PLOTS_DIR, exist_ok=True)


# Hard-coded accuracy table from RESEARCH_STATE.md (same as GPU aggregator).
ACC = {
    "FNO":                  (0.1024, "FNO-family"),
    "F_FNO":                (0.2331, "FNO-family"),
    "U_FNO":                (0.1218, "FNO-family"),
    "U_NO":                 (0.1697, "FNO-family"),
    "U_Net":                (0.1387, "U-Net"),
    "LSM":                  (0.1951, "Spectral"),
    "MWT":                  (0.1944, "Spectral"),
    "Galerkin_Transformer": (0.1097, "Transformer"),
    "Transformer":          (0.0284, "Transformer"),
    "OFormer":              (0.2162, "Transformer"),
    "FactFormer":           (0.1639, "Transformer"),
    "Transolver":           (0.2247, "Transformer"),
    "GNOT":                 (0.2362, "Graph/Attn"),
    "ONO":                  (0.3443, "Other"),
    "KNO2d":                (0.3092, "Other"),
    "SUNet_ablation_best":  (0.1643, "Ours"),
    "SUNet_goldstd":        (0.1942, "Ours"),
    "SUNet_ar2d":           (0.0941, "Ours"),
    "SUNet_ar2d_w32":       (0.0851, "Ours"),
    "SUNet_ar2d_w48":       (0.0836, "Ours"),
    "SUNet_ar2d_w64":       (0.0869, "Ours"),
    "SUNet_ar3d":           (0.1006, "Ours"),
}

CATEGORY_COLORS = {
    "FNO-family":  "#1f77b4",
    "U-Net":       "#9467bd",
    "Spectral":    "#2ca02c",
    "Transformer": "#d62728",
    "Graph/Attn":  "#8c564b",
    "Other":       "#7f7f7f",
    "Ours":        "#ff7f0e",
}


def display_label(model: str, variant: str) -> str:
    if model == "SUNet":
        pretty = {"ablation_best": "ablation",
                  "goldstd": "single-shot variant",
                  "ar2d": "w=20",
                  "ar2d_w32": "w=32 (canonical)",
                  "ar2d_w48": "w=48",
                  "ar2d_w64": "w=64",
                  "ar3d": "3D-spectral variant"}.get(variant, variant)
        return f"SpectraNet ({pretty})"
    return model


def acc_key(model: str, variant: str) -> str:
    if model == "SUNet" and variant:
        return f"SUNet_{variant}"
    return model


def load_rows():
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*_timing.csv"))):
        if os.path.basename(path) == "leaderboard_speed.csv":
            continue
        with open(path) as f:
            rd = csv.DictReader(f)
            for r in rd:
                r["batch_size"]  = int(r["batch_size"])
                r["params"]      = int(r["params"]) if r["params"] else 0
                for k in ("latency_ms_median", "latency_ms_p25", "latency_ms_p75",
                          "throughput_samples_per_sec", "peak_mem_mb"):
                    r[k] = float(r[k])
                r["display"] = display_label(r["model"], r.get("variant", ""))
                key = acc_key(r["model"], r.get("variant", ""))
                accuracy = ACC.get(key)
                r["test_l2"]  = accuracy[0] if accuracy else None
                r["category"] = accuracy[1] if accuracy else "Other"
                rows.append(r)
    return rows


def write_leaderboard(rows):
    out = os.path.join(RESULTS_DIR, "leaderboard_speed.csv")
    cols = ["display", "category", "params", "test_l2",
            "batch_size", "latency_ms_median", "latency_ms_p25", "latency_ms_p75",
            "throughput_samples_per_sec", "peak_mem_mb", "status"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out}  ({len(rows)} rows)")


def _by_batch(rows, B):
    keep = [r for r in rows if r["batch_size"] == B]
    keep.sort(key=lambda r: (r["status"] != "ok", r["latency_ms_median"]))
    return keep


def plot_latency_bar(rows, B, fname, title):
    sub = _by_batch(rows, B)
    if not sub:
        print(f"skip {fname}: no rows")
        return
    labels = [r["display"] for r in sub]
    med    = np.array([r["latency_ms_median"] for r in sub])
    lo     = np.array([r["latency_ms_p25"]    for r in sub])
    hi     = np.array([r["latency_ms_p75"]    for r in sub])
    err    = np.vstack([med - lo, hi - med])
    colors = [CATEGORY_COLORS.get(r["category"], "#999999") for r in sub]
    statuses = [r["status"] for r in sub]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(labels))))
    ypos = np.arange(len(labels))
    for i, (y, m, e_lo, e_hi, c, st) in enumerate(zip(ypos, med, err[0], err[1], colors, statuses)):
        if st != "ok":
            ax.barh(y, max(m, 1), color="white", edgecolor=c, hatch="///",
                    linewidth=1.5, label="_nolegend_")
            ax.text(max(m, 1) * 1.02, y, st, va="center", fontsize=7)
        else:
            ax.barh(y, m, xerr=[[e_lo], [e_hi]], color=c, edgecolor="black",
                    linewidth=0.5, capsize=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("latency per 10-step prediction (ms)  —  CPU")
    ax.set_title(title + "  (CPU)")
    seen = []
    handles = []
    for cat, c in CATEGORY_COLORS.items():
        if any(r["category"] == cat for r in sub) and cat not in seen:
            handles.append(plt.Rectangle((0, 0), 1, 1, color=c))
            seen.append(cat)
    ax.legend(handles, seen, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=140)
    plt.close(fig)
    print(f"wrote {fname}")


def plot_throughput(rows, B, fname, title):
    sub = _by_batch(rows, B)
    if not sub:
        print(f"skip {fname}: no rows"); return
    sub = sorted(sub,
                 key=lambda r: (r["status"] != "ok", -r["throughput_samples_per_sec"]))
    labels = [r["display"] for r in sub]
    thpt   = np.array([r["throughput_samples_per_sec"] for r in sub])
    colors = [CATEGORY_COLORS.get(r["category"], "#999999") for r in sub]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(labels))))
    ypos = np.arange(len(labels))
    for i, (y, t, c, st) in enumerate(zip(ypos, thpt, colors, [r["status"] for r in sub])):
        if st != "ok":
            ax.barh(y, max(t, 0.1), color="white", edgecolor=c, hatch="///")
            ax.text(max(t, 0.1) * 1.02, y, st, va="center", fontsize=7)
        else:
            ax.barh(y, t, color=c, edgecolor="black", linewidth=0.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("throughput (samples/sec)  —  CPU")
    ax.set_title(title + "  (CPU)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=140)
    plt.close(fig)
    print(f"wrote {fname}")


def plot_params_vs_latency(rows):
    sub = [r for r in _by_batch(rows, 1) if r["status"] == "ok" and r["params"] > 0]
    if not sub:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in sub:
        ax.scatter(r["params"], r["latency_ms_median"],
                   s=70, color=CATEGORY_COLORS.get(r["category"], "#999"),
                   edgecolor="black", linewidth=0.5, zorder=3)
        ax.annotate(r["display"], (r["params"], r["latency_ms_median"]),
                    xytext=(5, 3), textcoords="offset points", fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("trainable parameters (log)")
    ax.set_ylabel("latency at batch=1 (ms, log)  —  CPU")
    ax.set_title("Params vs latency on CPU  —  attention is far above the param-fit line")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "params_vs_latency.png"), dpi=140)
    plt.close(fig)
    print("wrote params_vs_latency.png")


def plot_l2_vs_latency(rows):
    sub = [r for r in _by_batch(rows, 1)
           if r["status"] == "ok" and r["test_l2"] is not None]
    if not sub:
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    for r in sub:
        marker = "*" if r["category"] == "Ours" else "o"
        size = 220 if r["category"] == "Ours" else 90
        ax.scatter(r["latency_ms_median"], r["test_l2"],
                   s=size, marker=marker,
                   color=CATEGORY_COLORS.get(r["category"], "#999"),
                   edgecolor="black", linewidth=0.7, zorder=3)
        ax.annotate(r["display"], (r["latency_ms_median"], r["test_l2"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=8)

    pts = sorted([(r["latency_ms_median"], r["test_l2"], r["display"]) for r in sub])
    pareto = []
    best_l2 = math.inf
    for lat, l2, name in pts:
        if l2 < best_l2:
            pareto.append((lat, l2, name))
            best_l2 = l2
    if len(pareto) >= 2:
        xs, ys, _ = zip(*pareto)
        ax.plot(xs, ys, "--", color="black", linewidth=1.2, alpha=0.5,
                label="Pareto frontier")
        ax.legend(loc="upper right")

    ax.set_xscale("log")
    ax.set_xlabel("latency at batch=1 (ms, log)  —  CPU, 10-step prediction")
    ax.set_ylabel("test L2 (lower = better)")
    ax.set_title("Accuracy vs Inference Cost on CPU  —  NS ν=1e-5")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "test_l2_vs_latency.png"), dpi=140)
    plt.close(fig)
    print("wrote test_l2_vs_latency.png  (HEADLINE)")


def plot_peak_memory(rows):
    # CPU run uses largest batch we have (4); fall back to 1 if needed.
    for B in (4, 1):
        sub = [r for r in _by_batch(rows, B) if r["status"] == "ok"]
        if sub:
            break
    if not sub:
        return
    sub = sorted(sub, key=lambda r: -r["peak_mem_mb"])
    labels = [r["display"] for r in sub]
    mem    = np.array([r["peak_mem_mb"] for r in sub])
    colors = [CATEGORY_COLORS.get(r["category"], "#999") for r in sub]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(labels))))
    ypos = np.arange(len(labels))
    ax.barh(ypos, mem, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(ypos); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlabel(f"process RSS delta (MB) at batch={B}")
    ax.set_title(f"Peak Process RSS Delta on CPU  —  batch={B}")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "peak_memory.png"), dpi=140)
    plt.close(fig)
    print("wrote peak_memory.png")


def main():
    rows = load_rows()
    if not rows:
        print(f"no rows found in {RESULTS_DIR}")
        return
    write_leaderboard(rows)
    plot_latency_bar(rows, 1, "latency_b1.png", "Latency  —  batch=1 (10-step prediction)")
    plot_latency_bar(rows, 4, "latency_b4.png", "Latency  —  batch=4 (10-step prediction)")
    plot_throughput(rows, 4, "throughput_b4.png", "Throughput  —  batch=4")
    plot_params_vs_latency(rows)
    plot_l2_vs_latency(rows)
    plot_peak_memory(rows)


if __name__ == "__main__":
    main()
