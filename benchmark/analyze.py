#!/usr/bin/env python3
"""Analyze benchmark results: summary stats, hypothesis tests, and plots.

Reads one or more JSONL result files produced by benchmark/harness.py.
Each file should represent a single experimental condition.

Usage:
    # Compare two verbosity conditions
    python -m benchmark.analyze \\
        --results results/full_c1.jsonl results/compact_c1.jsonl \\
        --labels  "full" "compact" \\
        --output  results/

    # Single file summary (no comparison)
    python -m benchmark.analyze --results results/full_c1.jsonl --output results/

Outputs:
    results/summary.csv              Per-condition aggregate stats
    results/plots/success_rate.png   Bar chart: success rate by condition × tier
    results/plots/tokens_box.png     Box plot: total tokens per task
    results/plots/rounds_cdf.png     CDF: rounds to success
    results/plots/latency_box.png    Box plot: task latency
    results/hyp_tests.txt            Mann-Whitney U results for each metric pair
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_conditions(paths: list[str], labels: list[str]) -> list[tuple[str, list[dict]]]:
    """Returns [(label, records), ...] in the same order as paths."""
    if labels and len(labels) != len(paths):
        raise ValueError(f"--labels count ({len(labels)}) must match --results count ({len(paths)})")
    result = []
    for i, path in enumerate(paths):
        label = labels[i] if labels else Path(path).stem
        records = load_jsonl(path)
        print(f"Loaded {len(records)} records from {path!r} → label={label!r}")
        result.append((label, records))
    return result


# ---------------------------------------------------------------------------
# Per-condition statistics
# ---------------------------------------------------------------------------

def _vals(records: list[dict], key: str) -> list[float]:
    return [r[key] for r in records if key in r and r[key] is not None]


def compute_stats(label: str, records: list[dict]) -> dict[str, Any]:
    import numpy as np

    n = len(records)
    if n == 0:
        return {"label": label, "n": 0}

    successes = [r for r in records if r.get("success")]
    n_success = len(successes)

    tokens = _vals(records, "total_tokens")
    latency = _vals(records, "total_latency_s")
    rounds_success = _vals(successes, "total_rounds")

    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {"mean": None, "median": None, "p25": None, "p75": None}
        a = np.array(vals)
        return {
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)),
        }

    # Per-tier breakdown
    tiers = ["easy", "medium", "hard", "unknown"]
    tier_success = {}
    for tier in tiers:
        tier_recs = [r for r in records if r.get("perturbation_tier") == tier]
        if tier_recs:
            tier_success[tier] = {
                "n": len(tier_recs),
                "success_rate": sum(1 for r in tier_recs if r.get("success")) / len(tier_recs),
            }

    return {
        "label": label,
        "n": n,
        "success_rate": n_success / n,
        "n_success": n_success,
        "tokens": _stats(tokens),
        "latency_s": _stats(latency),
        "rounds_to_success": _stats(rounds_success),
        "tier_breakdown": tier_success,
    }


def print_summary(stats_list: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for s in stats_list:
        label = s["label"]
        n = s["n"]
        sr = s.get("success_rate")
        tok = s.get("tokens", {})
        lat = s.get("latency_s", {})
        rts = s.get("rounds_to_success", {})

        print(f"\n[{label}]  n={n}  success={sr:.1%}" if sr is not None else f"\n[{label}]  n={n}")
        if tok.get("mean") is not None:
            print(f"  tokens:   mean={tok['mean']:.0f}  median={tok['median']:.0f}  "
                  f"[{tok['p25']:.0f}–{tok['p75']:.0f}]")
        if lat.get("mean") is not None:
            print(f"  latency:  mean={lat['mean']:.2f}s  median={lat['median']:.2f}s")
        if rts.get("mean") is not None:
            print(f"  rounds (success): mean={rts['mean']:.2f}  median={rts['median']:.2f}")
        if s.get("tier_breakdown"):
            for tier, ts in s["tier_breakdown"].items():
                print(f"  {tier:<8}: n={ts['n']}  success={ts['success_rate']:.1%}")


def save_summary_csv(stats_list: list[dict], output_dir: str) -> None:
    import csv

    path = os.path.join(output_dir, "summary.csv")
    rows = []
    for s in stats_list:
        row = {
            "label": s["label"],
            "n": s["n"],
            "success_rate": s.get("success_rate", ""),
            "tokens_mean": s.get("tokens", {}).get("mean", ""),
            "tokens_median": s.get("tokens", {}).get("median", ""),
            "latency_mean_s": s.get("latency_s", {}).get("mean", ""),
            "latency_median_s": s.get("latency_s", {}).get("median", ""),
            "rounds_mean": s.get("rounds_to_success", {}).get("mean", ""),
        }
        for tier, ts in (s.get("tier_breakdown") or {}).items():
            row[f"success_rate_{tier}"] = ts["success_rate"]
        rows.append(row)

    fieldnames = sorted({k for row in rows for k in row})
    # Put label first
    fieldnames = ["label"] + [f for f in fieldnames if f != "label"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary CSV → {path}")


# ---------------------------------------------------------------------------
# Hypothesis testing
# ---------------------------------------------------------------------------

def run_hypothesis_tests(
    conditions: list[tuple[str, list[dict]]],
    output_dir: str,
) -> None:
    try:
        from scipy import stats as sci_stats
    except ImportError:
        print("\n[WARN] scipy not installed — skipping hypothesis tests. pip install scipy")
        return

    if len(conditions) < 2:
        print("\n[INFO] Need ≥2 conditions for hypothesis tests.")
        return

    lines: list[str] = []
    lines.append("Hypothesis Tests (Mann-Whitney U, two-sided, α=0.05)")
    lines.append("=" * 60)

    metrics = [
        ("total_tokens", "Total tokens per task"),
        ("total_latency_s", "Task latency (s)"),
        ("total_rounds", "Rounds per task"),
    ]

    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):
            label_a, recs_a = conditions[i]
            label_b, recs_b = conditions[j]
            lines.append(f"\n{label_a!r} vs {label_b!r}")
            lines.append("-" * 40)

            # Success rate Fisher's exact test
            na = len(recs_a)
            nb = len(recs_b)
            sa = sum(1 for r in recs_a if r.get("success"))
            sb = sum(1 for r in recs_b if r.get("success"))
            table = [[sa, na - sa], [sb, nb - sb]]
            _, p_fisher = sci_stats.fisher_exact(table)
            sig = "**" if p_fisher < 0.05 else "ns"
            lines.append(
                f"  success rate: {sa}/{na}={sa/na:.1%} vs {sb}/{nb}={sb/nb:.1%}  "
                f"p={p_fisher:.4f} {sig}"
            )

            for key, label in metrics:
                vals_a = _vals(recs_a, key)
                vals_b = _vals(recs_b, key)
                if not vals_a or not vals_b:
                    lines.append(f"  {label}: insufficient data")
                    continue
                stat, p = sci_stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
                # Rank-biserial correlation as effect size
                n1, n2 = len(vals_a), len(vals_b)
                r_effect = 1 - (2 * stat) / (n1 * n2)
                sig = "**" if p < 0.05 else "ns"
                import numpy as np
                lines.append(
                    f"  {label}: median {np.median(vals_a):.2f} vs {np.median(vals_b):.2f}  "
                    f"U={stat:.0f}  p={p:.4f}  r={r_effect:.3f}  {sig}"
                )

    report = "\n".join(lines)
    print("\n" + report)
    path = os.path.join(output_dir, "hyp_tests.txt")
    with open(path, "w") as f:
        f.write(report + "\n")
    print(f"\nHypothesis tests → {path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_ttft_bar(
    prom_paths: list[str],
    labels: list[str],
    output_dir: str,
) -> None:
    """Bar chart of mean TTFT and mean e2e latency from Prometheus delta snapshots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\n[WARN] matplotlib not installed — skipping TTFT plot.")
        return

    ttft_means, e2e_means = [], []
    for path in prom_paths:
        with open(path) as f:
            snap = json.load(f)
        before, after = snap["before"], snap["after"]
        d_count = after["vllm:time_to_first_token_seconds_count"] - before["vllm:time_to_first_token_seconds_count"]
        d_ttft = after["vllm:time_to_first_token_seconds_sum"] - before["vllm:time_to_first_token_seconds_sum"]
        d_e2e = after["vllm:e2e_request_latency_seconds_sum"] - before["vllm:e2e_request_latency_seconds_sum"]
        ttft_means.append((d_ttft / d_count * 1000) if d_count > 0 else 0.0)  # ms
        e2e_means.append((d_e2e / d_count * 1000) if d_count > 0 else 0.0)

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(labels)), 5))
    ax.bar(x - width / 2, ttft_means, width, label="Mean TTFT (ms)", color="#cce5ff")
    ax.bar(x + width / 2, e2e_means, width, label="Mean e2e latency (ms)", color="#d4edda")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Engine-level latency by configuration (Prometheus)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(plots_dir, "ttft_bar.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot → {path}")


def make_plots(
    conditions: list[tuple[str, list[dict]]],
    stats_list: list[dict],
    output_dir: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\n[WARN] matplotlib not installed — skipping plots. pip install matplotlib")
        return

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    labels = [label for label, _ in conditions]

    # ------------------------------------------------------------------
    # 1. Success rate by condition × perturbation tier
    # ------------------------------------------------------------------
    tiers = ["easy", "medium", "hard"]
    n_cond = len(conditions)
    x = np.arange(len(tiers))
    width = 0.8 / n_cond

    fig, ax = plt.subplots(figsize=(8, 5))
    for ci, (label, recs) in enumerate(conditions):
        sr_per_tier = []
        for tier in tiers:
            tier_recs = [r for r in recs if r.get("perturbation_tier") == tier]
            if tier_recs:
                sr_per_tier.append(sum(1 for r in tier_recs if r.get("success")) / len(tier_recs))
            else:
                sr_per_tier.append(0.0)
        offset = (ci - n_cond / 2 + 0.5) * width
        ax.bar(x + offset, sr_per_tier, width, label=label)

    ax.set_xlabel("Perturbation tier")
    ax.set_ylabel("Success rate")
    ax.set_title("Agent success rate by tier and condition")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    plt.tight_layout()
    path = os.path.join(plots_dir, "success_rate.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot → {path}")

    # ------------------------------------------------------------------
    # 2. Total tokens box plot
    # ------------------------------------------------------------------
    fig_w = max(6, 1.4 * n_cond)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    data = [_vals(recs, "total_tokens") for _, recs in conditions]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#cce5ff")
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Total tokens per task")
    ax.set_title("Token usage by condition")
    plt.tight_layout()
    path = os.path.join(plots_dir, "tokens_box.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot → {path}")

    # ------------------------------------------------------------------
    # 3. Rounds to success CDF
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    max_rounds = 0
    for label, recs in conditions:
        successes = [r for r in recs if r.get("success")]
        rounds = sorted(_vals(successes, "total_rounds"))
        if not rounds:
            continue
        max_rounds = max(max_rounds, int(rounds[-1]))
        cdf = np.arange(1, len(rounds) + 1) / len(recs)  # fraction of ALL tasks
        ax.step(rounds, cdf, where="post", label=label)
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Fraction of tasks resolved")
    ax.set_title("CDF of rounds to success")
    if max_rounds:
        ax.set_xlim(0, max_rounds + 0.5)
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(plots_dir, "rounds_cdf.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot → {path}")

    # ------------------------------------------------------------------
    # 4. Task latency box plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    data = [_vals(recs, "total_latency_s") for _, recs in conditions]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#d4edda")
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Task latency (s)")
    ax.set_title("End-to-end latency by condition")
    plt.tight_layout()
    path = os.path.join(plots_dir, "latency_box.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot → {path}")

    # ------------------------------------------------------------------
    # 5. Reflection budget curve
    # ------------------------------------------------------------------
    budgets = list(range(1, 6))
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, recs in conditions:
        n_total = len(recs)
        sr_by_budget = []
        for b in budgets:
            solved = sum(1 for r in recs if r.get("success") and r.get("total_rounds", 99) <= b)
            sr_by_budget.append(solved / n_total if n_total else 0.0)
        ax.plot(budgets, sr_by_budget, marker="o", label=label)
    ax.set_xlabel("Reflection budget (max rounds)")
    ax.set_ylabel("Success rate")
    ax.set_title("Success rate vs. reflection budget")
    ax.set_xticks(budgets)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    plt.tight_layout()
    path = os.path.join(plots_dir, "budget_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Analyze benchmark JSONL results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--results", nargs="+", required=True,
                   help="One or more JSONL result files (one per condition)")
    p.add_argument("--labels", nargs="*", default=None,
                   help="Display labels for each file (defaults to filename stems)")
    p.add_argument("--output", default="results/",
                   help="Directory for output files (default: results/)")
    p.add_argument("--prom-snapshots", nargs="*", default=None,
                   help="Prometheus _prom.json files (one per condition, same order as --results)")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    labels = args.labels or []
    conditions = load_conditions(args.results, labels)
    resolved_labels = [label for label, _ in conditions]

    stats_list = [compute_stats(label, recs) for label, recs in conditions]
    print_summary(stats_list)
    save_summary_csv(stats_list, args.output)
    run_hypothesis_tests(conditions, args.output)
    make_plots(conditions, stats_list, args.output)

    if args.prom_snapshots:
        if len(args.prom_snapshots) != len(conditions):
            print(f"[WARN] --prom-snapshots count ({len(args.prom_snapshots)}) != conditions ({len(conditions)}), skipping TTFT plot")
        else:
            make_ttft_bar(args.prom_snapshots, resolved_labels, args.output)

    print(f"\nAll outputs written to {args.output}")


if __name__ == "__main__":
    main()
