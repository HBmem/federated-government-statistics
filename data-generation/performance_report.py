#!/usr/bin/env python3
"""
Generate performance graphs from a federated government statistics output JSON.

This script reads a coordinator output JSON file and produces a set of PNG charts
that summarize:
- worker-level timing metrics
- worker CPU and memory usage
- per-query-template timing and database metrics
- row quality metrics by node and metric
- selected global aggregate distributions

Usage:
    python3 performance_report.py path/to/run_output.json
    python3 performance_report.py path/to/run_output.json --out-dir reports/run_001

Dependencies:
    - Python 3.9+
    - matplotlib
    - pandas
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate performance graphs from a federated output JSON file.")
    parser.add_argument("input_json", help="Path to the coordinator output JSON file.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory where graphs and summary tables will be written. Defaults to a folder next to the input file.",
    )
    return parser.parse_args()


def safe_float(value: Any) -> Optional[float]:
    """Convert a value to float when possible. Return None for null/empty/invalid values."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def safe_int(value: Any) -> Optional[int]:
    """Convert a value to int when possible. Return None for null/empty/invalid values."""
    num = safe_float(value)
    if num is None:
        return None
    return int(num)


def short_template_name(path: Optional[str]) -> str:
    """Return the final file name of a template path."""
    if not path:
        return ""
    return Path(path).name


def schema_sort_key(schema: str) -> tuple[int, str]:
    """Stable schema ordering for plots."""
    order = {"A": 0, "B": 1, "C": 2}
    return (order.get(schema, 99), schema)


def load_report(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_worker_df(report: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for node in report.get("nodes", []):
        wm = node.get("worker_metrics", {}) or {}
        discovery = node.get("discovery", {}) or {}

        row = {
            "rank": node.get("rank"),
            "county_fips": node.get("county_fips"),
            "schema_type": discovery.get("schema_type") or node.get("results", {}).get("schema_type"),
            "status": node.get("status"),
            "worker_total_ms": safe_float(wm.get("worker_total_ms")),
            "db_connect_ms": safe_float(wm.get("db_connect_ms")),
            "discovery_ms": safe_float(wm.get("discovery_ms")),
            "execution_ms": safe_float(wm.get("execution_ms")),
            "result_serialize_ms": safe_float(wm.get("result_serialize_ms")),
            "result_send_ms": safe_float(wm.get("result_send_ms")),
            "cpu_user_ms": safe_float(wm.get("cpu_user_ms")),
            "cpu_system_ms": safe_float(wm.get("cpu_system_ms")),
            "rss_kb_at_start": safe_float(wm.get("rss_kb_at_start")),
            "rss_kb_at_end": safe_float(wm.get("rss_kb_at_end")),
            "max_rss_kb": safe_float(wm.get("max_rss_kb")),
        }
        row["node_label"] = f"rank {row['rank']} ({row['county_fips']}, {row['schema_type']})"
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["schema_type", "rank"], key=lambda s: s.map(lambda x: schema_sort_key(str(x))))
    return df


def extract_query_df(report: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for node in report.get("nodes", []):
        schema_type = node.get("discovery", {}).get("schema_type") or node.get("results", {}).get("schema_type")
        county_fips = node.get("county_fips")
        rank = node.get("rank")

        for qm in node.get("query_metrics", []) or []:
            row = {
                "rank": rank,
                "county_fips": county_fips,
                "schema_type": schema_type,
                "metric_name": qm.get("metric_name"),
                "template_path": qm.get("template_path"),
                "template_name": short_template_name(qm.get("template_path")),
                "query_ms": safe_float(qm.get("query_ms")),
                "rows_returned": safe_float(qm.get("rows_returned")),
                "success": bool(qm.get("success")),
                "error_message": qm.get("error_message"),
                "db_cpu_time_ms": safe_float(qm.get("db_cpu_time_ms")),
                "db_shared_hits": safe_float(qm.get("db_shared_hits")),
                "db_reads": safe_float(qm.get("db_reads")),
                "db_temp_reads": safe_float(qm.get("db_temp_reads")),
                "db_temp_writes": safe_float(qm.get("db_temp_writes")),
            }
            row["node_label"] = f"rank {rank} ({county_fips}, {schema_type})"
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["metric_name", "schema_type", "rank"], key=lambda s: s.map(lambda x: schema_sort_key(str(x)) if s.name == "schema_type" else x))
    return df


def extract_quality_df(report: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for node in report.get("nodes", []):
        rank = node.get("rank")
        county_fips = node.get("county_fips")
        schema_type = node.get("discovery", {}).get("schema_type") or node.get("results", {}).get("schema_type")
        metrics = (node.get("results", {}) or {}).get("metrics", {}) or {}

        for metric_name, metric_payload in metrics.items():
            row = {
                "rank": rank,
                "county_fips": county_fips,
                "schema_type": schema_type,
                "metric_name": metric_name,
                "rows_scanned": safe_float(metric_payload.get("rows_scanned")),
                "rows_used": safe_float(metric_payload.get("rows_used")),
                "rows_dropped": safe_float(metric_payload.get("rows_dropped")),
                "drop_missing_required": safe_float(metric_payload.get("drop_missing_required")),
                "drop_missing_verification": safe_float(metric_payload.get("drop_missing_verification")),
                "drop_invalid_date": safe_float(metric_payload.get("drop_invalid_date")),
                "drop_invalid_age": safe_float(metric_payload.get("drop_invalid_age")),
                "drop_negative_income": safe_float(metric_payload.get("drop_negative_income")),
                "drop_inconsistent_residency": safe_float(metric_payload.get("drop_inconsistent_residency")),
                "drop_other": safe_float(metric_payload.get("drop_other")),
            }
            row["drop_rate"] = (
                (row["rows_dropped"] / row["rows_scanned"]) if row["rows_scanned"] not in (None, 0) else None
            )
            row["node_label"] = f"rank {rank} ({county_fips}, {schema_type})"
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["metric_name", "schema_type", "rank"], key=lambda s: s.map(lambda x: schema_sort_key(str(x)) if s.name == "schema_type" else x))
    return df


def save_df(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    if not df.empty:
        df.to_csv(out_dir / f"{name}.csv", index=False)


def rotate_xticks(ax, rotation: int = 30) -> None:
    for tick in ax.get_xticklabels():
        tick.set_rotation(rotation)
        tick.set_ha("right")


def save_current(fig, out_dir: Path, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_worker_total_time(worker_df: pd.DataFrame, out_dir: Path) -> None:
    if worker_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(worker_df["node_label"], worker_df["worker_total_ms"])
    ax.set_title("Worker Total Runtime by Node")
    ax.set_ylabel("Milliseconds")
    ax.set_xlabel("Worker Node")
    rotate_xticks(ax)
    save_current(fig, out_dir, "01_worker_total_runtime.png")


def plot_worker_phase_breakdown(worker_df: pd.DataFrame, out_dir: Path) -> None:
    if worker_df.empty:
        return

    phases = ["db_connect_ms", "discovery_ms", "execution_ms", "result_serialize_ms", "result_send_ms"]
    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = pd.Series([0.0] * len(worker_df))

    for phase in phases:
        values = worker_df[phase].fillna(0.0)
        ax.bar(worker_df["node_label"], values, bottom=bottom, label=phase)
        bottom = bottom + values

    ax.set_title("Worker Runtime Phase Breakdown")
    ax.set_ylabel("Milliseconds")
    ax.set_xlabel("Worker Node")
    ax.legend()
    rotate_xticks(ax)
    save_current(fig, out_dir, "02_worker_phase_breakdown.png")


def plot_worker_cpu(worker_df: pd.DataFrame, out_dir: Path) -> None:
    if worker_df.empty:
        return

    x = range(len(worker_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], worker_df["cpu_user_ms"].fillna(0), width=width, label="cpu_user_ms")
    ax.bar([i + width / 2 for i in x], worker_df["cpu_system_ms"].fillna(0), width=width, label="cpu_system_ms")

    ax.set_xticks(list(x))
    ax.set_xticklabels(worker_df["node_label"])
    ax.set_title("Worker CPU Time")
    ax.set_ylabel("Milliseconds")
    ax.set_xlabel("Worker Node")
    ax.legend()
    rotate_xticks(ax)
    save_current(fig, out_dir, "03_worker_cpu_time.png")


def plot_worker_memory(worker_df: pd.DataFrame, out_dir: Path) -> None:
    if worker_df.empty:
        return

    x = range(len(worker_df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar([i - width for i in x], worker_df["rss_kb_at_start"].fillna(0), width=width, label="rss_kb_at_start")
    ax.bar(list(x), worker_df["rss_kb_at_end"].fillna(0), width=width, label="rss_kb_at_end")
    ax.bar([i + width for i in x], worker_df["max_rss_kb"].fillna(0), width=width, label="max_rss_kb")

    ax.set_xticks(list(x))
    ax.set_xticklabels(worker_df["node_label"])
    ax.set_title("Worker Memory Usage")
    ax.set_ylabel("KB")
    ax.set_xlabel("Worker Node")
    ax.legend()
    rotate_xticks(ax)
    save_current(fig, out_dir, "04_worker_memory_usage.png")


def plot_query_runtime_by_node(query_df: pd.DataFrame, out_dir: Path) -> None:
    if query_df.empty:
        return

    pivot = query_df.pivot_table(index="metric_name", columns="node_label", values="query_ms", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Per-Query Runtime by Node")
    ax.set_ylabel("Milliseconds")
    ax.set_xlabel("Metric")
    ax.legend(title="Node", bbox_to_anchor=(1.02, 1), loc="upper left")
    rotate_xticks(ax, rotation=0)
    save_current(fig, out_dir, "05_query_runtime_by_node.png")


def plot_query_runtime_by_schema(query_df: pd.DataFrame, out_dir: Path) -> None:
    if query_df.empty:
        return

    pivot = query_df.pivot_table(index="metric_name", columns="schema_type", values="query_ms", aggfunc="mean")
    pivot = pivot.reindex(columns=sorted(pivot.columns, key=schema_sort_key))

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Average Query Runtime by Schema")
    ax.set_ylabel("Milliseconds")
    ax.set_xlabel("Metric")
    ax.legend(title="Schema")
    rotate_xticks(ax, rotation=0)
    save_current(fig, out_dir, "06_query_runtime_by_schema.png")


def plot_db_cpu_vs_wall(query_df: pd.DataFrame, out_dir: Path) -> None:
    if query_df.empty:
        return

    subset = query_df.dropna(subset=["db_cpu_time_ms", "query_ms"]).copy()
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for schema in sorted(subset["schema_type"].dropna().unique(), key=schema_sort_key):
        sdf = subset[subset["schema_type"] == schema]
        ax.scatter(sdf["db_cpu_time_ms"], sdf["query_ms"], label=f"Schema {schema}")

        for _, row in sdf.iterrows():
            ax.annotate(
                f"{row['metric_name']} r{row['rank']}",
                (row["db_cpu_time_ms"], row["query_ms"]),
                fontsize=7,
                alpha=0.8,
            )

    max_val = max(subset["db_cpu_time_ms"].max(), subset["query_ms"].max())
    ax.plot([0, max_val], [0, max_val], linestyle="--")
    ax.set_title("Database CPU Time vs Wall Query Time")
    ax.set_xlabel("db_cpu_time_ms")
    ax.set_ylabel("query_ms")
    ax.legend()
    save_current(fig, out_dir, "07_db_cpu_vs_wall_time.png")


def plot_buffer_hits_reads(query_df: pd.DataFrame, out_dir: Path) -> None:
    if query_df.empty:
        return

    agg = (
        query_df.groupby(["schema_type", "metric_name"], dropna=False)[["db_shared_hits", "db_reads"]]
        .mean()
        .reset_index()
    )
    agg["label"] = agg["schema_type"].astype(str) + "-" + agg["metric_name"].astype(str)

    x = range(len(agg))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([i - width / 2 for i in x], agg["db_shared_hits"].fillna(0), width=width, label="db_shared_hits")
    ax.bar([i + width / 2 for i in x], agg["db_reads"].fillna(0), width=width, label="db_reads")

    ax.set_xticks(list(x))
    ax.set_xticklabels(agg["label"])
    ax.set_title("Average Database Shared Hits and Reads by Schema/Metric")
    ax.set_ylabel("Blocks")
    ax.set_xlabel("Schema-Metric")
    ax.legend()
    rotate_xticks(ax)
    save_current(fig, out_dir, "08_db_buffer_hits_reads.png")


def plot_rows_used_dropped(quality_df: pd.DataFrame, out_dir: Path) -> None:
    if quality_df.empty:
        return

    subset = quality_df[quality_df["metric_name"] == "population"].copy()
    if subset.empty:
        subset = quality_df.copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(subset["node_label"], subset["rows_used"].fillna(0), label="rows_used")
    ax.bar(
        subset["node_label"],
        subset["rows_dropped"].fillna(0),
        bottom=subset["rows_used"].fillna(0),
        label="rows_dropped",
    )
    ax.set_title("Rows Used vs Dropped by Node")
    ax.set_ylabel("Rows")
    ax.set_xlabel("Worker Node")
    ax.legend()
    rotate_xticks(ax)
    save_current(fig, out_dir, "09_rows_used_vs_dropped.png")


def plot_drop_reason_heatmap(quality_df: pd.DataFrame, out_dir: Path) -> None:
    if quality_df.empty:
        return

    reason_cols = [
        "drop_missing_required",
        "drop_missing_verification",
        "drop_invalid_date",
        "drop_invalid_age",
        "drop_negative_income",
        "drop_inconsistent_residency",
        "drop_other",
    ]

    subset = (
        quality_df.groupby(["schema_type", "metric_name"], dropna=False)[reason_cols]
        .sum()
        .reset_index()
    )
    if subset.empty:
        return

    labels = subset["schema_type"].astype(str) + "-" + subset["metric_name"].astype(str)
    matrix = subset[reason_cols].fillna(0).to_numpy()

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_title("Drop Reason Totals by Schema/Metric")
    ax.set_xlabel("Drop Reason")
    ax.set_ylabel("Schema-Metric")
    ax.set_xticks(range(len(reason_cols)))
    ax.set_xticklabels(reason_cols)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    rotate_xticks(ax, rotation=35)
    fig.colorbar(im, ax=ax)
    save_current(fig, out_dir, "10_drop_reason_heatmap.png")


def plot_drop_rate_by_metric(quality_df: pd.DataFrame, out_dir: Path) -> None:
    if quality_df.empty:
        return

    agg = quality_df.groupby(["schema_type", "metric_name"], dropna=False)["drop_rate"].mean().reset_index()
    agg["label"] = agg["schema_type"].astype(str) + "-" + agg["metric_name"].astype(str)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(agg["label"], agg["drop_rate"].fillna(0))
    ax.set_title("Average Drop Rate by Schema/Metric")
    ax.set_ylabel("Drop Rate")
    ax.set_xlabel("Schema-Metric")
    rotate_xticks(ax)
    save_current(fig, out_dir, "11_drop_rate_by_metric.png")


def plot_age_distribution(report: Dict[str, Any], out_dir: Path) -> None:
    global_aggs = report.get("global_aggregates", {}) or {}

    age_keys = [
        "age_0_4", "age_5_9", "age_10_14", "age_15_17", "age_18_24", "age_25_34",
        "age_35_44", "age_45_54", "age_55_64", "age_65_74", "age_75_84", "age_85_plus"
    ]
    values = [safe_float(global_aggs.get(k)) or 0 for k in age_keys]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(age_keys, values)
    ax.set_title("Global Age Bucket Distribution")
    ax.set_ylabel("Count")
    ax.set_xlabel("Age Bucket")
    rotate_xticks(ax)
    save_current(fig, out_dir, "12_global_age_distribution.png")


def plot_income_distribution(report: Dict[str, Any], out_dir: Path) -> None:
    global_aggs = report.get("global_aggregates", {}) or {}

    inc_keys = [
        "inc_lt_25k", "inc_25_50k", "inc_50_75k", "inc_75_100k",
        "inc_100_150k", "inc_150_200k", "inc_200k_plus"
    ]
    values = [safe_float(global_aggs.get(k)) or 0 for k in inc_keys]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(inc_keys, values)
    ax.set_title("Global Income Bucket Distribution")
    ax.set_ylabel("Count")
    ax.set_xlabel("Income Bucket")
    rotate_xticks(ax)
    save_current(fig, out_dir, "13_global_income_distribution.png")


def plot_success_rate(query_df: pd.DataFrame, out_dir: Path) -> None:
    if query_df.empty:
        return

    agg = query_df.groupby(["schema_type", "metric_name"], dropna=False)["success"].mean().reset_index()
    agg["label"] = agg["schema_type"].astype(str) + "-" + agg["metric_name"].astype(str)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(agg["label"], agg["success"].fillna(0))
    ax.set_title("Query Success Rate by Schema/Metric")
    ax.set_ylabel("Success Rate")
    ax.set_xlabel("Schema-Metric")
    ax.set_ylim(0, 1.05)
    rotate_xticks(ax)
    save_current(fig, out_dir, "14_query_success_rate.png")


def plot_query_time_boxplot(query_df: pd.DataFrame, out_dir: Path) -> None:
    if query_df.empty:
        return

    metrics = list(query_df["metric_name"].dropna().unique())
    if not metrics:
        return

    data = [query_df.loc[query_df["metric_name"] == m, "query_ms"].dropna().tolist() for m in metrics]
    if not any(data):
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, tick_labels=metrics)
    ax.set_title("Query Runtime Distribution by Metric")
    ax.set_ylabel("Milliseconds")
    ax.set_xlabel("Metric")
    rotate_xticks(ax, rotation=0)
    save_current(fig, out_dir, "15_query_time_boxplot.png")


def build_summary_text(report: Dict[str, Any], worker_df: pd.DataFrame, query_df: pd.DataFrame, quality_df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("Performance Summary")
    lines.append("===================")
    lines.append("")

    run_info = report.get("run", {})
    tw = run_info.get("time_window", {})
    lines.append(f"Category: {run_info.get('category')}")
    lines.append(f"Time window: {tw.get('start_date')} to {tw.get('end_date')}")
    lines.append(f"Enabled metrics: {', '.join(run_info.get('enabled_metrics', []))}")
    lines.append("")

    if not worker_df.empty:
        slowest = worker_df.loc[worker_df["worker_total_ms"].idxmax()]
        fastest = worker_df.loc[worker_df["worker_total_ms"].idxmin()]
        lines.append("Worker runtime")
        lines.append(f"- Slowest worker: {slowest['node_label']} at {slowest['worker_total_ms']:.2f} ms")
        lines.append(f"- Fastest worker: {fastest['node_label']} at {fastest['worker_total_ms']:.2f} ms")
        lines.append(f"- Mean worker_total_ms: {worker_df['worker_total_ms'].mean():.2f}")
        lines.append("")

    if not query_df.empty:
        avg_metric = query_df.groupby("metric_name", dropna=False)["query_ms"].mean().sort_values(ascending=False)
        lines.append("Average query runtime by metric")
        for metric, value in avg_metric.items():
            lines.append(f"- {metric}: {value:.2f} ms")
        lines.append("")

        avg_schema = query_df.groupby("schema_type", dropna=False)["query_ms"].mean().sort_values(ascending=False)
        lines.append("Average query runtime by schema")
        for schema, value in avg_schema.items():
            lines.append(f"- Schema {schema}: {value:.2f} ms")
        lines.append("")

    if not quality_df.empty:
        avg_drop = quality_df.groupby(["schema_type", "metric_name"], dropna=False)["drop_rate"].mean().sort_values(ascending=False)
        lines.append("Average drop rate by schema/metric")
        for (schema, metric), value in avg_drop.items():
            if pd.notna(value):
                lines.append(f"- Schema {schema} / {metric}: {value:.4f}")
        lines.append("")

    failures = report.get("failures", [])
    lines.append(f"Failure count: {len(failures)}")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json).resolve()
    report = load_report(input_path)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else input_path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)

    worker_df = extract_worker_df(report)
    query_df = extract_query_df(report)
    quality_df = extract_quality_df(report)

    save_df(worker_df, out_dir, "worker_metrics")
    save_df(query_df, out_dir, "query_metrics")
    save_df(quality_df, out_dir, "quality_metrics")

    plot_worker_total_time(worker_df, out_dir)
    plot_worker_phase_breakdown(worker_df, out_dir)
    plot_worker_cpu(worker_df, out_dir)
    plot_worker_memory(worker_df, out_dir)

    plot_query_runtime_by_node(query_df, out_dir)
    plot_query_runtime_by_schema(query_df, out_dir)
    plot_db_cpu_vs_wall(query_df, out_dir)
    plot_buffer_hits_reads(query_df, out_dir)
    plot_rows_used_dropped(quality_df, out_dir)
    plot_drop_reason_heatmap(quality_df, out_dir)
    plot_drop_rate_by_metric(quality_df, out_dir)
    plot_age_distribution(report, out_dir)
    plot_income_distribution(report, out_dir)
    plot_success_rate(query_df, out_dir)
    plot_query_time_boxplot(query_df, out_dir)

    summary = build_summary_text(report, worker_df, query_df, quality_df)
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")

    print(f"Saved graphs and summaries to: {out_dir}")


if __name__ == "__main__":
    main()
