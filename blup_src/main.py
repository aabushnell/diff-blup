from __future__ import annotations

import sys
import csv
import time
from typing import Any

import numpy as np
from bokeh.events import DocumentReady
from bokeh.io import curdoc
from bokeh.models.callbacks import CustomJS
import pallas_trace as pallas

from data_model import QuantaQuery
from trace_session import TraceSession
from controller import AppController
from utils import timed

DEBUG_EXACT_COMPARE = False
DEBUG_EXPORT_FAST_BALANCED = True
DEBUG_TRACE_SIDE = "t1"          # "t1" or "t2"
DEBUG_THREAD_NAME = None         # e.g. "Rank 0 main thread"; None => first thread
DEBUG_N_BINS = 1                 # start with 
DEBUG_TOP_K = -1                 # disable truncation
DEBUG_TOP_N = 20                 # number of diff rows to print
DEBUG_WORST_BINS = 5
DEBUG_ROWS_PER_BIN = 5

def ns_to_ms(x: int) -> float:
    return float(x) / 1e6


def pct(x: float) -> str:
    return f"{100.0 * float(x):.2f}%"


def token_name_from_row(row: dict[str, Any], token_key_to_name: dict[str, str]) -> str:
    key = str(row["token_key"])
    return token_key_to_name.get(key, key)


def print_exact_compare_summary(report: dict[str, Any]) -> None:
    s = report["summary"]
    print("=== Exact implementation comparison ===")
    print(f"thread_id            : {s['thread_id']}")
    print(f"n_bins               : {s['n_bins']}")
    print(f"top_k                : {s['top_k']}")
    print(f"top_n                : {s['top_n']}")
    print(f"old_total_ns         : {s['old_total_ns']}")
    print(f"new_total_ns         : {s['new_total_ns']}")
    print(f"matched_total_ns     : {s['matched_total_ns']}")
    print(f"union_total_ns       : {s['union_total_ns']}")
    print(f"global_overlap_ratio : {pct(s['global_overlap_ratio'])}")
    print(f"old_row_count        : {s['old_row_count']}")
    print(f"new_row_count        : {s['new_row_count']}")
    print()


def print_diff_rows(
    rows: list[dict[str, Any]],
    token_key_to_name: dict[str, str],
    *,
    title: str,
    limit: int = 10,
) -> None:
    print(f"=== {title} ===")
    if not rows:
        print("(none)")
        print()
        return

    for i, row in enumerate(rows[:limit], start=1):
        name = token_name_from_row(row, token_key_to_name)
        print(
            f"{i:2d}. "
            f"{name:<30} "
            f"old={pct(row['old_prop']):>8} "
            f"new={pct(row['new_prop']):>8} "
            f"delta_ns={int(row['delta_ns']):>14} "
            f"abs_delta_prop={pct(row['abs_delta_prop']):>8}"
        )
    print()


def sort_bins_by_disagreement(per_bin: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        per_bin,
        key=lambda b: (
            float(b["overlap_ratio"]),
            -abs(int(b["old_total_ns"]) - int(b["new_total_ns"])),
        ),
    )


def print_worst_bins(
    report: dict[str, Any],
    token_key_to_name: dict[str, str],
    *,
    n_bins: int = 5,
    n_rows_per_bin: int = 5,
) -> None:
    bins = sort_bins_by_disagreement(report["per_bin"])
    print("=== Worst bins ===")
    if not bins:
        print("(none)")
        print()
        return

    for b in bins[:n_bins]:
        print(
            f"bin {b['bin_index']:>4d}  "
            f"[{ns_to_ms(int(b['start_ns'])):12.3f}, {ns_to_ms(int(b['finish_ns'])):12.3f}] ms  "
            f"overlap={pct(b['overlap_ratio'])}  "
            f"old_total={int(b['old_total_ns'])}  "
            f"new_total={int(b['new_total_ns'])}"
        )
        rows = b.get("largest_diffs", [])
        for row in rows[:n_rows_per_bin]:
            name = token_name_from_row(row, token_key_to_name)
            print(
                f"    {name:<30} "
                f"old={pct(row['old_prop']):>8} "
                f"new={pct(row['new_prop']):>8} "
                f"delta_ns={int(row['delta_ns']):>14}"
            )
        print()


def get_raw_trace(session: TraceSession) -> pallas.Trace:
    # Adjust this if your TraceSession stores the raw Trace under a different name.
    # Common choices would be .trace, ._trace, or similar.
    if session._trace is None:
        raise NotImplementedError
    return session._trace


def run_exact_debug(session: TraceSession) -> dict[str, Any]:
    meta = session.meta
    raw_trace = get_raw_trace(session)

    if DEBUG_THREAD_NAME is None:
        thread_id = int(meta.thread_ids[0])
        thread_name = meta.thread_id_to_name.get(thread_id, str(thread_id))
    else:
        thread_name = DEBUG_THREAD_NAME
        thread_id = int(meta.thread_name_to_id[thread_name])

    edges = np.linspace(
        int(meta.start_ns),
        int(meta.end_ns),
        int(DEBUG_N_BINS) + 1,
        dtype=np.uint64,
    )

    report = raw_trace.compare_exact_impls(
        thread_id=thread_id,
        bin_edges_ns=edges,
        top_k=DEBUG_TOP_K,
        top_n=DEBUG_TOP_N,
    )

    token_key_to_name = dict(meta.token_key_to_name)

    print()
    print(f"=== Debug trace side: {DEBUG_TRACE_SIDE} ===")
    print(f"=== Debug thread: {thread_name} ({thread_id}) ===")
    print_exact_compare_summary(report)
    print_diff_rows(
        report.get("largest_diffs", []), # type: ignore
        token_key_to_name,
        title="Largest whole-window differences",
        limit=DEBUG_TOP_N,
    )
    print_worst_bins(
        report,
        token_key_to_name,
        n_bins=min(DEBUG_WORST_BINS, DEBUG_N_BINS),
        n_rows_per_bin=DEBUG_ROWS_PER_BIN,
    )

    return report

def save_quanta_res_csv(res, path: str, token_key_to_name: dict[str, str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "start_ns", "finish_ns", "thread_id",
                "token_type", "token_id", "token_key",
                "token_name", "excl_ns", "proportion",
            ],
        )
        w.writeheader()
        for i in range(len(res)):
            token_type = int(res.token_type[i])
            token_id = int(res.token_id[i])
            tkey = f"{token_type}:{token_id}"
            w.writerow(
                {
                    "start_ns": int(res.start_ns[i]),
                    "finish_ns": int(res.finish_ns[i]),
                    "thread_id": int(res.thread_id[i]),
                    "token_type": token_type,
                    "token_id": token_id,
                    "token_key": tkey,
                    "token_name": token_key_to_name.get(tkey, tkey),
                    "excl_ns": int(res.excl_ns[i]),
                    "proportion": float(res.proportion[i]),
                }
            )

def save_quanta_bundle_csv(bundle, path: str, token_key_to_name: dict[str, str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "start_ns", "finish_ns", "thread_id",
                "token_type", "token_id", "token_key",
                "token_name", "excl_ns", "proportion",
            ],
        )
        w.writeheader()
        for i in range(len(bundle.start_ns)):
            token_type = int(bundle.token_type[i])
            token_id = int(bundle.token_id[i])
            tkey = f"{token_type}:{token_id}"
            w.writerow(
                {
                    "start_ns": int(bundle.start_ns[i]),
                    "finish_ns": int(bundle.end_ns[i]),
                    "thread_id": int(bundle.thread_id[i]),
                    "token_type": token_type,
                    "token_id": token_id,
                    "token_key": tkey,
                    "token_name": token_key_to_name.get(tkey, tkey),
                    "excl_ns": int(bundle.excl_ns[i]),
                    "proportion": float(bundle.proportion[i]),
                }
            )


def extract_bin_edges_from_quanta_res(res) -> tuple[int, ...]:
    bins = sorted(
        {
            (int(res.start_ns[i]), int(res.finish_ns[i]))
            for i in range(len(res))
        }
    )
    if not bins:
        return ()

    edges = [bins[0][0]]
    for start_ns, finish_ns in bins:
        if start_ns != edges[-1]:
            raise ValueError(
                f"Non-contiguous bins in debug exact result: got {start_ns}, expected {edges[-1]}"
            )
        if finish_ns <= start_ns:
            raise ValueError(f"Invalid bin [{start_ns}, {finish_ns})")
        edges.append(finish_ns)
    return tuple(edges)

def main():

    if len(sys.argv) < 3:
        raise SystemExit("Usage: bokeh serve --show main.py --args TRACE1 TRACE2")

    js_startup_timer = CustomJS(code="""
        const t_ready = performance.now();
        const t0 = window.pallas_startup_begin ?? t_ready;
        const startup_ms = t_ready - t0;

        console.log("=========================================");
        console.log("[BENCHMARK] PALLAS Startup Metrics");
        console.log(`  - Browser startup to document_ready : ${startup_ms.toFixed(2)} ms`);
        console.log("=========================================");
    """)
    curdoc().js_on_event(DocumentReady, js_startup_timer)

    print(">> Reading Traces")

    t1 = TraceSession(sys.argv[1])
    t2 = TraceSession(sys.argv[2])

    print(">> Opening Traces")

    with timed("t1 open"):
        t1.open()
    with timed("t2 open"):
        t2.open()

    print(">> Traces Read")

    if DEBUG_EXACT_COMPARE:
        debug_session = t1 if DEBUG_TRACE_SIDE == "t1" else t2
        with timed("exact debug compare"):
            report = run_exact_debug(debug_session)

        token_key_to_name = dict(debug_session.meta.token_key_to_name)
        save_quanta_res_csv(report["old"], "debug_exact_old.csv", token_key_to_name)
        save_quanta_res_csv(report["new"], "debug_exact_new.csv", token_key_to_name)

        if DEBUG_EXPORT_FAST_BALANCED:
            bin_edges_ns = extract_bin_edges_from_quanta_res(report["new"])
            if not bin_edges_ns:
                raise ValueError("Could not derive bin edges from exact debug output")

            thread_ids = tuple(sorted(set(int(x) for x in report["new"].thread_id)))
            if not thread_ids:
                raise ValueError("Could not derive thread ids from exact debug output")

            with timed("debug export fast"):
                fast_bundle = debug_session.query_quanta(
                    QuantaQuery(
                        thread_ids=thread_ids,
                        bin_edges_ns=bin_edges_ns,
                        fidelity="fast",
                        top_k=None,
                    )
                )
            save_quanta_bundle_csv(fast_bundle, "debug_fast.csv", token_key_to_name)

            with timed("debug export balanced"):
                balanced_bundle = debug_session.query_quanta(
                    QuantaQuery(
                        thread_ids=thread_ids,
                        bin_edges_ns=bin_edges_ns,
                        fidelity="balanced",
                        top_k=None,
                    )
                )
            save_quanta_bundle_csv(balanced_bundle, "debug_balanced.csv", token_key_to_name)

    with timed("build"):
        controller = AppController(t1, t2)
        curdoc().add_root(controller.build())
        curdoc().title = "Blup"

main()
