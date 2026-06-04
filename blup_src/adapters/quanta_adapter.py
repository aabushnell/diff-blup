from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from trace_session import QuantaBundle


PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ab",
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
]

def empty_quanta_source() -> dict:
    return dict(
        left=[],
        right=[],
        top=[],
        bottom=[],
        color=[],
        function=[],
        thread=[],
        proportion=[],
        exclusive_s=[],
    )

def build_color_map(functions: Iterable[str]) -> dict[str, str]:
    funcs = sorted(set(str(f) for f in functions))
    return {f: PALETTE[i % len(PALETTE)] for i, f in enumerate(funcs)}

def quanta_bundle_to_bokeh_source(
    bundle: QuantaBundle,
    *,
    active_threads: list[str],
    trace_side: str,
    color_map: dict[str, str],
    stack_order: str = "global",
) -> dict:
    if len(bundle.function) == 0:
        return empty_quanta_source()

    rows = []
    for i in range(len(bundle.function)):
        rows.append({
            "left": int(bundle.bin_start_ns[i]) / 1e6,
            "right": int(bundle.bin_end_ns[i]) / 1e6,
            "thread": str(bundle.thread_name[i]),
            "function": str(bundle.function[i]),
            "proportion": float(bundle.proportion[i]),
            "exclusive_s": int(bundle.exclusive_ns[i]) / 1e9,
        })

    if stack_order == "global":
        totals = defaultdict(float)
        for r in rows:
            totals[r["function"]] += r["exclusive_s"]
        rank = {f: i for i, (f, _) in enumerate(sorted(totals.items(), key=lambda kv: -kv[1]))}
        sort_key_fn = lambda row: rank.get(row["function"], 10**9)
    else:
        sort_key_fn = lambda row: -row["proportion"]

    half = 0.45
    padding = 0.02
    grouped = defaultdict(list)
    for r in rows:
        key = (r["thread"], r["left"], r["right"])
        grouped[key].append(r)

    out = empty_quanta_source()
    for (thread, left, right), grp in grouped.items():
        center = len(active_threads) - 0.5 - active_threads.index(thread)
        grp = sorted(grp, key=sort_key_fn)
        cumsum = 0.0
        for row in grp:
            p = row["proportion"]
            if trace_side == "lower":
                top = center - padding - cumsum * half
                bottom = center - padding - (cumsum + p) * half
            else:
                bottom = center + padding + cumsum * half
                top = center + padding + (cumsum + p) * half

            out["left"].append(left)
            out["right"].append(right)
            out["top"].append(top)
            out["bottom"].append(bottom)
            out["color"].append(color_map.get(row["function"], "#999999"))
            out["function"].append(row["function"])
            out["thread"].append(thread)
            out["proportion"].append(p)
            out["exclusive_s"].append(row["exclusive_s"])
            cumsum += p

    return out
