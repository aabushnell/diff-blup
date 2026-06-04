from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from data_model import QuantaBundle, TraceInfo, as_token_key


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
        token_key=[],
        token_name=[],
        thread=[],
        proportion=[],
        exclusive_s=[],
    )

def build_color_map(token_keys: Iterable[str]) -> dict[str, str]:
    keys = sorted(set(str(k) for k in token_keys))
    return {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(keys)}

def quanta_bundle_to_bokeh_source(
    bundle: QuantaBundle,
    *,
    meta: TraceInfo,
    active_threads: list[str],
    trace_side: str,
    color_map: dict[str, str],
    stack_order: str = "global",
) -> dict:
    if len(bundle.start_ns) == 0:
        return empty_quanta_source()

    if trace_side not in {"lower", "upper"}:
        raise ValueError(f"invalid trace_side: {trace_side!r}")

    thread_centers = {
        name: len(active_threads) - 0.5 - i
        for i, name in enumerate(active_threads)
    }

    rows = []
    for i in range(len(bundle.start_ns)):
        tid = int(bundle.thread_id[i])
        thread_name = meta.thread_id_to_name.get(tid)
        if thread_name is None or thread_name not in thread_centers:
            continue

        token_type = int(bundle.token_type[i])
        token_id = int(bundle.token_id[i])
        token_key = as_token_key(token_type, token_id)
        token_name = meta.token_key_to_name.get(token_key, token_key)

        rows.append({
            "left": int(bundle.start_ns[i]) / 1e6,
            "right": int(bundle.end_ns[i]) / 1e6,
            "thread": thread_name,
            "token_key": token_key,
            "token_name": token_name,
            "proportion": float(bundle.proportion[i]),
            "exclusive_s": int(bundle.excl_ns[i]) / 1e9,
        })

    if not rows:
        return empty_quanta_source()

    if stack_order == "global":
        totals = defaultdict(float)
        for r in rows:
            totals[r["token_key"]] += r["exclusive_s"]
        rank = {
            token_key: i
            for i, (token_key, _) in enumerate(
                sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
            )
        }
        sort_key_fn = lambda row: rank.get(row["token_key"], 10**9)
    elif stack_order == "local":
        sort_key_fn = lambda row: (-row["proportion"], row["token_key"])
    else:
        raise ValueError(f"invalid stack_order: {stack_order!r}")

    half = 0.45
    padding = 0.02
    grouped = defaultdict(list)

    for r in rows:
        key = (r["thread"], r["left"], r["right"])
        grouped[key].append(r)

    out = empty_quanta_source()
    for (thread, left, right), grp in grouped.items():
        center = thread_centers[thread]
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
            out["color"].append(color_map.get(row["token_key"], "#999999"))
            out["token_key"].append(row["token_key"])
            out["token_name"].append(row["token_name"])
            out["thread"].append(thread)
            out["proportion"].append(p)
            out["exclusive_s"].append(row["exclusive_s"])

            cumsum += p

    return out
