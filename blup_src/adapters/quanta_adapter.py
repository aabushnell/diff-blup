from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from data_model import QuantaBundle, TraceInfo, as_token_key
from utils import timed


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

    with timed("qbbs.build_thread_centers"):
        thread_centers = {
            name: len(active_threads) - 0.5 - i
            for i, name in enumerate(active_threads)
        }

    with timed("qbbs.build_rows"):
        start_ns_arr = bundle.start_ns
        end_ns_arr = bundle.end_ns
        thread_id_arr = bundle.thread_id
        token_type_arr = bundle.token_type
        token_id_arr = bundle.token_id
        proportion_arr = bundle.proportion
        excl_ns_arr = bundle.excl_ns

        lefts: list[float] = []
        rights: list[float] = []
        threads: list[str] = []
        token_keys: list[str] = []
        token_names: list[str] = []
        proportions: list[float] = []
        exclusive_ss: list[float] = []

        for i in range(len(start_ns_arr)):
            tid = int(thread_id_arr[i])
            thread_name = meta.thread_id_to_name.get(tid)
            if thread_name is None or thread_name not in thread_centers:
                continue

            token_type = int(token_type_arr[i])
            token_id = int(token_id_arr[i])
            token_key = as_token_key(token_type, token_id)
            token_name = meta.token_key_to_name.get(token_key, token_key)

            lefts.append(int(start_ns_arr[i]) / 1e6)
            rights.append(int(end_ns_arr[i]) / 1e6)
            threads.append(thread_name)
            token_keys.append(token_key)
            token_names.append(token_name)
            proportions.append(float(proportion_arr[i]))
            exclusive_ss.append(int(excl_ns_arr[i]) / 1e9)

    if not lefts:
        return empty_quanta_source()

    with timed("qbbs.build_rank"):
        if stack_order == "global":
            totals: dict[str, float] = defaultdict(float)
            for token_key, exclusive_s in zip(token_keys, exclusive_ss):
                totals[token_key] += exclusive_s

            rank = {
                token_key: i
                for i, (token_key, _) in enumerate(
                    sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
                )
            }

            sort_key_fn = lambda row: rank.get(row[0], 10**9)
        elif stack_order == "local":
            sort_key_fn = lambda row: (-row[2], row[0])
        else:
            raise ValueError(f"invalid stack_order: {stack_order!r}")

    half = 0.45
    padding = 0.02

    with timed("qbbs.group_rows"):
        grouped: dict[tuple[str, float, float], list[tuple[str, str, float, float]]] = defaultdict(list)

        for left, right, thread, token_key, token_name, proportion, exclusive_s in zip(
            lefts,
            rights,
            threads,
            token_keys,
            token_names,
            proportions,
            exclusive_ss,
        ):
            grouped[(thread, left, right)].append(
                (token_key, token_name, proportion, exclusive_s)
            )

    with timed("qbbs.emit_output"):
        out = empty_quanta_source()

        out_left = out["left"]
        out_right = out["right"]
        out_top = out["top"]
        out_bottom = out["bottom"]
        out_color = out["color"]
        out_token_key = out["token_key"]
        out_token_name = out["token_name"]
        out_thread = out["thread"]
        out_proportion = out["proportion"]
        out_exclusive_s = out["exclusive_s"]

        for (thread, left, right), grp in grouped.items():
            center = thread_centers[thread]
            grp = sorted(grp, key=sort_key_fn)

            cumsum = 0.0
            for token_key, token_name, proportion, exclusive_s in grp:
                if trace_side == "lower":
                    top = center - padding - cumsum * half
                    bottom = center - padding - (cumsum + proportion) * half
                else:
                    bottom = center + padding + cumsum * half
                    top = center + padding + (cumsum + proportion) * half

                out_left.append(left)
                out_right.append(right)
                out_top.append(top)
                out_bottom.append(bottom)
                out_color.append(color_map.get(token_key, "#999999"))
                out_token_key.append(token_key)
                out_token_name.append(token_name)
                out_thread.append(thread)
                out_proportion.append(proportion)
                out_exclusive_s.append(exclusive_s)

                cumsum += proportion

    return out

