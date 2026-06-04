from __future__ import annotations

import numpy as np

from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.ranges import Range1d
from bokeh.plotting import figure

from data_model import QuantaQuery, as_token_key
from trace_session import TraceSession, QuantaQuery
from adapters.quanta_adapter import (
    empty_quanta_source,
    build_color_map,
    quanta_bundle_to_bokeh_source,
)
from utils import timed


class QuantaComparisonView:
    def __init__(
        self,
        t1: TraceSession,
        t2: TraceSession,
        *,
        width: int = 1400,
        height: int = 900,
    ) -> None:
        self.t1 = t1
        self.t2 = t2
        self.width = width
        self.height = height

        self.fig = None
        self.source1 = ColumnDataSource(empty_quanta_source())
        self.source2 = ColumnDataSource(empty_quanta_source())

    def build(self):
        all_threads = self.get_all_thread_names()
        fig = figure(
            width           = self.width,
            height          = self.height,
            x_range         = Range1d(0, 1),
            y_range         = list(reversed(all_threads)),  # type: ignore
            tools           = ["box_zoom", "xwheel_pan", "xbox_zoom", "reset", "undo", "redo", "save"],
            active_drag     = "box_zoom",
            output_backend  = "webgl",
            title           = "Quanta comparison",
            x_axis_label    = "Time (ms)",
        )
        fig.add_tools(HoverTool(tooltips=[
            ("thread", "@thread"),
            ("token", "@token_name"),
            ("token_key", "@token_key"),
            ("proportion", "@proportion{0.000}"),
            ("exclusive_s", "@exclusive_s{0.000000} s"),
        ]))
        fig.quad(
            left="left", right="right", top="top", bottom="bottom",
            color="color", line_color=None, fill_alpha=0.90,
            source=self.source1, name="quanta_t1",
        )
        fig.quad(
            left="left", right="right", top="top", bottom="bottom",
            color="color", line_color=None, fill_alpha=0.90,
            source=self.source2, name="quanta_t2",
        )
        self.fig = fig
        return fig

    def update(
        self,
        *,
        active_thread_names: list[str],
        n_quanta: int,
        mode: str,
        stack_order: str,
    ) -> dict:
        with timed("setup metadata"):
            common_names = self.get_shared_thread_names(active_thread_names)
            top_tokens = set(self.t1.summary.top_tokens) | set(self.t2.summary.top_tokens)

            thread_ids_1 = tuple(
                self.t1.meta.thread_name_to_id[name]
                for name in common_names
                if name in self.t1.meta.thread_name_to_id
            )
            thread_ids_2 = tuple(
                self.t2.meta.thread_name_to_id[name]
                for name in common_names
                if name in self.t2.meta.thread_name_to_id
            )

            start_ns = min(self.t1.meta.start_ns, self.t2.meta.start_ns)
            end_ns = max(self.t1.meta.end_ns, self.t2.meta.end_ns)
            edges = np.linspace(start_ns, end_ns, int(n_quanta) + 1, dtype=np.int64)

        with timed("compute_quanta q1"):
            q1 = self.t1.query_quanta(
                QuantaQuery(
                    thread_ids      = thread_ids_1,
                    bin_edges_ns    = tuple(int(x) for x in edges),
                    fidelity        = mode,  # type: ignore
                    top_k           = 16,
                )
            )
        with timed("compute_quanta q2"):
            q2 = self.t2.query_quanta(
                QuantaQuery(
                    thread_ids      = thread_ids_2,
                    bin_edges_ns    = tuple(int(x) for x in edges),
                    fidelity        = mode,  # type: ignore
                    top_k           = 16,
                )
            )

        with timed("setup color_map"):
            all_token_keys = set(top_tokens)
            all_token_keys |= self.bundle_token_keys(q1)
            all_token_keys |= self.bundle_token_keys(q2)
            color_map = build_color_map(all_token_keys)

        with timed("quanta_bundle_to_bokeh_source"):
            src1 = quanta_bundle_to_bokeh_source(
                q1,
                meta            = self.t1.meta,
                active_threads  = common_names,
                trace_side      = "lower",
                color_map       = color_map,
                stack_order     = stack_order,
            )
            src2 = quanta_bundle_to_bokeh_source(
                q2,
                meta            = self.t2.meta,
                active_threads  = common_names,
                trace_side      = "upper",
                color_map       = color_map,
                stack_order     = stack_order,
            )

        with timed("setup data"):
            self.source1.data = src1
            self.source2.data = src2

            if self.fig is not None:
                self.fig.y_range.factors = list(reversed(common_names))  # type: ignore
                if len(edges) >= 2:
                    self.fig.x_range.start = int(edges[0]) / 1e6  # type: ignore
                    self.fig.x_range.end = int(edges[-1]) / 1e6  # type: ignore

            return {
                "threads": common_names,
                "n_bins": int(n_quanta),
                "mode": mode,
                "stack_order": stack_order,
                "color_map": color_map,
            }

    def bundle_token_keys(self, bundle) -> set[str]:
        return {
            as_token_key(int(tt), int(tid))
            for tt, tid in zip(bundle.token_type, bundle.token_id)
        }

    def get_all_thread_names(self) -> list[str]:
        names = set(map(str, self.t1.meta.thread_names)) | set(map(str, self.t2.meta.thread_names))
        return sorted(names)

    def get_shared_thread_names(self, requested: list[str]) -> list[str]:
        available = set(self.get_all_thread_names())
        chosen = [name for name in requested if name in available]
        return chosen or self.get_all_thread_names()
