from __future__ import annotations

import re
import math
from dataclasses import dataclass

from bokeh.events import RangesUpdate
from bokeh.layouts import column, row
from bokeh.models.widgets.inputs import MultiSelect, Select, Spinner
from bokeh.models.widgets.markups import Div

from data_model import CATEGORY_TOKEN_TYPE
from state import AppState
from trace_session import TraceSession
from adapters.summary_adapter import SequenceSummaryDiffAdapter, SequenceSummaryDiffRow, format_duration_ns
from views.quanta_comparison_view import QuantaComparisonView
from views.inspector_view import InspectorView
from views.summary_diff_view import SequenceSummaryDiffView
from utils import timed

THREAD_RE = re.compile(r"^P#(\d+)T#(\d+)$")

def thread_sort_key(name: str):
    m = THREAD_RE.match(name)
    if m:
        return (0, int(m.group(1)), int(m.group(2)), name)
    nums = tuple(int(x) for x in re.findall(r"\d+", name))
    if nums:
        return (1, *nums, name)
    return (2, name)

class AppController:
    def __init__(self, t1: TraceSession, t2: TraceSession):
        self.t1 = t1
        self.t2 = t2

        self.install_merged_category_namespace()

        self.state = self.initial_state()

        self.time_view = QuantaComparisonView(t1, t2, width=1350, height=950)
        self.time_view.on_token_selected = self.on_quanta_token_selected  # type: ignore
        self.inspector = InspectorView(t1, t2, width=360)
        self.summary_view = SequenceSummaryDiffView(width=750, height=400)
        self.summary_adapter = SequenceSummaryDiffAdapter(t1, t2)
        
        self.sequence_token_select: Select | None = None
        self.selected_token: tuple[int, int] | None = None
        self._sequence_rows = ()

        self.thread_select: MultiSelect | None = None
        self.n_quanta_spinner: Spinner | None = None
        self.token_mode_select: Select | None = None
        self.snapshot_mode_select: Select | None = None
        self.stack_order_select: Select | None = None

        self.root = None

        self._range_refresh_scheduled = False
        self._ignore_range_callbacks = False
        self._ignore_sequence_callbacks = False

    def build(self):
        with timed("build.title"):
            title = Div(text="<h2 style='margin:0'>Blup</h2>")

        with timed("build.widgets"):
            self.thread_select = MultiSelect(
                title   = "Threads",
                value   = list(self.state.active_threads),
                options = self.all_thread_names(),  # type: ignore
                size    = 12,
                width   = 260,
            )
            self.n_quanta_spinner = Spinner(
                title   = "Quanta bins",
                low     = 1,
                high    = 500,
                step    = 1,
                value   = self.state.n_quanta,
                width   = 130,
            )
            self.token_mode_select = Select(
                title="Token view",
                value=self.state.token_mode,
                options=["raw", "named"],
                width=120,
            )
            self.snapshot_mode_select = Select(
                title   = "Snapshot mode",
                value   = self.state.quanta_mode,
                options = ["fast", "balanced", "exact"],
                width   = 120,
            )
            self.stack_order_select = Select(
                title   = "Stack order",
                value   = self.state.quanta_stack_order,
                options = ["global", "local"],
                width   = 120,
            )
            self.sequence_token_select = Select(
                title   = "Sequence",
                value   = "",
                options = [],
                width   = 320,
            )

        with timed("build.callbacks"):
            self.thread_select.on_change("value", self.on_threads_changed)
            self.n_quanta_spinner.on_change("value", self.on_n_quanta_changed)
            self.token_mode_select.on_change("value", self.on_token_mode_changed)
            self.snapshot_mode_select.on_change("value", self.on_snapshot_mode_changed)
            self.stack_order_select.on_change("value", self.on_stack_order_changed)
            self.sequence_token_select.on_change("value", self.on_sequence_token_changed)

        with timed("build.layout"):
            controls = row(
                title,
                self.thread_select,
                self.n_quanta_spinner,
                self.token_mode_select,
                self.snapshot_mode_select,
                self.stack_order_select,
                self.sequence_token_select,
                sizing_mode="stretch_width",
            )
            time_fig = self.time_view.build()
            self.bind_time_range_callbacks()
            self.bind_quanta_selection_callbacks()
            main = row(
                time_fig,
                # self.inspector.build(),
                self.summary_view.build(),
                self.time_view._benchmark_trigger,
                sizing_mode="stretch_width",
            )
            self.root = column(
                controls,
                main,
                sizing_mode="stretch_width",
            )

        with timed("build.refresh"):
            self.refresh()
        return self.root

    def refresh(self) -> None:
        self._ignore_range_callbacks = True
        query_token_mode = "category" if self.state.token_mode == "named" else "raw"
        try:
            with timed("time_view.update"):
                self.time_view.update(
                    active_thread_names = list(self.state.active_threads),
                    n_quanta            = self.state.n_quanta,
                    mode                = self.state.quanta_mode,
                    token_mode          = query_token_mode,
                    stack_order         = self.state.quanta_stack_order,
                    window_t0_ns        = self.state.quanta_window_t0_ns,
                    window_t1_ns        = self.state.quanta_window_t1_ns,
                )
            with timed("inspector.update"):
                self.inspector.update(
                    active_threads      = list(self.state.active_threads),
                    n_quanta            = self.state.n_quanta,
                    mode                = self.state.quanta_mode,
                    token_mode          = query_token_mode,
                    stack_order         = self.state.quanta_stack_order,
                )
            with timed("summary_view.update"):
                rows = self.summary_adapter.build_rows(
                    token_mode=query_token_mode,
                    fidelity="fast",
                    top_k=32,
                    active_thread_names=tuple(self.state.active_threads),
                )
                self._sequence_rows = rows
                self.refresh_sequence_token_select(rows)
                self.refresh_sequence_views()
        finally:
            self._ignore_range_callbacks = False

    def initial_state(self) -> AppState:
        names = self.all_thread_names()
        return AppState(active_threads=tuple(names))

    def all_thread_names(self) -> list[str]:
        names = set(map(str, self.t1.meta.thread_names)) | set(map(str, self.t2.meta.thread_names))
        return sorted(names, key=thread_sort_key)

    def bind_time_range_callbacks(self) -> None:
        fig = self.time_view.fig
        if fig is None:
            return
        fig.x_range.on_change("start", self.on_time_range_changed)  # type: ignore
        fig.x_range.on_change("end", self.on_time_range_changed)  # type: ignore

    def on_time_range_changed(self, attr, old, new) -> None:
        if self._ignore_range_callbacks:
            return
        if self._range_refresh_scheduled:
            return
        if self.time_view.doc is None:
            return
        self._range_refresh_scheduled = True
        self.time_view.doc.add_next_tick_callback(self.apply_time_range_change)

    def apply_time_range_change(self) -> None:
        self._range_refresh_scheduled = False

        fig = self.time_view.fig
        if fig is None:
            return

        start_ms = fig.x_range.start  # type: ignore
        end_ms = fig.x_range.end  # type: ignore
        if start_ms is None or end_ms is None:
            return
        if not math.isfinite(start_ms) or not math.isfinite(end_ms):
            return
        if end_ms <= start_ms:
            return

        full_t0_ns = self.time_view.full_start_ns
        full_t1_ns = self.time_view.full_end_ns
        full_t0_ms = full_t0_ns / 1e6
        full_t1_ms = full_t1_ns / 1e6

        eps_ms = 1e-9
        if abs(start_ms - full_t0_ms) <= eps_ms and abs(end_ms - full_t1_ms) <= eps_ms:
            changed = (
                self.state.quanta_window_t0_ns is not None
                or self.state.quanta_window_t1_ns is not None
                or self.state.quanta_zoom_level != 0
            )
            self.state.quanta_window_t0_ns = None
            self.state.quanta_window_t1_ns = None
            self.state.quanta_zoom_level = 0

            if changed:
                self.refresh()
            return

        raw_t0_ns = max(full_t0_ns, int(start_ms * 1e6))
        raw_t1_ns = min(full_t1_ns, int(end_ms * 1e6))
        if raw_t1_ns <= raw_t0_ns:
            return

        zoom_level, snapped_t0_ns, snapped_t1_ns = self.snap_quanta_window(raw_t0_ns, raw_t1_ns)

        if (
            self.state.quanta_zoom_level == zoom_level
            and self.state.quanta_window_t0_ns == snapped_t0_ns
            and self.state.quanta_window_t1_ns == snapped_t1_ns
        ):
            return

        self.state.quanta_zoom_level = zoom_level
        self.state.quanta_window_t0_ns = snapped_t0_ns
        self.state.quanta_window_t1_ns = snapped_t1_ns

        self.refresh()

    def snap_quanta_window(self, raw_t0_ns: int, raw_t1_ns: int) -> tuple[int, int, int]:
        full_t0_ns = self.time_view.full_start_ns
        full_t1_ns = self.time_view.full_end_ns
        full_span = max(1, full_t1_ns - full_t0_ns)

        raw_t0_ns = max(full_t0_ns, min(raw_t0_ns, full_t1_ns))
        raw_t1_ns = max(full_t0_ns, min(raw_t1_ns, full_t1_ns))
        raw_span = max(1, raw_t1_ns - raw_t0_ns)

        min_span = max(1, full_span // 1024)
        raw_span = max(min_span, min(raw_span, full_span))

        zoom_level = max(0, int(round(math.log2(full_span / raw_span))))
        zoom_level = min(zoom_level, 10)

        snapped_span = max(min_span, full_span // (2 ** zoom_level))
        snapped_span = min(snapped_span, full_span)

        center = (raw_t0_ns + raw_t1_ns) // 2
        stride = max(1, snapped_span // 2)

        ideal_t0_ns = center - snapped_span // 2
        idx = round((ideal_t0_ns - full_t0_ns) / stride)
        snapped_t0_ns = full_t0_ns + idx * stride
        snapped_t0_ns = max(full_t0_ns, min(snapped_t0_ns, full_t1_ns - snapped_span))
        snapped_t1_ns = snapped_t0_ns + snapped_span

        return zoom_level, snapped_t0_ns, snapped_t1_ns

    def on_threads_changed(self, attr, old, new):
        chosen = tuple(new) if new else tuple(self.all_thread_names())
        self.state.active_threads = chosen
        self.refresh()

    def on_n_quanta_changed(self, attr, old, new):
        if new is None:
            return
        self.state.n_quanta = int(new)
        self.refresh()

    def on_token_mode_changed(self, attr, old, new):
        self.state.token_mode = new
        self.refresh()

    def on_snapshot_mode_changed(self, attr, old, new):
        self.state.quanta_mode = new
        self.refresh()

    def on_stack_order_changed(self, attr, old, new):
        self.state.quanta_stack_order = new
        self.refresh()

    def on_token_selected(self, token: tuple[int, int] | None) -> None:
        self.selected_token = token
        self.refresh()

    def token_to_select_value(self, token: tuple[int, int] | None) -> str:
        if token is None:
            return ""
        return f"{token[0]}:{token[1]}"

    def select_value_to_token(self, value: str) -> tuple[int, int] | None:
        if not value:
            return None
        a, b = value.split(":", 1)
        return (int(a), int(b))

    def build_sequence_token_options(self, rows) -> list[tuple[str, str]]:
        opts: list[tuple[str, str]] = [("", "(none)")]
        for row in rows:
            value = self.token_to_select_value((row.token_type, row.token_id))
            label = (
                f"#{row.contribution_rank} "
                f"{row.name} "
                f"({format_duration_ns(row.contribution_abs_ns)}, "
                f"{row.contribution_share_pct:.1f}%) "
                f"[{row.token_type}:{row.token_id}]"
            )
            opts.append((value, label))
        return opts

    def on_sequence_token_changed(self, attr, old, new):
        if self._ignore_sequence_callbacks:
            return
        self.selected_token = self.select_value_to_token(new)
        self.refresh_sequence_views()

    def refresh_sequence_token_select(self, rows) -> None:
        if self.sequence_token_select is None:
            return

        options = self.build_sequence_token_options(rows)
        valid_values = {value for value, _ in options}
        current_value = self.token_to_select_value(self.selected_token)

        if current_value not in valid_values:
            current_value = options[1][0] if len(options) > 1 else ""
            self.selected_token = self.select_value_to_token(current_value)

        self._ignore_sequence_callbacks = True
        try:
            self.sequence_token_select.options = options  # type: ignore
            if self.sequence_token_select.value != current_value:
                self.sequence_token_select.value = current_value
        finally:
            self._ignore_sequence_callbacks = False

    def refresh_sequence_views(self) -> None:
        row = next((r for r in self._sequence_rows if r.token == self.selected_token), None)

        t0_ns = self.state.quanta_window_t0_ns
        t1_ns = self.state.quanta_window_t1_ns
        if t0_ns is None or t1_ns is None:
            t0_ns = self.time_view.full_start_ns
            t1_ns = self.time_view.full_end_ns

        query_token_mode = "category" if self.state.token_mode == "named" else "raw"

        model = self.summary_adapter.build_display_model(
            row,
            active_thread_names=tuple(self.state.active_threads),
            token_mode=query_token_mode,
            t0_ns=t0_ns,
            t1_ns=t1_ns,
            histogram_bins=20,
        )
        self.summary_view.update(model)

    def on_quanta_token_selected(self, token: tuple[int, int] | None) -> None:
        self.selected_token = token

        if self.sequence_token_select is not None:
            value = self.token_to_select_value(token)
            self._ignore_sequence_callbacks = True
            try:
                if self.sequence_token_select.value != value:
                    self.sequence_token_select.value = value
            finally:
                self._ignore_sequence_callbacks = False

        self.refresh_sequence_views()

    def bind_quanta_selection_callbacks(self) -> None:
        source = getattr(self.time_view, "source", None)
        if source is None:
            return
        source.selected.on_change("indices", self.on_quanta_bar_indices_changed)

    def on_quanta_bar_indices_changed(self, attr, old, new) -> None:
        source = getattr(self.time_view, "source", None)
        if source is None:
            return

        indices = list(new or [])
        if not indices:
            return

        i = int(indices[0])
        data = source.data

        token_types = data.get("token_type")
        token_ids = data.get("token_id")
        if token_types is None or token_ids is None:
            return
        if i < 0 or i >= len(token_types) or i >= len(token_ids):
            return

        token = (int(token_types[i]), int(token_ids[i]))
        self.on_quanta_token_selected(token)

    def install_merged_category_namespace(self) -> None:
        names = sorted(
            set(str(name) for name in self.t1.meta.cat_key_to_name.values())
            | set(str(name) for name in self.t2.meta.cat_key_to_name.values())
        )

        name_to_cat_token = {
            name: (CATEGORY_TOKEN_TYPE, int(i))
            for i, name in enumerate(names)
        }

        self.t1.install_category_namespace(name_to_cat_token)
        self.t2.install_category_namespace(name_to_cat_token)
