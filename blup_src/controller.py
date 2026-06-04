from __future__ import annotations

from bokeh.layouts import column, row
from bokeh.models.widgets.inputs import MultiSelect, Select, Spinner
from bokeh.models.widgets.markups import Div

from state import AppState
from trace_session import TraceSession
from views.quanta_comparison_view import QuantaComparisonView
from views.inspector_view import InspectorView
from utils import timed


class AppController:
    def __init__(self, t1: TraceSession, t2: TraceSession):
        self.t1 = t1
        self.t2 = t2

        self.state = self._initial_state()

        self.time_view = QuantaComparisonView(t1, t2, width=1350, height=950)
        self.inspector = InspectorView(t1, t2, width=360)

        self.thread_select: MultiSelect | None = None
        self.n_quanta_spinner: Spinner | None = None
        self.snapshot_mode_select: Select | None = None
        self.stack_order_select: Select | None = None

        self.root = None

    def build(self):
        with timed("build.title"):
            title = Div(text="<h2 style='margin:0'>Blup</h2>")

        with timed("build.widgets"):
            self.thread_select = MultiSelect(
                title   = "Threads",
                value   = list(self.state.active_threads),
                options = self._all_thread_names(),  # type: ignore
                size    = 12,
                width   = 260,
            )
            self.n_quanta_spinner = Spinner(
                title   = "Quanta bins",
                low     = 10,
                high    = 5000,
                step    = 10,
                value   = self.state.n_quanta,
                width   = 130,
            )
            self.snapshot_mode_select = Select(
                title   = "Snapshot mode",
                value   = self.state.quanta_mode,
                options = ["fast"],
                width   = 120,
            )
            self.stack_order_select = Select(
                title   = "Stack order",
                value   = self.state.quanta_stack_order,
                options = ["global", "local"],
                width   = 120,
            )

        with timed("build.callbacks"):
            self.thread_select.on_change("value", self._on_threads_changed)
            self.n_quanta_spinner.on_change("value", self._on_n_quanta_changed)
            self.snapshot_mode_select.on_change("value", self._on_snapshot_mode_changed)
            self.stack_order_select.on_change("value", self._on_stack_order_changed)

        with timed("build.layout"):
            controls = row(
                title,
                self.thread_select,
                self.n_quanta_spinner,
                self.snapshot_mode_select,
                self.stack_order_select,
                sizing_mode="stretch_width",
            )
            main = row(
                self.time_view.build(),
                self.inspector.build(),
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
        with timed("time_view.update"):
            self.time_view.update(
                active_thread_names = list(self.state.active_threads),
                n_quanta            = self.state.n_quanta,
                mode                = self.state.quanta_mode,
                stack_order         = self.state.quanta_stack_order,
            )
        with timed("inspector.update"):
            self.inspector.update(
                active_threads      = list(self.state.active_threads),
                n_quanta            = self.state.n_quanta,
                mode                = self.state.quanta_mode,
                stack_order         = self.state.quanta_stack_order,
            )

    def _initial_state(self) -> AppState:
        names = self._all_thread_names()
        return AppState(active_threads=tuple(names))

    def _all_thread_names(self) -> list[str]:
        names = set(map(str, self.t1.meta.thread_names)) | set(map(str, self.t2.meta.thread_names))
        return sorted(names)

    def _on_threads_changed(self, attr, old, new):
        chosen = tuple(new) if new else tuple(self._all_thread_names())
        self.state.active_threads = chosen
        self.refresh()

    def _on_n_quanta_changed(self, attr, old, new):
        if new is None:
            return
        self.state.n_quanta = int(new)
        self.refresh()

    def _on_snapshot_mode_changed(self, attr, old, new):
        self.state.quanta_mode = new
        self.refresh()

    def _on_stack_order_changed(self, attr, old, new):
        self.state.quanta_stack_order = new
        self.refresh()
