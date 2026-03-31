from bokeh.models import ColumnDataSource, HoverTool, RangeTool, Range1d
from bokeh.plotting import figure
from bokeh.transform import dodge
from bokeh import events

import pandas as pd

from blup_trace import BlupTrace, prepare_display_df, prepare_quanta_display_df
from blup_utils import natural_keys, choose_palette, apply_top_bottom

def _empty_display_dict():
    return dict(
        start=[],
        finish=[],
        top=[],
        bottom=[],
        color=[],
        function=[],
        duration=[],
        thread=[],
        depth=[]
    )

def _empty_quanta_dict():
    return dict(
        left=[],
        right=[],
        top=[],
        bottom=[],
        color=[],
        function=[],
        thread=[],
        proportion=[],
        exclusive_s=[]
    )

class Timeline:

    GANTT   = "gantt"
    FLAME   = "flame"
    SUBTREE = "subtree"
    QUANTA  = "quanta"

    def __init__(self, t1: BlupTrace, t2: BlupTrace,
                 width: int, height: int):
        # set traces for comparison
        self.t1 = t1
        self.t2 = t2

        # set panel parameters
        self.width = width
        self.height = height
        self.mode = self.GANTT
        self.active_threads = sorted(
            set(t1.threads) | set(t2.threads), key=natural_keys
        )
        self.fig = None
        self.minimap_fig = None
        self._func_filter = None
        self._func_filter_mode = "off"
        self._selected_func: str | None = None
        self._stack_mode: str = "diverge"
        self._saved_x_range: tuple[float, float] | None = None
        self._duration_filter: tuple[str, float, float] | None = None
        self._n_quanta: int = 200
        self._quanta_stack_order: str = "global"

        # process trace data
        df1, df2, self.all_threads, self.all_functions = self._prepare()

        # set data sources
        self.source1 = ColumnDataSource(df1)
        self.source2 = ColumnDataSource(df2)
        self._subtree_source1 = ColumnDataSource(_empty_display_dict())  # type: ignore
        self._subtree_source2 = ColumnDataSource(_empty_display_dict())  # type: ignore
        self._quanta_source1  = ColumnDataSource(_empty_quanta_dict())   # type: ignore
        self._quanta_source2  = ColumnDataSource(_empty_quanta_dict())   # type: ignore

        # set color palette
        self._used_palette = choose_palette(self.all_functions)

    def build(self, minimap_height: int = 50) -> tuple:
        self.fig = self._make_figure()
        self._add_gantt_glyphs(self.fig)
        self.fig.legend.click_policy = "hide"
        self._fit_x_range_to_data()
        self.minimap_fig = self._make_minimap(minimap_height)
        return self.fig, self.minimap_fig

    def set_width(self, width: int):
        self.width = width
        if self.fig is not None:
            self.fig.width = width
        if self.minimap_fig is not None:
            self.minimap_fig.width = width

    def set_active_threads(self, threads: list[str]):
        self.active_threads = threads
        self._refresh_sources()
        if self.fig is not None:
            self.fig.y_range.factors = list(reversed(self.active_threads)) # type: ignore

    def set_function_filter(self, func: str | None, mode: str = "off"):
        self._func_filter      = func
        self._func_filter_mode = mode
        self._refresh_sources()

    def set_selected_function(self, func: str):
        self._selected_func = func
        if self.mode == self.SUBTREE:
            self._refresh_subtree_sources()

    def set_duration_filter(self, func: str, low_s: float, high_s: float):
        self._duration_filter = (func, low_s, high_s)
        self._refresh_sources()

    def clear_duration_filter(self):
        if self._duration_filter is None:
            return
        self._duration_filter = None
        self._refresh_sources()

    def set_n_quanta(self, n: int):
        self._n_quanta = n
        if self.mode == self.QUANTA:
            self._refresh_quanta_sources()

    def set_quanta_stack_order(self, order: str):
        assert order in ("global", "local")
        if order == self._quanta_stack_order:
            return
        self._quanta_stack_order = order
        if self.mode == self.QUANTA:
            self._refresh_quanta_sources()

    def set_mode(self, mode: str):
        assert mode in (self.GANTT, self.FLAME, self.SUBTREE, self.QUANTA)
        if mode == self.mode:
            return

        leaving_quanta  = self.mode == self.QUANTA
        leaving_subtree = self.mode == self.SUBTREE

        if leaving_quanta:
            self._quanta_source1.data = _empty_quanta_dict()   # type: ignore
            self._quanta_source2.data = _empty_quanta_dict()   # type: ignore
        if leaving_subtree:
            self._subtree_source1.data = _empty_display_dict() # type: ignore
            self._subtree_source2.data = _empty_display_dict() # type: ignore

        if mode == self.SUBTREE:
            # save current view
            if self.fig is not None:
                self._saved_x_range = (
                    self.fig.x_range.start,  # type: ignore
                    self.fig.x_range.end,    # type: ignore
                )
            self._ensure_subtree_glyphs()
            self._refresh_subtree_sources()

        elif mode == self.QUANTA:
            if self.fig is not None:
                self._saved_x_range = (
                    self.fig.x_range.start,  # type: ignore
                    self.fig.x_range.end,    # type: ignore
                )
            self._ensure_quanta_glyphs()
            self._refresh_quanta_sources()

        elif leaving_subtree or leaving_quanta:
            # restore full-trace x range
            if self.fig is not None and self._saved_x_range is not None:
                self.fig.x_range.start = self._saved_x_range[0]  # type: ignore
                self.fig.x_range.end   = self._saved_x_range[1]  # type: ignore

        if mode == self.FLAME:
            self._ensure_flame_glyphs()

        self.mode = mode
        self._sync_glyph_visibility()

    def set_stack_mode(self, mode: str):
        assert mode in ("diverge", "converge")
        if mode == self._stack_mode:
            return
        self._stack_mode = mode
        self._refresh_sources()
        if self.mode == self.SUBTREE:
            self._refresh_subtree_sources()

    def toggle_legend(self):
        if self.fig is not None:
            self.fig.legend.visible = not self.fig.legend.visible  # type: ignore

    def _prepare(self):
        return prepare_display_df(
            self.t1, self.t2,
            active_threads     = self.active_threads,
            func_filter        = self._func_filter,
            func_filter_mode   = self._func_filter_mode,
            stack_mode         = self._stack_mode,
            duration_filter    = self._duration_filter,
        )

    def _refresh_sources(self):
        df1, df2, _, _ = self._prepare()
        self.source1.data = ColumnDataSource.from_df(df1)
        self.source2.data = ColumnDataSource.from_df(df2)
        if self.mode == self.SUBTREE:
            self._refresh_subtree_sources()
        elif self.mode == self.QUANTA:
            self._refresh_quanta_sources()

    def _fit_x_range_to_data(self):
        if self.fig is None:
            return
        starts  = []
        ends    = []
        for src in (self.source1, self.source2):
            if len(src.data.get("start", [])) > 0:
                starts.append(min(src.data["start"]))
                ends.append(max(src.data["finish"]))
        if starts:
            self.fig.x_range.start = min(starts)  # type: ignore
            self.fig.x_range.end   = max(ends)    # type: ignore

    def _make_figure(self) -> figure:
        g = figure(
            width=self.width, height=self.height,
            output_backend="webgl",
            tools=["box_zoom", "xwheel_pan", "xbox_zoom", "reset", "undo", "redo",
                   "ycrosshair"],
            active_drag="box_zoom",
            x_range=Range1d(start=0, end=1),
            y_range=list(reversed(self.active_threads)),  # type: ignore
            x_axis_type="datetime",
        )
        g.add_tools(HoverTool(tooltips=[
            ("function", "@function"), ("start", "@start"),
            ("finish",   "@finish"),   ("duration", "@duration"),
        ]))
        g.on_event(events.RangesUpdate, self._on_ranges_update)
        return g

    def _add_gantt_glyphs(self, g):
        g.hbar(y=dodge("thread", -0.12, range=g.y_range),
               left="start", right="finish", height=0.22,
               color="color", legend_field="function",
               fill_alpha="alpha", line_alpha="alpha",
               source=self.source1, name="gantt_t1")
        g.hbar(y=dodge("thread", +0.12, range=g.y_range),
               left="start", right="finish", height=0.22,
               color="color", legend_field="function",
               fill_alpha="alpha", line_alpha="alpha",
               source=self.source2, name="gantt_t2")

    def _ensure_flame_glyphs(self):
        if self.fig.select_one({"name": "flame_t1"}) is None: # type: ignore
            self.fig.quad(left="start", right="finish", top="top", bottom="bottom",  # type: ignore
                          color="color", legend_field="function",
                          fill_alpha="alpha", line_alpha="alpha",
                          source=self.source1, name="flame_t1")
            self.fig.quad(left="start", right="finish", top="top", bottom="bottom",  # type: ignore
                          color="color", legend_field="function",
                          fill_alpha="alpha", line_alpha="alpha",
                          source=self.source2, name="flame_t2")

    def _ensure_subtree_glyphs(self):
        if self.fig.select_one({"name": "subtree_t1"}) is None:  # type: ignore
            for src, name in [(self._subtree_source1, "subtree_t1"),
                            (self._subtree_source2, "subtree_t2")]:
                self.fig.quad(left="start", right="finish", top="top", bottom="bottom",  # type: ignore
                            color="color", legend_field="function",
                            fill_alpha="alpha", line_alpha="alpha",
                            source=src, name=name)

    def _ensure_quanta_glyphs(self):
        if self.fig.select_one({"name": "quanta_t1"}) is None:  # type: ignore
            for src, name in [(self._quanta_source1, "quanta_t1"),
                              (self._quanta_source2, "quanta_t2")]:
                self.fig.quad(  # type: ignore
                    left="left", right="right", top="top", bottom="bottom",
                    color="color", legend_field="function",
                    fill_alpha=0.85, line_color=None,
                    source=src, name=name)

    def _sync_glyph_visibility(self):
        show_gantt   = self.mode == self.GANTT
        show_flame   = self.mode == self.FLAME
        show_subtree = self.mode == self.SUBTREE
        for name, visible in [
            ("gantt_t1",   show_gantt),   ("gantt_t2",   show_gantt),
            ("flame_t1",   show_flame),   ("flame_t2",   show_flame),
            ("subtree_t1", show_subtree), ("subtree_t2", show_subtree),
        ]:
            r = self.fig.select_one({"name": name})  # type: ignore
            if r is not None:
                r.visible = visible  # type: ignore

    def _make_minimap(self, height: int) -> figure:
        mini = figure(
            height=height, width=self.width,
            y_range=self.fig.y_range,  # type: ignore
            x_axis_type="datetime", y_axis_type=None,
            tools="", toolbar_location=None,
            background_fill_color="#efefef",
        )
        rt = RangeTool(x_range=self.fig.x_range)  # type: ignore
        rt.overlay.fill_color = "navy"
        rt.overlay.fill_alpha = 0.2
        mini.add_tools(rt)
        return mini

    def _refresh_quanta_sources(self):
        df1, df2 = prepare_quanta_display_df(
            self.t1, self.t2, self.active_threads, 
            self._n_quanta, self._quanta_stack_order
        )
        self._quanta_source1.data = (  # type: ignore
            ColumnDataSource.from_df(df1) if not df1.empty else _empty_quanta_dict()
        )
        self._quanta_source2.data = (  # type: ignore
            ColumnDataSource.from_df(df2) if not df2.empty else _empty_quanta_dict()
        )
        if self.fig is not None:
            lefts  = list(self._quanta_source1.data.get("left",  []))  # type: ignore
            rights = list(self._quanta_source1.data.get("right", []))  # type: ignore
            if lefts:
                self.fig.x_range.start = min(lefts)   # type: ignore
                self.fig.x_range.end   = max(rights)  # type: ignore

    def _build_subtree_trace_df(
        self,
        parts: list[pd.DataFrame],
        label: str,
        color_map: dict[str, str],
        depth_step: float,
        max_depth: int,
    ) -> pd.DataFrame:
        df: pd.DataFrame = pd.concat(parts, ignore_index=True)

        df["start"]    = pd.to_timedelta(df["start"],    unit="ns")
        df["finish"]   = pd.to_timedelta(df["finish"],   unit="ns")
        df["duration"] = pd.to_timedelta(df["duration"], unit="ns")

        df["color"]  = df["function"].map(color_map)
        df["center"] = df["thread"].apply(
            lambda t: len(self.active_threads) - 0.5 - self.active_threads.index(t)
        )

        apply_top_bottom(df, label, self._stack_mode, depth_step, max_depth)

        df["alpha"] = 1.0
        return df

    def _prepare_subtree_df(self) -> tuple[pd.DataFrame, pd.DataFrame, float]:
        func = self._selected_func
        if func is None:
            return pd.DataFrame(), pd.DataFrame(), 0.0

        color_map  = {f: self._used_palette[i % len(self._used_palette)]
                    for i, f in enumerate(self.all_functions)}
        depth_step = 0.1
        gap        = depth_step / 10.0
        results    = []
        all_parts  = []

        per_trace_parts    = []
        max_finish_ns: int = 0
        for trace_obj, label in [(self.t1, "Trace 1"), (self.t2, "Trace 2")]:
            parts = []
            for thread in self.active_threads:
                instances = trace_obj.get_call_instances(func, thread)
                if not instances:
                    continue
                sub = trace_obj.get_call_subtree(instances[0])

                root_start    = sub["start"].min()
                sub["start"]  = sub["start"]  - root_start
                sub["finish"] = sub["finish"] - root_start

                max_finish_ns = max(max_finish_ns, sub["finish"].max().value)

                sub["trace"]  = label
                parts.append(sub)
                all_parts.append(sub)
            per_trace_parts.append((parts, label))

        if all_parts:
            combined_depth = pd.concat(all_parts, ignore_index=True)
            max_depth = int(combined_depth["depth"].max())
        else:
            max_depth = 0

        for parts, label in per_trace_parts:
            if not parts:
                results.append(pd.DataFrame())
            else:
                results.append(
                    self._build_subtree_trace_df(parts, label, color_map, depth_step, max_depth)
                )

        return results[0], results[1], max_finish_ns / 1e6

    def _refresh_subtree_sources(self):
        df1, df2, x_end_ms = self._prepare_subtree_df()
        self._subtree_source1.data = (ColumnDataSource.from_df(df1)  # type: ignore
                                    if not df1.empty else _empty_display_dict())
        self._subtree_source2.data = (ColumnDataSource.from_df(df2)  # type: ignore
                                    if not df2.empty else _empty_display_dict())
        if self.fig is not None and x_end_ms > 0:
            self.fig.x_range.start = 0          # type: ignore
            self.fig.x_range.end   = x_end_ms   # type: ignore


    def _on_ranges_update(self, event):
        pass


