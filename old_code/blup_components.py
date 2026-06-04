from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from bokeh.layouts import column, gridplot, row as bk_row
from bokeh.models import ColumnDataSource, Div, HoverTool, RadioButtonGroup, Range1d
from bokeh.plotting import figure

from blup_trace import BlupTrace
from blup_utils import choose_palette, natural_keys


_PIE_SIZE   = 260
_BAR_WIDTH  = 320
_BAR_HEIGHT = 280
_EMPTY_MSG  = "<i style='color:#999;font-size:12px'>No subtree data available.</i>"

# Relative-difference helpers
def _rel_diff(v1: float, v2: float) -> float | None:
    """(v2 - v1) / v1 * 100.  Returns None when baseline is zero."""
    if v1 == 0:
        return None
    return (v2 - v1) / v1 * 100.0

def _fmt_rel(rel: float | None) -> str:
    if rel is None:
        return "<span style='color:#999'>N/A</span>"
    color = "#c0392b" if rel > 0 else "#27ae60" if rel < 0 else "#555"
    sign  = "+" if rel > 0 else ""
    return f"<span style='color:{color};font-weight:600'>{sign}{rel:.1f}%</span>"


class Components:
    """
    Components panel — per-thread breakdown of the selected function's subtree.

    Layout
    ------
    A bk_row of:
      • gridplot  — pie or bar charts (rows = threads, cols = traces for pie;
                                       rows = threads, 1 col for bar)
      • stats Div — relative differences between the two traces, per thread

    Computation modes
    -----------------
    direct    — proportion from immediate callees (depth == root + 1)
    exclusive — top-of-stack (exclusive) time per function
    """

    def __init__(self, t1: BlupTrace, t2: BlupTrace) -> None:
        self.t1 = t1
        self.t2 = t2
        self._mode: str = "direct"
        self._view: str = "pie"
        self._func: Optional[str] = None
        self._subtrees: list[tuple[str, str, pd.DataFrame]] = []

        self._container:   Optional[column]           = None # type: ignore
        self._mode_toggle: Optional[RadioButtonGroup] = None
        self._view_toggle: Optional[RadioButtonGroup] = None
        self._empty_div:   Optional[Div]              = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> column: # type: ignore
        self._mode_toggle = RadioButtonGroup(
            labels=["Direct Subroutines", "Top of Stack (Exclusive)"],
            active=0,
            width=340,
        )
        self._mode_toggle.on_change("active", self._on_mode_change)

        self._view_toggle = RadioButtonGroup(
            labels=["Pie", "Bar"],
            active=0,
            width=140,
        )
        self._view_toggle.on_change("active", self._on_view_change)

        title_div = Div(
            text="<b style='font-size:13px'>Components</b>",
            margin=(4, 0, 6, 0),
        )
        self._empty_div = Div(text=_EMPTY_MSG, margin=(12, 0, 0, 0))
        self._container = column(
            children=[self._empty_div],
            sizing_mode="stretch_width",
        )
        return column(
            title_div,
            bk_row(self._mode_toggle, self._view_toggle),
            self._container, # type: ignore
            sizing_mode="stretch_width",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        func: str,
        subtrees: list[tuple[str, str, pd.DataFrame]],
    ) -> None:
        self._func     = func
        self._subtrees = subtrees
        self._rebuild()

    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------

    def _on_mode_change(self, attr, old, new) -> None:
        self._mode = "direct" if new == 0 else "exclusive"
        self._rebuild()

    def _on_view_change(self, attr, old, new) -> None:
        self._view = "pie" if new == 0 else "bar"
        self._rebuild()

    # ------------------------------------------------------------------
    # Rebuild
    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        if self._container is None:
            return
        if not self._subtrees:
            self._container.children = [self._empty_div]  # type: ignore
            return

        # Global colour map — consistent across all charts and the stats panel
        all_funcs = sorted({
            f
            for _, _, df in self._subtrees
            for f in df["function"].unique()
        })
        palette  = choose_palette(all_funcs)
        colormap = {f: palette[i % len(palette)] for i, f in enumerate(all_funcs)}

        # Group by thread; insertion order preserves Trace 1 → Trace 2
        all_threads = sorted(
            {thread for _, thread, _ in self._subtrees}, key=natural_keys
        )
        by_thread: dict[str, list[tuple[str, pd.DataFrame]]] = {}
        for label, thread, df in self._subtrees:
            by_thread.setdefault(thread, []).append((label, df))

        # ----- charts -----
        grid_rows: list[list] = []
        for thread in all_threads:
            entries = by_thread.get(thread, [])
            if self._view == "pie":
                row_figs = []
                for label, df in entries:
                    props = self._proportions(df)
                    title = f"{label} — {thread}"
                    row_figs.append(
                        self._make_pie_fig(props, colormap, title)
                        if props else self._make_empty_pie(title)
                    )
                grid_rows.append(row_figs)
            else:
                trace_props = {label: self._proportions(df) for label, df in entries}
                grid_rows.append([
                    self._make_bar_fig(trace_props, colormap, all_funcs, thread)
                ])

        grid = gridplot(grid_rows, merge_tools=False, toolbar_location=None)

        # ----- stats -----
        stats_div = self._build_stats_div(by_thread, all_threads, colormap)

        self._container.children = [bk_row(grid, stats_div)]  # type: ignore

    # ------------------------------------------------------------------
    # Stats panel
    # ------------------------------------------------------------------

    def _build_stats_div(
        self,
        by_thread:   dict[str, list[tuple[str, pd.DataFrame]]],
        all_threads: list[str],
        colormap:    dict[str, str],
    ) -> Div:
        parts: list[str] = [
            "<div style='font-family:monospace;font-size:12px;"
            "min-width:260px;max-width:420px;padding:4px 10px'>"
        ]

        any_comparison = False

        for thread in all_threads:
            entries = by_thread.get(thread, [])
            if len(entries) < 2:
                continue

            label1, df1 = entries[0]
            label2, df2 = entries[1]
            any_comparison = True

            dur1 = self._get_root_duration(df1)
            dur2 = self._get_root_duration(df2)
            total_rel = _rel_diff(dur1, dur2)
            total_diff_ms = (dur2 - dur1) * 1e3
            total_diff_fmt = f"{'+'if total_diff_ms > 0 else ''}{total_diff_ms:.3f} ms"
            total_diff_color = "#c0392b" if total_diff_ms > 0 else "#27ae60" if total_diff_ms < 0 else "#555"

            parts.append(
                f"<div style='margin-bottom:14px;'>"
                f"<div style='font-size:13px;font-weight:700;margin-bottom:2px'>"
                f"{thread}</div>"
                f"<div style='color:#555;font-size:11px;margin-bottom:4px'>"
                f"{label1} → {label2}</div>"
            )
            parts.append(
                f"<div style='margin-bottom:6px;padding:4px 6px;"
                f"background:#f5f5f5;border-radius:3px'>"
                f"<b>Total ({self._func or '—'})</b>&nbsp;&nbsp;"
                f"{_fmt_rel(total_rel)}"
                f"&nbsp;<span style='color:{total_diff_color};font-weight:600'>({total_diff_fmt})</span>"
                f"<br><span style='color:#888;font-size:10px'>"
                f"T1 {dur1*1e3:.3f} ms &nbsp;→&nbsp; T2 {dur2*1e3:.3f} ms"
                f"</span></div>"
            )

            props1 = self._proportions(df1)
            props2 = self._proportions(df2)
            all_comp = sorted(set(props1) | set(props2))

            # contribution = (prop2 * dur2) - (prop1 * dur1)  [seconds]
            comp_rows: list[tuple[str, float, float, float | None, float]] = []
            for f in all_comp:
                p1 = props1.get(f, 0.0)
                p2 = props2.get(f, 0.0)
                contrib = (p2 * dur2) - (p1 * dur1)
                comp_rows.append((f, p1, p2, _rel_diff(p1, p2), contrib))

            # Order by |contribution| descending
            comp_rows.sort(key=lambda r: abs(r[4]), reverse=True)

            parts.append("<table style='width:100%;border-collapse:collapse'>")
            parts.append(
                "<tr style='color:#888;font-size:10px'>"
                "<th style='text-align:left;padding:2px 4px'>Function</th>"
                "<th style='text-align:right;padding:2px 4px'>T1</th>"
                "<th style='text-align:right;padding:2px 4px'>T2</th>"
                "<th style='text-align:right;padding:2px 4px'>Δ%</th>"
                "<th style='text-align:right;padding:2px 4px'>Contrib</th>"
                "</tr>"
            )
            for i, (fname, p1, p2, rel, contrib) in enumerate(comp_rows):
                dot_color = colormap.get(fname, "#888888")
                bg        = "background:#fafafa" if i % 2 == 0 else ""
                c_color   = "#c0392b" if contrib > 0 else "#27ae60" if contrib < 0 else "#555"
                c_sign    = "+" if contrib > 0 else ""
                c_ms      = contrib * 1e3
                c_fmt     = f"{c_sign}{c_ms:.3f} ms"
                parts.append(
                    f"<tr style='font-size:11px;{bg}'>"
                    f"<td style='padding:2px 4px;max-width:160px;"
                    f"overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>"
                    f"<span style='color:{dot_color}'>&#9632;</span>&nbsp;{fname}</td>"
                    f"<td style='text-align:right;padding:2px 4px;color:#555'>"
                    f"{p1*100:.1f}%</td>"
                    f"<td style='text-align:right;padding:2px 4px;color:#555'>"
                    f"{p2*100:.1f}%</td>"
                    f"<td style='text-align:right;padding:2px 4px'>"
                    f"{_fmt_rel(rel)}</td>"
                    f"<td style='text-align:right;padding:2px 4px;"
                    f"color:{c_color};font-weight:600'>{c_fmt}</td>"
                    f"</tr>"
                )
            parts.append("</table>")
            parts.append("</div>")

        if not any_comparison:
            parts.append(
                "<i style='color:#999'>No cross-trace comparison available "
                "(need matching threads in both traces).</i>"
            )

        parts.append("</div>")
        return Div(text="".join(parts), sizing_mode="fixed", width=420)

    # ------------------------------------------------------------------
    # Proportion computation
    # ------------------------------------------------------------------

    def _proportions(self, df: pd.DataFrame) -> dict[str, float]:
        return (
            self._compute_direct(df)
            if self._mode == "direct"
            else self._compute_exclusive(df)
        )

    @staticmethod
    def _to_seconds(v) -> float:
        if hasattr(v, "total_seconds"):
            return v.total_seconds()
        return float(v) / 1e9

    @classmethod
    def _get_root_duration(cls, df: pd.DataFrame) -> float:
        """Total duration of the root call(s) in this subtree (in seconds)."""
        if df.empty:
            return 0.0
        root_depth = int(df["depth"].min())
        root_rows  = df[df["depth"] == root_depth]
        return cls._to_seconds(root_rows["duration"].sum())

    @classmethod
    def _compute_direct(cls, df: pd.DataFrame) -> dict[str, float]:
        """Proportion of subtree duration from immediate callees, normalised."""
        if df.empty:
            return {}
        root_depth = int(df["depth"].min())
        direct     = df[df["depth"] == root_depth + 1]
        if direct.empty:
            return {}
        sums    = direct.groupby("function")["duration"].sum()
        total_s = cls._to_seconds(sums.sum())
        if total_s == 0:
            return {}
        return {f: cls._to_seconds(d) / total_s for f, d in sums.items()} # type: ignore

    @classmethod
    def _compute_exclusive(cls, df: pd.DataFrame) -> dict[str, float]:
        """Exclusive (top-of-stack) time per function, normalised.  O(n²)."""
        if df.empty:
            return {}
        result: dict[str, float] = defaultdict(float)
        for _, row in df.iterrows():
            children = df[
                (df["depth"]  == row["depth"] + 1) &
                (df["start"]  >= row["start"])      &
                (df["finish"] <= row["finish"])
            ]
            excl = (
                cls._to_seconds(row["duration"])
                - cls._to_seconds(children["duration"].sum())
            )
            result[row["function"]] += max(0.0, excl)
        total = sum(result.values())
        if total == 0:
            return {}
        return {f: v / total for f, v in result.items()}

    # ------------------------------------------------------------------
    # Figure construction — Pie
    # ------------------------------------------------------------------

    @staticmethod
    def _make_pie_fig(
        proportions: dict[str, float],
        colormap:    dict[str, str],
        title:       str,
        size:        int = _PIE_SIZE,
    ) -> figure:
        funcs  = list(proportions.keys())
        vals   = [proportions[f] for f in funcs]
        angles = np.array([v * 2 * math.pi for v in vals])
        starts = np.concatenate([[0.0], np.cumsum(angles[:-1])])
        ends   = starts + angles

        src = ColumnDataSource(dict(
            start_angle = starts.tolist(),
            end_angle   = ends.tolist(),
            color       = [colormap.get(f, "#aaaaaa") for f in funcs],
            function    = funcs,
            proportion  = [f"{v * 100:.1f}%" for v in vals],
        ))
        p = figure(
            width=size, height=size, title=title,
            x_range=Range1d(-1.25, 1.25),
            y_range=Range1d(-1.25, 1.25),
            tools="", toolbar_location=None,
        )
        p.title.text_font_size = "11px"  # type: ignore
        r = p.annular_wedge(
            x=0, y=0,
            inner_radius=0.35, outer_radius=0.9,
            start_angle="start_angle", end_angle="end_angle",
            color="color", line_color="white", line_width=1.5,
            source=src,
        )
        p.add_tools(HoverTool(
            renderers=[r],
            tooltips=[("Function", "@function"), ("Proportion", "@proportion")],
        ))
        p.axis.visible = False
        p.grid.visible = False
        return p

    @staticmethod
    def _make_empty_pie(title: str, size: int = _PIE_SIZE) -> figure:
        p = figure(
            width=size, height=size, title=title,
            x_range=Range1d(-1.25, 1.25),
            y_range=Range1d(-1.25, 1.25),
            tools="", toolbar_location=None,
        )
        p.title.text_font_size = "11px"  # type: ignore
        p.annular_wedge(
            x=0, y=0, inner_radius=0.35, outer_radius=0.9,
            start_angle=0, end_angle=2 * math.pi,
            color="#eeeeee", line_color="white", line_width=1.5,
        )
        p.axis.visible = False
        p.grid.visible = False
        return p

    # ------------------------------------------------------------------
    # Figure construction — Stacked Bar
    # ------------------------------------------------------------------

    @staticmethod
    def _make_bar_fig(
        trace_props: dict[str, dict[str, float]],
        colormap:    dict[str, str],
        all_funcs:   list[str],
        title:       str,
        width:       int = _BAR_WIDTH,
        height:      int = _BAR_HEIGHT,
    ) -> figure:
        trace_labels = list(trace_props.keys())
        active_funcs = [
            f for f in all_funcs
            if any(trace_props.get(t, {}).get(f, 0.0) > 0.0 for t in trace_labels)
        ]
        src_data: dict[str, list] = {"traces": trace_labels}
        for f in active_funcs:
            src_data[f] = [trace_props.get(t, {}).get(f, 0.0) for t in trace_labels]
        src = ColumnDataSource(src_data) # type: ignore

        p = figure(
            width=width, height=height, title=title,
            x_range=trace_labels, # type: ignore
            y_range=Range1d(0, 1.05),
            tools="", toolbar_location=None,
        )
        p.title.text_font_size = "11px"  # type: ignore
        colors = [colormap.get(f, "#aaaaaa") for f in active_funcs]
        p.vbar_stack(
            active_funcs, x="traces", width=0.6, color=colors, source=src,
        )
        p.add_tools(HoverTool(tooltips=[
            ("Function",   "$name"),
            ("Proportion", "@$name{0.0%}"),
            ("Trace",      "@traces"),
        ]))
        p.yaxis.axis_label = "Proportion"
        p.yaxis.formatter.use_scientific = False  # type: ignore
        p.xgrid.grid_line_color = None
        return p
