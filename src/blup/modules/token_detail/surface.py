from __future__ import annotations

from typing import Callable

from blup.types import TimestampNS, TraceMode, TraceSide
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models.annotations.geometry import Span
from bokeh.models.callbacks import CustomJS
from bokeh.models.css import InlineStyleSheet
from bokeh.models.formatters import CustomJSTickFormatter
from bokeh.models.layouts import LayoutDOM
from bokeh.models.ranges import Range1d
from bokeh.models.tickers import AdaptiveTicker
from bokeh.models.tools import HoverTool
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.markups import Div
from bokeh.plotting import ColumnDataSource, figure

from blup.bokeh.theme import PALETTE
from blup.modules.token_detail.types import (
    TokenDetailHistogramResult,
    TokenDetailScatterResult,
    TokenDetailTableModel,
    select_value_to_token,
)
from blup.state import TokenDetailChartMode


# TODO: add these to theme

_UPPER_COLOR = "#458588"  # gruvbox bright blue
_LOWER_COLOR = "#cc241d"  # gruvbox bright red

# _UPPER_COLOR = "#00b0ff"  # vivid azure
# _LOWER_COLOR = "#ff1744"  # vivid red

_MARKUP_FILL_STYLESHEET = InlineStyleSheet(
    css="""
    :host {
        margin: 0 !important;
    }
    .bk-clearfix {
        display: block !important;
    }
    """
)

def _empty_histogram_source() -> dict:
    return {"left": [], "right": [], "top": []}

def _empty_scatter_source() -> dict:
    return {"x": [], "y": []}

# TODO: shared with time_profile; find a centralized location for it
def _make_time_axis_formatter() -> CustomJSTickFormatter:
    return CustomJSTickFormatter(
        code="""
            const span = Math.max(...ticks) - Math.min(...ticks);

            let factor, unit, decimals;
            if (span >= 3600000) {
                factor = 3600000; unit = "h"; decimals = 1;
            } else if (span >= 60000) {
                factor = 60000; unit = "min"; decimals = 1;
            } else if (span >= 1000) {
                factor = 1000; unit = "s"; decimals = 1;
            } else {
                factor = 1; unit = "ms"; decimals = 0;
            }

            const val = tick / factor;
            let label = val.toFixed(decimals);
            label = label.replace(/\\.0+$/, "");
            return label + " " + unit;
        """
    )

def _header_html(model: TokenDetailTableModel) -> str:
    p = PALETTE

    if model.color:
        chip = (
            f'<div style="width: 16px; height: 16px; flex-shrink: 0;'
            f' background: {model.color};'
            f' border: 1px solid {p.bg3};"></div>'
        )
    else:
        chip = (
            f'<div style="width: 16px; height: 16px; flex-shrink: 0;'
            f' background: transparent;'
            f' border: 1px dashed {p.bg3};"></div>'
        )

    if model.subtitle:
        subtitle = (
            f'<div style="color: {p.fg0}; font-weight: 700;'
            f' font-size: 12px; margin-top: 2px;">{model.subtitle}</div>'
        )
    else:
        subtitle = (
            f'<div style="color: {p.muted}; font-size: 12px;'
            f' margin-top: 2px;">No token selected</div>'
        )

    return (
        f'<div style="display: flex; align-items: center; gap: 10px;'
        f' padding: 8px 10px; background: {p.bg1};'
        f' border-bottom: 1px solid {p.bg3}; font-family: monospace;'
        f' color: {p.fg1};">'
        f"{chip}"
        f'<div style="min-width: 0;">'
        f'<div style="color: {p.muted}; font-size: 10px;'
        f' letter-spacing: 0.08em; text-transform: uppercase;">'
        f"{model.title}</div>"
        f"{subtitle}</div></div>"
    )

def _delta_color(text: str) -> str:
    # delta = lower - upper: (+) => (green); (-) => (red)
    if text.startswith("+"):
        return PALETTE.red
    if text.startswith("-"):
        return PALETTE.green
    return PALETTE.fg0

class TokenDetailSurface:

    def __init__(self, *, height: int = 700) -> None:
        self.height = height
        self.doc = None
        self.root: LayoutDOM | None = None

        self.header: Div | None = None
        self.body: Div | None = None
        self.chart_fig = None

        self.hist_sources: dict[TraceSide, ColumnDataSource] = {}
        self.hist_renderers: dict[TraceSide, object] = {}
        self.scatter_sources: dict[TraceSide, ColumnDataSource] = {}
        self.scatter_renderers: dict[TraceSide, object] = {}

    def build(self) -> LayoutDOM:
        self.doc = curdoc()
        p = PALETTE

        self.header = Div(
            text="",
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            render_as_text=False,
            stylesheets=[_MARKUP_FILL_STYLESHEET],
        )
        self.body = Div(
            text="",
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 0),
            render_as_text=False,
            stylesheets=[_MARKUP_FILL_STYLESHEET],
        )

        fig = figure(
            title               = "Token Chart",
            min_width           = 320,
            min_height          = 240,
            sizing_mode         = "stretch_both",
            toolbar_location    = None,
            # context_menu        = None,
            output_backend      = "webgl",
            x_range             = Range1d(0, 1),
            active_drag="xbox_zoom",
            tools               = [
                "box_zoom", "xwheel_pan", "xbox_zoom",
                "reset", "save"
            ],
            # x_axis_label    = "Time (ms)",
            # y_axis_label    = "Exclusive (ms)",
        )
        fig.xaxis.formatter = _make_time_axis_formatter()
        fig.xaxis.ticker = AdaptiveTicker(
            desired_num_ticks=4,
            num_minor_ticks=0,
        )

        side_colors: dict[TraceSide, str] = {
            "upper": _UPPER_COLOR,
            "lower": _LOWER_COLOR,
        }

        for side in ("upper", "lower"):
            hist_src = ColumnDataSource(data=_empty_histogram_source())
            self.hist_sources[side] = hist_src
            self.hist_renderers[side] = fig.quad(
                left            = "left",
                right           = "right",
                bottom          = 0,
                top             = "top",
                source          = hist_src,
                fill_alpha      = 0.6,
                line_alpha      = 0.9,
                line_width      = 2,
                color           = side_colors[side],
                visible         = False,
                name            = f"token_detail_hist_{side}",
            )

            scatter_src = ColumnDataSource(data=_empty_scatter_source())
            self.scatter_sources[side] = scatter_src
            self.scatter_renderers[side] = fig.scatter(
                x               = "x",
                y               = "y",
                source          = scatter_src,
                size            = 5,
                alpha           = 0.8,
                color           = side_colors[side],
                marker          = "circle",
                visible         = False,
                name            = f"token_detail_scatter_{side}",
            )

        hist_hover = HoverTool(
            point_policy    = "follow_mouse",
            renderers       = list(self.hist_renderers.values()),           # type: ignore[attr-defined]
            tooltips        = f"""
                <div style="
                    font-family: monospace;
                    font-size: 11px;
                    color: {p.fg1};
                    background: {p.bg1};
                    padding: 6px 9px;
                    min-width: 150px;
                ">
                    <div style="
                        color: {p.muted};
                        font-size: 10px;
                        letter-spacing: 0.05em;
                        text-transform: uppercase;
                    ">time bin</div>
                    <div style="
                        color: {p.fg0};
                        font-weight: 700;
                        margin-top: 2px;
                    ">@left{{0.0}} – @right{{0.0}} ms</div>
                    <div style="
                        margin-top: 6px;
                        padding-top: 5px;
                        border-top: 1px solid {p.bg3};
                        display: flex;
                        justify-content: space-between;
                    ">
                        <span style="color: {p.muted};">excl</span>
                        <span style="color: {p.fg0};">@top{{0.000}} ms</span>
                    </div>
                </div>
            """,
        )
        scatter_hover = HoverTool(
            point_policy    = "snap_to_data",
            renderers       = list(self.scatter_renderers.values()),        # type: ignore[attr-defined]
            tooltips        = f"""
                <div style="
                    font-family: monospace;
                    font-size: 11px;
                    color: {p.fg1};
                    background: {p.bg1};
                    padding: 6px 9px;
                    min-width: 150px;
                ">
                    <div style="
                        margin-top: 2px;
                        display: flex;
                        justify-content: space-between;
                    ">
                        <span style="color: {p.muted};">start</span>
                        <span style="color: {p.fg0};">@x{{0.000}} ms</span>
                    </div>
                    <div style="
                        margin-top: 6px;
                        padding-top: 5px;
                        border-top: 1px solid {p.bg3};
                        display: flex;
                        justify-content: space-between;
                    ">
                        <span style="color: {p.muted};">duration</span>
                        <span style="color: {p.fg0};">@y{{0.000}} ms</span>
                    </div>
                </div>
            """,
        )
        fig.add_tools(hist_hover, scatter_hover)

        self.cursor_line = Span(
            location        = 0,
            dimension       = "height",
            line_color      = p.fg2,
            line_width      = 1,
            line_dash       = "dashed",
            line_alpha      = 0.5,
            visible         = False,
        )
        fig.add_layout(self.cursor_line)
        fig.js_on_event(
            "mousemove",
            CustomJS(
                args={"span": self.cursor_line, "fig": fig},
                code="""
                    const x = cb_obj.x;

                    if (typeof x !== "number" || !isFinite(x)) {
                        span.visible = false;
                        return;
                    }

                    const range = fig.x_range;
                    if (x < range.start || x > range.end) {
                        span.visible = false;
                        return;
                    }

                    span.location = x;
                    span.visible = true;
                """,
            ),
        )
        fig.js_on_event(
            "mouseleave",
            CustomJS(
                args={"span": self.cursor_line},
                code="span.visible = false;",
            ),
        )

        self.chart_fig = fig
        self.root = column(
            self.header,
            self.body,
            fig,
            sizing_mode="stretch_both",
        )
        return self.root

    def prepare_display(
        self,
        *,
        start_ns: TimestampNS,
        end_ns: TimestampNS,
        trace_mode: TraceMode,
        chart_mode: TokenDetailChartMode,
        show_stats: bool,
        show_chart: bool,
    ) -> None:
        fig = self.chart_fig
        if fig is None:
            raise RuntimeError(
                "TokenDetailSurface.build must be called before prepare_display"
            )
        if chart_mode not in ("histogram", "scatter"):
            raise ValueError(f"invalid chart_mode: {chart_mode!r}")

        dual = trace_mode == "dual"

        fig.x_range.start = start_ns / 1e6                                  # type: ignore[attr-defined]
        fig.x_range.end = end_ns / 1e6                                      # type: ignore[attr-defined]

        fig.visible = show_chart
        if self.body:
            self.body.visible = show_stats

        if chart_mode == "histogram":
            if fig.title:
                fig.title.text = "Exclusive total by time bin"              # type: ignore[attr-defined]
            # fig.yaxis.axis_label = "Exclusive (ms)"
        else:
            if fig.title:
                fig.title.text = "Call durations over time"                 # type: ignore[attr-defined]
            # fig.yaxis.axis_label = "Duration (ms)"

        for side in ("upper", "lower"):
            self.hist_sources[side].data = _empty_histogram_source()
            self.scatter_sources[side].data = _empty_scatter_source()

            side_active = side == "upper" or dual
            self.hist_renderers[side].visible = (                           # type: ignore[attr-defined]
                show_chart and chart_mode == "histogram" and side_active
            )
            self.scatter_renderers[side].visible = (                        # type: ignore[attr-defined]
                show_chart and chart_mode == "scatter" and side_active
            )

    def apply_table_result(self, model: TokenDetailTableModel) -> None:
        if self.header is not None:
            self.header.text = _header_html(model)
        if self.body is not None:
            self.body.text = self._render_stats_html(model)

    def apply_histogram_result(self, result: TokenDetailHistogramResult) -> None:
        self.hist_sources[result.trace_side].data = result.src

    def apply_scatter_result(self, result: TokenDetailScatterResult) -> None:
        self.scatter_sources[result.trace_side].data = result.src

    def _render_stats_html(self, model: TokenDetailTableModel) -> str:
        if not model.metric:
            return ""

        p = PALETTE
        th = (
            f'text-align: left; color: {p.muted}; font-size: 10px;'
            f' text-transform: uppercase; letter-spacing: 0.05em;'
            f' font-weight: 400; padding: 5px 8px 4px 8px;'
            f' border-bottom: 1px solid {p.bg3};'
        )
        td = f"padding: 3px 8px; border-bottom: 1px solid {p.bg2};"

        def _row(metric: str, cells: list[tuple[str, str]]) -> str:
            out = f'<tr><td style="{td} color: {p.fg2};">{metric}</td>'
            for text, color in cells:
                out += f'<td style="{td} color: {color};">{text}</td>'
            return out + "</tr>"

        if model.dual_mode:
            rows = "".join(
                _row(m, [
                    (a, p.fg0),
                    (b, p.fg0),
                    (d, _delta_color(d)),
                    (pc, _delta_color(pc)),
                ])
                for m, a, b, d, pc in zip(
                    model.metric,
                    model.upper,
                    model.lower,
                    model.delta,
                    model.percent,
                )
            )
            head = (
                f'<th style="{th}">Metric</th>'
                f'<th style="{th}">{model.upper_label}</th>'
                f'<th style="{th}">{model.lower_label}</th>'
                f'<th style="{th}">Delta</th>'
                f'<th style="{th}">% diff</th>'
            )
        else:
            rows = "".join(
                _row(m, [
                    (a, p.fg0),
                    (d, _delta_color(d)),
                    (pc, _delta_color(pc)),
                ])
                for m, a, d, pc in zip(
                    model.metric,
                    model.upper,
                    model.delta,
                    model.percent,
                )
            )
            head = (
                f'<th style="{th}">Metric</th>'
                f'<th style="{th}">{model.upper_label}</th>'
                f'<th style="{th}">Delta</th>'
                f'<th style="{th}">% diff</th>'
            )

        return (
            f'<div style="padding: 2px 2px 8px 2px;">'
            f'<table style="width: 100%; border-collapse: collapse;'
            f' font-family: monospace; font-size: 11px;'
            f' color: {p.fg1};">'
            f"<thead><tr>{head}</tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )


