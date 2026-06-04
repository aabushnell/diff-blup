from bokeh.models import ColumnDataSource, Range1d, TapTool, BoxSelectTool, HoverTool
from bokeh.plotting import figure

import numpy as np

from scipy.stats import gaussian_kde

from blup_trace import BlupTrace
from blup_utils import natural_keys

def _empty_seq_dict():
    return dict(x=[], y=[], thread=[], trace=[])

T1_COLORS = ["#1f77b4", "#4e9dcc", "#85c1e0", "#aedcf0", "#d0eef9"]
T2_COLORS = ["#d62728", "#e5605f", "#f0908f", "#f8bcbb", "#fde0df"]

class Distribution:

    def __init__(self, t1: BlupTrace, t2: BlupTrace,
                 width: int, height: int):
        # set traces for comparison
        self.t1 = t1
        self.t2 = t2

        # set panel parameters
        self.width  = width
        self.height = height
        self.fig = None
        self._current_func: str | None = None
        self._on_bin_click = None

        self._hist_xrange = Range1d(0, 1)
        self._hist_yrange = Range1d(0, 1)
        self._cdf_xrange  = Range1d(0, 1)
        self._cdf_yrange  = Range1d(0, 1)
        self._seq_xrange  = Range1d(0, 1)
        self._seq_yrange  = Range1d(0, 1)

        # init data sources
        self.src_h1 = ColumnDataSource(data=dict(top=[], left=[], right=[]))
        self.src_h2 = ColumnDataSource(data=dict(top=[], left=[], right=[]))
        self.src_k1 = ColumnDataSource(data=dict(x=[], y=[]))
        self.src_k2 = ColumnDataSource(data=dict(x=[], y=[]))
        self.src_c1 = ColumnDataSource(data=dict(x=[], y=[]))
        self.src_c2 = ColumnDataSource(data=dict(x=[], y=[]))

        self._seq_sources: list[ColumnDataSource] = []

    def build(self) -> figure:
        p = figure(
            width=self.width, height=self.height,
            x_range=self._hist_xrange,
            y_range=self._hist_yrange,
            x_axis_label="Duration (s)",
            y_axis_label="Count",
            title="Histogram",
        )

        tap = TapTool()
        box_select = BoxSelectTool(dimensions="width", description="Select bin range")
        p.add_tools(tap, box_select)
        p.toolbar.active_drag = box_select

        # histogram glyphs
        self.r_h1 = p.quad(top="top", bottom=0, left="left", right="right",
                           source=self.src_h1,
                           fill_color="navy", fill_alpha=0.35, line_color=None,
                           legend_label="Trace 1 hist")
        self.r_h2 = p.quad(top="top", bottom=0, left="left", right="right",
                           source=self.src_h2,
                           fill_color="firebrick", fill_alpha=0.35, line_color=None,
                           legend_label="Trace 2 hist")

        # KDE glyphs
        self.r_k1 = p.line("x", "y", source=self.src_k1,
                           color="navy", line_width=2,
                           legend_label="Trace 1 KDE")
        self.r_k2 = p.line("x", "y", source=self.src_k2,
                           color="firebrick", line_width=2,
                           legend_label="Trace 2 KDE")

        # CDF glyphs
        self.r_c1 = p.line("x", "y", source=self.src_c1,
                           color="navy", line_width=2,
                           legend_label="Trace 1 CDF", visible=False)
        self.r_c2 = p.line("x", "y", source=self.src_c2,
                           color="firebrick", line_width=2,
                           legend_label="Trace 2 CDF", visible=False)

        self._seq_renderers: list = []

        self.src_h1.selected.on_change("indices", self._on_bin_selected)
        self.src_h2.selected.on_change("indices", self._on_bin_selected)

        p.legend.click_policy = "hide"
        self.fig = p
        return p

    def update(self, func: str, mode: str = "hist", bins: int = 40):
        self._current_func = func
        d1 = self.t1.get_durations(func)
        d2 = self.t2.get_durations(func)

        is_hist = mode == "hist"
        is_cdf  = mode == "cdf"
        is_seq  = mode == "seq"

        for r in (self.r_h1, self.r_h2, self.r_k1, self.r_k2):
            r.visible = is_hist
        for r in (self.r_c1, self.r_c2):
            r.visible = is_cdf
        for r in self._seq_renderers:
            r.visible = False

        if self.fig is not None:
            self.fig.legend.visible = not is_seq

        if is_seq:
            self._update_seq(func)
            if self.fig is not None:
                self.fig.x_range = self._seq_xrange          # type: ignore
                self.fig.y_range = self._seq_yrange          # type: ignore
                self.fig.xaxis[0].axis_label = "Occurrence #"
                self.fig.yaxis[0].axis_label = "Duration (s)"
                self.fig.title.text = "Duration per Occurrence"  # type: ignore
            return

        empty = dict(top=[], left=[], right=[])
        if len(d1) == 0 and len(d2) == 0:
            for src in [self.src_h1, self.src_h2]:
                src.data = empty  # type: ignore
            for src in [self.src_k1, self.src_k2, self.src_c1, self.src_c2]:
                src.data = dict(x=[], y=[])
            return

        all_d = np.concatenate([d1, d2])
        xmin, xmax = all_d.min(), all_d.max()
        if xmin == xmax:
            xmin -= 1e-9; xmax += 1e-9

        h1, edges = np.histogram(d1, bins=bins, range=(xmin, xmax))
        h2, _     = np.histogram(d2, bins=bins, range=(xmin, xmax))
        self.src_h1.data = dict(top=h1, left=edges[:-1], right=edges[1:])
        self.src_h2.data = dict(top=h2, left=edges[:-1], right=edges[1:])
        self.src_h1.selected.indices = []
        self.src_h2.selected.indices = []

        x_grid = np.linspace(xmin, xmax, 200)
        bin_width = edges[1] - edges[0]
        for src_k, d in [(self.src_k1, d1), (self.src_k2, d2)]:
            if len(d) > 1:
                y = gaussian_kde(d)(x_grid) * len(d) * bin_width
            else:
                y = np.zeros_like(x_grid)
            src_k.data = dict(x=x_grid, y=y)

        for src_c, h in [(self.src_c1, h1), (self.src_c2, h2)]:
            total = h.sum()
            cdf = np.cumsum(h) / total if total > 0 else np.zeros_like(h)
            src_c.data = dict(x=edges[1:], y=cdf)

        self._hist_xrange.start = float(xmin)
        self._hist_xrange.end   = float(xmax)
        self._cdf_xrange.start  = float(xmin)
        self._cdf_xrange.end    = float(xmax)

        hist_ymax = max(
            float(h1.max()) if len(h1) else 0.0,
            float(h2.max()) if len(h2) else 0.0,
            float(np.max(self.src_k1.data["y"])) if len(self.src_k1.data["y"]) else 0.0, # type: ignore
            float(np.max(self.src_k2.data["y"])) if len(self.src_k2.data["y"]) else 0.0, # type: ignore
        )
        self._hist_yrange.start = 0.0
        self._hist_yrange.end   = max(hist_ymax, 1.0) * 1.05
        self._cdf_yrange.start = 0.0
        self._cdf_yrange.end   = 1.02

        if self.fig is not None:
            if is_hist:
                self.fig.x_range           = self._hist_xrange
                self.fig.y_range           = self._hist_yrange  # type: ignore
                self.fig.xaxis[0].axis_label = "Duration (s)"
                self.fig.yaxis[0].axis_label = "Count"
                self.fig.title.text        = "Histogram"        # type: ignore
            else:
                self.fig.x_range           = self._cdf_xrange
                self.fig.y_range           = self._cdf_yrange   # type: ignore
                self.fig.xaxis[0].axis_label = "Duration (s)"
                self.fig.yaxis[0].axis_label = "Cumulative probability"
                self.fig.title.text        = "CDF"              # type: ignore

    def set_on_bin_click(self, callback):
        self._on_bin_click = callback

    def _build_seq_renderers(self, n_needed: int):
        if self.fig is None:
            return
        while len(self._seq_renderers) < n_needed:
            src = ColumnDataSource(_empty_seq_dict())  # type: ignore
            self._seq_sources.append(src)
            r = self.fig.scatter("x", "y", source=src,
                              line_width=1.5, visible=False,
                              name=f"seq_line_{len(self._seq_renderers)}")
            self.fig.add_tools(HoverTool(
                renderers=[r],
                tooltips=[
                    ("Trace",        "@trace"),
                    ("Thread",       "@thread"),
                    ("Occurrence",   "@x"),
                    ("Duration (s)", "@y{0.000000}"),
                ],
            ))
            self._seq_renderers.append(r)

    def _update_seq(self, func: str):
        threads = sorted(
            set(self.t1.threads) | set(self.t2.threads), key=natural_keys
        )

        series: list[tuple[str, str, np.ndarray]] = []
        for trace_obj, label in [(self.t1, "T1"), (self.t2, "T2")]:
            for thread in threads:
                mask = (trace_obj.df["function"] == func) & \
                       (trace_obj.df["thread"]   == thread)
                sub = trace_obj.df.loc[mask].sort_values("start")
                if sub.empty:
                    continue
                dur_s = sub["duration"].apply(
                    lambda td: td.total_seconds()
                ).to_numpy(dtype=np.float64)
                series.append((label, thread, dur_s))

        if not series:
            for r in self._seq_renderers:
                r.visible = False
            return

        self._build_seq_renderers(len(series))

        all_y   = np.concatenate([s[2] for s in series])
        y_pad   = (all_y.max() - all_y.min()) * 0.05 or all_y.max() * 0.05 or 1e-9
        self._seq_yrange.start = max(0.0, all_y.min() - y_pad)
        self._seq_yrange.end   = all_y.max() + y_pad

        max_len = max(len(s[2]) for s in series)
        self._seq_xrange.start = 0
        self._seq_xrange.end   = max(1, max_len - 1)

        t1_thread_idx: dict[str, int] = {}
        t2_thread_idx: dict[str, int] = {}

        for i, (label, thread, dur_s) in enumerate(series):
            src = self._seq_sources[i]
            r   = self._seq_renderers[i]

            src.data = dict(                                 # type: ignore
                x=list(range(len(dur_s))),
                y=dur_s.tolist(),
                thread=[thread] * len(dur_s),
                trace=[label]   * len(dur_s),
            )

            if label == "T1":
                cidx = t1_thread_idx.get(thread, len(t1_thread_idx))
                t1_thread_idx[thread] = cidx
                color = T1_COLORS[cidx % len(T1_COLORS)]
                marker = "circle"
            else:
                cidx = t2_thread_idx.get(thread, len(t2_thread_idx))
                t2_thread_idx[thread] = cidx
                color = T2_COLORS[cidx % len(T2_COLORS)]
                marker = "square"

            r.glyph.fill_color = color   # type: ignore
            r.glyph.line_color = color   # type: ignore
            r.glyph.marker     = marker  # type: ignore
            r.visible          = True

        for r in self._seq_renderers[len(series):]:
            r.visible = False

    def _on_bin_selected(self, attr, old, new):
        if self._on_bin_click is None or self._current_func is None:
            return
        if not new:
            self._on_bin_click(None, None, None)
            return
        lefts  = self.src_h1.data["left"]   # type: ignore
        rights = self.src_h1.data["right"]  # type: ignore
        low  = float(min(lefts[i]  for i in new)) # type: ignore
        high = float(max(rights[i] for i in new)) # type: ignore
        self._on_bin_click(self._current_func, low, high)
