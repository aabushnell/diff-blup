from bokeh.models import ColumnDataSource, Range1d, TapTool
from bokeh.plotting import figure

import numpy as np

from scipy.stats import gaussian_kde

from blup_trace import BlupTrace

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
        self._r   = {}
        self._hist_y_range = None
        self._cdf_y_range  = Range1d(0, 1)
        self._current_func: str | None = None
        self._on_bin_click = None

        # init data sources
        self.src_h1 = ColumnDataSource(data=dict(top=[], left=[], right=[]))
        self.src_h2 = ColumnDataSource(data=dict(top=[], left=[], right=[]))
        self.src_k1 = ColumnDataSource(data=dict(x=[], y=[]))
        self.src_k2 = ColumnDataSource(data=dict(x=[], y=[]))
        self.src_c1 = ColumnDataSource(data=dict(x=[], y=[]))
        self.src_c2 = ColumnDataSource(data=dict(x=[], y=[]))


    def build(self) -> figure:
        p = figure(width=self.width, height=self.height,
                   x_axis_label="Duration (s)", y_axis_label="Count",
                   title="Histogram")

        tap = TapTool()
        p.add_tools(tap)

        self._r["h1"] = p.quad(top="top", bottom=0, left="left", right="right",
                               source=self.src_h1, fill_color="navy",
                               fill_alpha=0.35, line_color=None,
                               legend_label="Trace 1 hist")
        self._r["h2"] = p.quad(top="top", bottom=0, left="left", right="right",
                               source=self.src_h2, fill_color="firebrick",
                               fill_alpha=0.35, line_color=None,
                               legend_label="Trace 2 hist")
        self._r["k1"] = p.line("x", "y", source=self.src_k1, color="navy",
                               line_width=2, legend_label="Trace 1 KDE")
        self._r["k2"] = p.line("x", "y", source=self.src_k2, color="firebrick",
                               line_width=2, legend_label="Trace 2 KDE")
        self._r["c1"] = p.line("x", "y", source=self.src_c1, color="navy",
                               line_width=2, legend_label="Trace 1 CDF",
                               visible=False)
        self._r["c2"] = p.line("x", "y", source=self.src_c2, color="firebrick",
                               line_width=2, legend_label="Trace 2 CDF",
                               visible=False)

        self.src_h1.selected.on_change("indices", self._on_bin_selected)
        self.src_h2.selected.on_change("indices", self._on_bin_selected)

        p.legend.click_policy = "hide"
        self._hist_y_range = p.y_range
        self.fig = p
        return p

    def update(self, func: str, mode: str = "hist", bins: int = 40):
        d1 = self.t1.get_durations(func)
        d2 = self.t2.get_durations(func)

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

        self._current_func = func
        self.src_h1.selected.indices = []
        self.src_h2.selected.indices = []

        x_grid    = np.linspace(xmin, xmax, 200)
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

        is_hist = (mode == "hist")
        for key in ("h1", "h2", "k1", "k2"):
            self._r[key].visible = is_hist
        for key in ("c1", "c2"):
            self._r[key].visible = not is_hist

        if self.fig is not None:
            if is_hist:
                self.fig.y_range = self._hist_y_range  # type: ignore
                self.fig.yaxis[0].axis_label = "Count"
                self.fig.title.text = "Histogram"  # type: ignore
            else:
                self.fig.y_range = self._cdf_y_range
                self.fig.yaxis[0].axis_label = "Cumulative probability"
                self.fig.title.text = "CDF"  # type: ignore

    def set_on_bin_click(self, callback):
        self._on_bin_click = callback

    def _on_bin_selected(self, attr, old, new):
        if self._on_bin_click is None or self._current_func is None:
            return
        if not new:
            self._on_bin_click(None, None, None)
            return
        i      = new[0]
        low_s  = float(self.src_h1.data["left"][i])  # type: ignore
        high_s = float(self.src_h1.data["right"][i]) # type: ignore
        self._on_bin_click(self._current_func, low_s, high_s)
