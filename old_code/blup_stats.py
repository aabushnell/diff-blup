from bokeh.models import ColumnDataSource, DataTable, TableColumn

from blup_trace import TraceComparison

class Stats:

    COLUMNS = [
        ("function",             "Function"),
        ("diff_mean_rel",        "Δ mean rel"),
        ("diff_p90_rel",         "Δ p90 rel"),
        ("ks",                   "KS"),
        ("wasserstein",          "Wasserstein"),
        ("diff_contention_rel",  "Δ content rel"),
        ("base_score",           "Unweighted Score"),
        ("time_frac",            "Time Fraction"),
        ("score",                "Final Score"),
        ("contention1",          "Contention T1"),
        ("contention2",          "Contention T2"),
    ]

    def __init__(self, comparison: TraceComparison,
                 width: int = 2200, height: int = 300):
        self.comparison   = comparison
        self.width        = width
        self.height       = height
        self.score_source = ColumnDataSource(comparison.score_df)
        self.table        = None

    @property
    def functions_scored(self) -> list[str]:
        return self.comparison.functions_scored

    def build(self) -> DataTable:
        cols = [TableColumn(field=f, title=t) for f, t in self.COLUMNS]
        self.table = DataTable(
            source=self.score_source, columns=cols,
            width=self.width, height=self.height,
            index_position=None, sortable=True,
        )
        return self.table

    def refresh(self):
        self.score_source.data = ColumnDataSource.from_df(
            self.comparison.score_df
        )



