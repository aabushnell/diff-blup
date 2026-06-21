from __future__ import annotations

from bokeh.layouts import column
from bokeh.models.widgets.markups import Div

from data_model import SummaryQuery, TokenMode, as_token_key
from trace_session import TraceSession
from utils import timed


class InspectorView:
    def __init__(self, t1: TraceSession, t2: TraceSession, width: int = 340) -> None:
        self.t1 = t1
        self.t2 = t2
        self.width = width
        self.div = Div(width=self.width, sizing_mode="fixed")
        self.root = column(self.div, width=self.width)

    def build(self):
        self.div.text = "<b>Inspector</b>"
        return self.root

    def update(
        self,
        *,
        active_threads: list[str],
        n_quanta: int,
        mode: str,
        token_mode: TokenMode,
        stack_order: str,
    ) -> None:
        top1 = self.top_token_labels(self.t1, limit=8, token_mode=token_mode)
        top2 = self.top_token_labels(self.t2, limit=8, token_mode=token_mode)

        token_view_label = "named" if token_mode == "category" else "raw"

        self.div.text = f"""
        <div style="padding:8px">
          <h3 style="margin-top:0">Inspector</h3>
          <p><b>Trace 1:</b> {self.t1.meta.path}</p>
          <p><b>Trace 2:</b> {self.t2.meta.path}</p>
          <p><b>Threads:</b> {len(active_threads)}</p>
          <p><b>Quanta bins:</b> {n_quanta}</p>
          <p><b>Token view:</b> {token_view_label}</p>
          <p><b>Snapshot mode:</b> {mode}</p>
          <p><b>Stack order:</b> {stack_order}</p>
          <p><b>Top functions T1:</b> {", ".join(top1) if top1 else "(none)"}</p>
          <p><b>Top functions T2:</b> {", ".join(top2) if top2 else "(none)"}</p>
        </div>
        """

    def top_token_labels(
        self,
        session: TraceSession,
        limit: int = 8,
        token_mode: TokenMode = "raw"
    ) -> list[str]:
        summary = session.summarize_tokens(
            SummaryQuery(
                thread_ids=tuple(int(id) for id in session.meta.thread_ids),
                fidelity="fast",
                token_mode=token_mode,
                top_k=limit,
            )
        )

        labels: list[str] = []
        for row in summary.tokens[:limit]:
            key = as_token_key(int(row.token_type), int(row.token_id))
            labels.append(session.meta.token_key_to_name.get(key, key))
        return labels

