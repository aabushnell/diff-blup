from __future__ import annotations

from bokeh.layouts import column
from bokeh.models.widgets.markups import Div

from trace_session import TraceSession


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
        stack_order: str,
    ) -> None:
        top1 = self.top_token_labels(self.t1, limit=8)
        top2 = self.top_token_labels(self.t2, limit=8)

        self.div.text = f"""
        <div style="padding:8px">
          <h3 style="margin-top:0">Inspector</h3>
          <p><b>Trace 1:</b> {self.t1.meta.path}</p>
          <p><b>Trace 2:</b> {self.t2.meta.path}</p>
          <p><b>Threads:</b> {len(active_threads)}</p>
          <p><b>Quanta bins:</b> {n_quanta}</p>
          <p><b>Snapshot mode:</b> {mode}</p>
          <p><b>Stack order:</b> {stack_order}</p>
          <p><b>Top functions T1:</b> {", ".join(top1) if top1 else "(none)"}</p>
          <p><b>Top functions T2:</b> {", ".join(top2) if top2 else "(none)"}</p>
        </div>
        """

    def top_token_labels(self, session: TraceSession, limit: int = 8) -> list[str]:
        out: list[str] = []
        for key in session.summary.top_tokens[:limit]:
            name = session.meta.token_key_to_name.get(key, key)
            if name == key:
                out.append(key)
            else:
                out.append(f"{name} ({key})")
        return out
