from __future__ import annotations

from typing import Callable

from blup.bokeh.styles import style_widget
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models.css import InlineStyleSheet
from bokeh.models.layouts import LayoutDOM
from bokeh.models.widgets.inputs import Select
from bokeh.models.widgets.tables import (
    DataTable,
    HTMLTemplateFormatter,
    TableColumn,
)
from bokeh.plotting import ColumnDataSource

from blup.bokeh.theme import PALETTE, make_widget_stylesheet
from blup.modules.token_list.types import TokenListResult
from blup.state import TokenListOrder, TokenListSortDirection
from blup.types import TokenKey, TraceMode


_ORDER_OPTIONS: list[tuple[TokenListOrder, str]] = [
    ("delta", "Delta excl"),
    ("excl", "Excl (upper)"),
    ("calls", "Call count"),
    ("name", "Name"),
    ("token", "Token id"),
]

_DIRECTION_OPTIONS: list[tuple[TokenListSortDirection, str]] = [
    ("descending", "Desc"),
    ("ascending", "Asc"),
]

_COLOR_CHIP_TEMPLATE = (
    '<div style="width:14px; height:14px; margin:2px;'
    "background:<%= color %>; border:1px solid #504945;"
    '"></div>'
)


def _empty_source() -> dict:
    return {
        "color":        [],
        "name":         [],
        "id_label":     [],
        "token_key":    [],
        "token_type":   [],
        "token_id":     [],
        "rank":         [],
        "excl_upper":   [],
        "excl_lower":   [],
        "delta_excl":   [],
        "share":        [],
    }

def _make_table_stylesheet() -> InlineStyleSheet:
    p = PALETTE
    return InlineStyleSheet(
        css=f"""
        :host {{
            background: {p.bg1};
            color: {p.fg1};
            font-family: monospace;
            font-size: 11px;
        }}

        :host .slick-header {{
            background: {p.bg2};
            border-bottom: 2px solid {p.bg3};
        }}
        :host .slick-header-column {{
            background: {p.bg2};
            color: {p.muted};
            font-family: monospace;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-right: 1px solid {p.bg3};
        }}
        :host .slick-header-column:hover {{
            color: {p.yellow};
        }}

        :host .slick-row {{
            background: {p.bg1};
            color: {p.fg1};
            border: none;
        }}
        :host .slick-row.odd {{
            background: {p.bg0};
        }}
        :host .slick-row:hover {{
            background: {p.bg2};
        }}
        :host .slick-row.active {{
            background: {p.bg3};
            color: {p.fg0};
        }}
        :host .slick-cell {{
            border: none;
            border-bottom: 1px solid {p.bg2};
            font-family: monospace;
            font-size: 11px;
        }}
        :host .slick-cell.selected {{
            background: {p.bg3};
        }}
        """
    )

class TokenListTable:

    def __init__(self, *, height: int = 700) -> None:
        self.height = height

        self.doc = None
        self.table: DataTable | None = None

        self.root: LayoutDOM | None = None

        self.source = ColumnDataSource(data=_empty_source())
        self.order_select: Select | None = None
        self.direction_select: Select | None = None

        self.on_token_selected: (
            Callable[[TokenKey | None], None] | None
        ) = None
        self.on_order_changed: (
            Callable[[TokenListOrder], None] | None
        ) = None
        self.on_direction_changed: (
            Callable[[TokenListSortDirection], None] | None
        ) = None

        self._ignore_selection_callbacks = False
        self._ignore_widget_callbacks = False

        self._single_columns: list[TableColumn] = []
        self._dual_columns: list[TableColumn] = []

    def build(self) -> LayoutDOM:
        self.doc = curdoc()

        self.order_select = Select(
            title       = "Order by",
            value       = "delta",
            options     = _ORDER_OPTIONS,                                   # type: ignore[attr-defined]
            sizing_mode = "stretch_width",
            stylesheets = [make_widget_stylesheet()],
        )
        style_widget(self.order_select)
        self.order_select.on_change("value", self._on_order_widget)

        self.direction_select = Select(
            title       = "Direction",
            value       = "descending",
            options     = _DIRECTION_OPTIONS,                               # type: ignore[attr-defined]
            sizing_mode = "stretch_width",
            stylesheets = [make_widget_stylesheet()],
        )
        style_widget(self.direction_select)
        self.direction_select.on_change("value", self._on_direction_widget)

        color_formatter = HTMLTemplateFormatter(template=_COLOR_CHIP_TEMPLATE)

        col_color = TableColumn(
            field       = "color",
            title       = "",
            width       = 6,
            formatter   = color_formatter,
        )
        col_name = TableColumn(
            field       = "name",
            title       = "Token",
        )
        col_id = TableColumn(
            field       = "id_label",
            title       = "ID",
            width       = 40,
        )
        col_excl_upper = TableColumn(
            field       = "excl_upper",
            title       = "Excl (upper)",
            width       = 100,
        )
        col_excl_lower = TableColumn(
            field       = "excl_lower",
            title       = "Excl (lower)",
            width       = 100,
        )
        col_delta = TableColumn(
            field       = "delta_excl",
            title       = "Delta excl",
            width       = 100,
        )
        col_share = TableColumn(
            field       = "share",
            title       = "Share",
            width       = 60,
        )

        self._single_columns = [
            col_color,
            col_name,
            col_id,
            col_excl_upper
        ]
        self._dual_columns = [
            col_color,
            col_name,
            col_id,
            col_excl_upper,
            col_excl_lower,
            col_delta,
            col_share,
        ]

        self.table = DataTable(
            source          = self.source,
            columns         = self._single_columns,
            selectable      = True,
            sortable        = False,
            reorderable     = False,
            index_position  = None,
            row_height      = 26,
            sizing_mode     = "stretch_both",
            stylesheets     = [_make_table_stylesheet()],
        )
        self.source.selected.on_change("indices", self._on_source_selected)

        controls = row(
            self.order_select,
            self.direction_select,
            sizing_mode = "stretch_width",
        )
        self.root = column(
            controls,
            self.table,
            sizing_mode = "stretch_both",
        )
        return self.root

    def prepare_display(
        self,
        *,
        trace_mode: TraceMode,
        order: TokenListOrder,
        direction: TokenListSortDirection,
    ) -> None:
        table = self.table
        if table is None:
            raise RuntimeError(
                "TokenListSurface.build must be called before prepare_display"
            )

        dual = trace_mode == "dual"
        table.columns = list(
            self._dual_columns if dual else self._single_columns
        )

        self._ignore_widget_callbacks = True
        try:
            if self.order_select is not None and self.order_select.value != order:
                self.order_select.value = order
            if (
                self.direction_select is not None
                and self.direction_select.value != direction
            ):
                self.direction_select.value = direction
        finally:
            self._ignore_widget_callbacks = False

        self._ignore_selection_callbacks = True
        try:
            self.source.selected.indices = []
            self.source.data = _empty_source()
        finally:
            self._ignore_selection_callbacks = False

    def apply_result(self, result: TokenListResult) -> None:
        self._ignore_selection_callbacks = True
        try:
            self.source.selected.indices = []
            self.source.data = result.src
        finally:
            self._ignore_selection_callbacks = False

    def sync_selection(self, token: tuple[int, int] | None) -> None:
        indices: list[int] = []
        if token is not None:
            data = self.source.data
            token_types = data.get("token_type", [])
            token_ids = data.get("token_id", [])
            for i in range(len(token_types)):
                if (
                    int(token_types[i]) == token[0]
                    and int(token_ids[i]) == token[1]
                ):
                    indices = [i]
                    break

        self._ignore_selection_callbacks = True
        try:
            if list(self.source.selected.indices) != indices:
                self.source.selected.indices = indices
        finally:
            self._ignore_selection_callbacks = False

    def _on_source_selected(self, attr: str, old, new) -> None:
        if self._ignore_selection_callbacks:
            return
        indices = list(new)
        if not indices:
            if self.on_token_selected is not None:
                self.on_token_selected(None)
            return

        i = int(indices[0])
        data = self.source.data
        token_types = data.get("token_type", [])
        token_ids = data.get("token_id", [])
        if i < 0 or i >= len(token_types) or i >= len(token_ids):
            return

        token = (int(token_types[i]), int(token_ids[i]))
        if self.on_token_selected is not None:
            self.on_token_selected(token)

    def _on_order_widget(
        self,
        attr: str,
        old: TokenListOrder,
        new: TokenListOrder,
    ) -> None:
        if self._ignore_widget_callbacks:
            return
        if self.on_order_changed is not None:
            self.on_order_changed(new)

    def _on_direction_widget(
        self,
        attr: str,
        old: TokenListSortDirection,
        new: TokenListSortDirection
    ) -> None:
        if self._ignore_widget_callbacks:
            return
        if self.on_direction_changed is not None:
            self.on_direction_changed(new)


