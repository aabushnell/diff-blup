from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal


TimeMode = Literal["quanta"]
UiTokenMode = Literal["raw", "named"]

@dataclass
class AppState:
    active_threads:         tuple[str, ...]
    time_mode:              TimeMode = "quanta"
    n_quanta:               int = 100
    token_mode:             UiTokenMode = "raw"
    quanta_mode:            Literal["fast", "balanced", "exact"] = "fast"
    quanta_stack_order:     Literal["global", "local"] = "global"

    quanta_window_t0_ns:    int | None = None
    quanta_window_t1_ns:    int | None = None
    quanta_zoom_level:      int = 0

    selected_function:      Optional[str] = None
    selected_thread:        Optional[str] = None
