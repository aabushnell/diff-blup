from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal


TimeMode = Literal["quanta"]

@dataclass
class AppState:
    active_threads: tuple[str, ...]
    time_mode: TimeMode = "quanta"
    n_quanta: int = 100
    quanta_mode: Literal["fast", "exact"] = "fast"
    quanta_stack_order: Literal["global", "local"] = "global"
    selected_function: Optional[str] = None
    selected_thread: Optional[str] = None
