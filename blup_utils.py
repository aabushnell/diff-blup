import re

import pandas as pd

from bokeh.palettes import Set3 as palette

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    # by default sorted(list) will generate:
    # P0T0
    # P1T0
    # P10T0
    # ...
    # P2T0

    # This function makes sure that P2 comes before P10:
    # P0T0
    # P1T0
    # P2T0
    # ...
    # P9T0
    # P10T0    
    
    return [ atoi(c) for c in re.split(r"(\d+)", text) ]

def choose_palette(functions):
    max_id=len(palette)
    min_id=3
    id=max(min_id, min(len(functions), max_id))
    return palette[id]

def apply_top_bottom(
    df: pd.DataFrame,
    label: str,
    stack_mode: str,
    depth_step: float,
    max_depth: int,
) -> None:
    c = df["center"]
    d = df["depth"]
    if stack_mode == "diverge":
        gap = depth_step / 10.0
        if label == "Trace 1":
            df["top"]    = c - gap - d * depth_step
            df["bottom"] = c - gap - (d + 1) * depth_step
        else:
            df["bottom"] = c + gap + d * depth_step
            df["top"]    = c + gap + (d + 1) * depth_step
    else:  # converge
        HALF_BAND = 0.45
        step = HALF_BAND / max(max_depth + 1, 1)
        if label == "Trace 1":
            df["top"]    = c + HALF_BAND - d * step
            df["bottom"] = c + HALF_BAND - (d + 1) * step
        else:
            df["bottom"] = c - HALF_BAND + d * step
            df["top"]    = c - HALF_BAND + (d + 1) * step


