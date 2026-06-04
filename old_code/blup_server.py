import sys

from bokeh.io import curdoc
from bokeh.layouts import layout, column
from bokeh.models import Select, RadioButtonGroup, MultiSelect, Button, Slider, Spinner, Div

import blup_trace as bt
from blup_utils import natural_keys

from blup_distribution import Distribution
from blup_stats import Stats
from blup_timeline import Timeline
from blup_components import Components

TIMELINE_WIDTH   = 1200
DIST_WIDTH       = 800
EXPANDED_WIDTH   = TIMELINE_WIDTH + DIST_WIDTH
dist_visible = True

# load both traces
t1 = bt.BlupTrace(sys.argv[1])
t2 = bt.BlupTrace(sys.argv[2])
comparison = bt.TraceComparison(t1, t2)

# init panel objects
timeline_obj   = Timeline(t1, t2, width=1200, height=1000)
distrib_obj    = Distribution(t1, t2, width=800,  height=1000)
stats_obj      = Stats(comparison)
components_obj = Components(t1, t2)

timeline_fig, minimap  = timeline_obj.build()
hist_fig               = distrib_obj.build()
stats_table            = stats_obj.build()
components_panel       = components_obj.build()

# init ui elements
initial_func    = stats_obj.functions_scored[0] if stats_obj.functions_scored else t1.functions[0]
all_threads     = sorted(set(t1.threads) | set(t2.threads), key=natural_keys)

thread_select   = MultiSelect(value=all_threads, options=all_threads)
mode_btns       = RadioButtonGroup(labels=["Gantt", "Flame", "Subtree", "Quanta"], active=0)
stack_toggle    = RadioButtonGroup(labels=["Diverge", "Converge"], active=0)
toggle_dist_btn = Button(label="◀ Hide Distribution", width=180)
view_toggle     = RadioButtonGroup(labels=["HIST", "CDF", "SEQ"], active=0)
function_select = Select(value=initial_func,
                          options=comparison.functions_scored or t1.functions)  # type: ignore
func_filter_btn = RadioButtonGroup(labels=["Off", "Highlight", "Isolate"], active=0)
legend_btn      = Button(label="Hide Legend", width=140)
quanta_slider   = Slider(start=50, end=500, value=100, step=50,
                          title="Quanta Resolution", width=180)
quanta_mode     = RadioButtonGroup(
                        labels=["Stable Order", "Local Order"], active=0, width=180)
subtree_mode    = RadioButtonGroup(
                        labels=["Nth", "Median", "Mean"], active=0, width=180)

occurrence_spinner = Spinner(low=0, high=0, step=1, value=0, title="Occurrence #", width=140)
occurrence_info = Div(text="", width=180, styles={"font-size": "11px", "color": "#777", "padding-top": "2px"})
components_btn = Button(label="▶ Components", width=160)

right_panel = column(children=[hist_fig])
_showing_components = False
# internal helper logic

def _update_occurrence_visibility():
    is_subtree = (mode_btns.active == 2)
    is_nth     = (subtree_mode.active == 0)
    occurrence_spinner.visible = is_subtree and is_nth
    occurrence_info.visible    = is_subtree and is_nth

def _update_occurrence_info(func: str):
    counts = timeline_obj.get_instance_counts(func)

    max_count = max(
        (c for thread_counts in counts.values() for c in thread_counts.values()),
        default=0,
    )
    occurrence_spinner.high = max(0, max_count - 1)
    if occurrence_spinner.value > occurrence_spinner.high:  # type: ignore
        occurrence_spinner.value = 0

    parts = []
    for label, thread_counts in counts.items():
        short = label.replace("Trace ", "T")
        for thread, count in thread_counts.items():
            entry = f"{thread}({short}): {count}"
            parts.append(entry)
    occurrence_info.text = " | ".join(parts) if parts else ""

def _refresh_components():
    if not _showing_components:
        return
    func     = function_select.value
    subtrees = timeline_obj.get_current_subtree_raw()
    components_obj.update(func, subtrees)


# callback logic

def on_mode_change(attr, old, new):
    global _showing_components
    mode = [Timeline.GANTT, Timeline.FLAME, Timeline.SUBTREE, Timeline.QUANTA][new]
    timeline_obj.set_mode(mode)
    is_quanta = (new == 3)
    is_subtree = (new == 2)
    quanta_slider.visible  = is_quanta
    quanta_mode.visible    = is_quanta
    subtree_mode.visible   = is_subtree
    components_btn.visible = is_subtree
    _update_occurrence_visibility()
    if not is_subtree and _showing_components:
        _showing_components = False
        right_panel.children = [hist_fig]           # type: ignore
        components_btn.label = "▶ Components"

def on_stack_mode_change(attr, old, new):
    timeline_obj.set_stack_mode("diverge" if new == 0 else "converge")

def on_controls_change(attr, old, new):
    func      = function_select.value
    view_mode = ["hist", "cdf", "seq"][view_toggle.active]  # type: ignore
    filt_mode = ["off", "highlight", "only"][func_filter_btn.active]  # type: ignore
    timeline_obj.clear_duration_filter()
    distrib_obj.update(func, mode=view_mode)
    timeline_obj.set_function_filter(func, mode=filt_mode)
    timeline_obj.set_selected_function(func)
    _update_occurrence_info(func)
    _refresh_components()

def on_threads_change(attr, old, new):
    comparison.clear_all_cache()
    t1.clear_quanta_cache()
    t2.clear_quanta_cache()
    timeline_obj.set_active_threads(new)
    stats_obj.refresh()

def on_toggle_dist():
    global dist_visible
    dist_visible = not dist_visible
    dist_col.visible = dist_visible
    new_w = TIMELINE_WIDTH if dist_visible else EXPANDED_WIDTH
    timeline_obj.set_width(new_w)
    toggle_dist_btn.label = "◀ Hide Distribution" if dist_visible else "▶ Show Distribution"

def on_bin_click(func, low_s, high_s):
    if func is None:
        timeline_obj.clear_duration_filter()
    else:
        timeline_obj.set_duration_filter(func, low_s, high_s)

def on_legend_toggle():
    timeline_obj.toggle_legend()
    legend_btn.label = "Show Legend" if legend_btn.label == "Hide Legend" else "Hide Legend"

def on_quanta_change(attr, old, new):
    timeline_obj.set_n_quanta(int(new))

def on_quanta_order_change(attr, old, new):
    timeline_obj.set_quanta_stack_order("global" if new == 0 else "local")

def on_subtree_mode_change(attr, old, new):
    modes = ["nth", "median", "mean"]
    timeline_obj.set_subtree_mode(modes[new])
    _update_occurrence_visibility()
    _refresh_components()

def on_occurrence_change(attr, old, new):
    timeline_obj.set_subtree_occurrence(int(new))
    _refresh_components()

def on_toggle_components():
    global _showing_components
    _showing_components = not _showing_components
    if _showing_components:
        right_panel.children = [components_panel]   # type: ignore
        components_btn.label = "◀ Distribution"
        _refresh_components()
    else:
        right_panel.children = [hist_fig]           # type: ignore
        components_btn.label = "▶ Components"


thread_select.on_change("value",      on_threads_change)
mode_btns.on_change("active",         on_mode_change)
stack_toggle.on_change("active",      on_stack_mode_change)
toggle_dist_btn.on_click(             on_toggle_dist)
view_toggle.on_change("active",       on_controls_change)
function_select.on_change("value",    on_controls_change)
func_filter_btn.on_change("active",   on_controls_change)
legend_btn.on_click(                  on_legend_toggle)
quanta_slider.on_change("value",      on_quanta_change)
quanta_mode.on_change("active",       on_quanta_order_change)
distrib_obj.set_on_bin_click(         on_bin_click)
subtree_mode.on_change("active",      on_subtree_mode_change)
occurrence_spinner.on_change("value", on_occurrence_change)
components_btn.on_click(on_toggle_components)

# set initial visibility

quanta_slider.visible      = False
quanta_mode.visible        = False
subtree_mode.visible       = False
occurrence_spinner.visible = False
occurrence_info.visible    = False
components_btn.visible = False

on_controls_change(None, None, None)

# layout

dist_col = column(hist_fig)


sidebar = column(
    thread_select,
    mode_btns,
    stack_toggle,
    # quanta mode
    quanta_slider,
    quanta_mode,
    # subtree mode
    subtree_mode,
    occurrence_spinner,
    components_btn,
    # occurrence_info,
    # shared
    toggle_dist_btn,
    view_toggle,
    function_select,
    func_filter_btn,
    legend_btn,
    sizing_mode="stretch_height"
)

curdoc().add_root(layout([  # type: ignore
    # main row
    [sidebar, column(timeline_fig, minimap), right_panel],
    # stats row
    [stats_table],
]))
curdoc().title = "Blup"

