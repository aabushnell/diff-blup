import sys

from bokeh.io import curdoc
from bokeh.layouts import layout, column
from bokeh.models import Select, RadioButtonGroup, MultiSelect, Button, Slider

import blup_trace as bt
from blup_utils import natural_keys

from blup_distribution import Distribution
from blup_stats import Stats
from blup_timeline import Timeline

TIMELINE_WIDTH   = 1200
DIST_WIDTH       = 800
EXPANDED_WIDTH   = TIMELINE_WIDTH + DIST_WIDTH
dist_visible = True

# load both traces
t1 = bt.BlupTrace(sys.argv[1])
t2 = bt.BlupTrace(sys.argv[2])
comparison = bt.TraceComparison(t1, t2)

# init panel objects
timeline_obj  = Timeline(t1, t2, width=1200, height=1000)
distrib_obj   = Distribution(t1, t2, width=800,  height=1000)
stats_obj     = Stats(comparison)

timeline_fig, minimap  = timeline_obj.build()
hist_fig               = distrib_obj.build()
stats_table            = stats_obj.build()

# init ui elements
initial_func    = stats_obj.functions_scored[0] if stats_obj.functions_scored else t1.functions[0]
all_threads     = sorted(set(t1.threads) | set(t2.threads), key=natural_keys)

thread_select   = MultiSelect(value=all_threads, options=all_threads)
mode_btns       = RadioButtonGroup(labels=["Gantt", "Flame", "Subtree", "Quanta"], active=0)
stack_toggle    = RadioButtonGroup(labels=["Diverge", "Converge"], active=0)
toggle_dist_btn = Button(label="◀ Hide Distribution", width=180)
view_toggle     = RadioButtonGroup(labels=["Histogram", "CDF"], active=0)
function_select = Select(value=initial_func,
                          options=comparison.functions_scored or t1.functions)  # type: ignore
func_filter_btn = RadioButtonGroup(labels=["Off", "Highlight", "Isolate"], active=0)
legend_btn      = Button(label="Hide Legend", width=140)
quanta_slider   = Slider(start=50, end=500, value=100, step=50,
                          title="Quanta Resolution", width=180)
quanta_toggle   = RadioButtonGroup(
                        labels=["Stable Order", "Local Order"], active=0, width=180)

# callback logic

def on_mode_change(attr, old, new):
    mode = [Timeline.GANTT, Timeline.FLAME, Timeline.SUBTREE, Timeline.QUANTA][new]
    timeline_obj.set_mode(mode)
    is_quanta = (new == 3)
    quanta_slider.visible = is_quanta
    quanta_toggle.visible = is_quanta

def on_stack_mode_change(attr, old, new):
    timeline_obj.set_stack_mode("diverge" if new == 0 else "converge")

def on_controls_change(attr, old, new):
    func      = function_select.value
    view_mode = "hist" if view_toggle.active == 0 else "cdf"
    filt_mode = ["off", "highlight", "only"][func_filter_btn.active]  # type: ignore
    timeline_obj.clear_duration_filter()
    distrib_obj.update(func, mode=view_mode)
    timeline_obj.set_function_filter(func, mode=filt_mode)
    timeline_obj.set_selected_function(func)

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

thread_select.on_change("value",    on_threads_change)
mode_btns.on_change("active",       on_mode_change)
stack_toggle.on_change("active",    on_stack_mode_change)
toggle_dist_btn.on_click(           on_toggle_dist)
view_toggle.on_change("active",     on_controls_change)
function_select.on_change("value",  on_controls_change)
func_filter_btn.on_change("active", on_controls_change)
legend_btn.on_click(                on_legend_toggle)
quanta_slider.on_change("value",    on_quanta_change)
quanta_toggle.on_change("active",   on_quanta_order_change)
distrib_obj.set_on_bin_click(       on_bin_click)

quanta_slider.visible = False
on_controls_change(None, None, None)

# layout

dist_col = column(hist_fig)

sidebar = column(
    thread_select,
    mode_btns,
    stack_toggle,
    quanta_slider,
    quanta_toggle,
    toggle_dist_btn,
    view_toggle,
    function_select,
    func_filter_btn,
    legend_btn,
    sizing_mode="stretch_height"
)

curdoc().add_root(layout([  # type: ignore
    # main row
    [sidebar, column(timeline_fig, minimap), dist_col],
    # stats row
    [stats_table],
]))
curdoc().title = "Blup"

