from __future__ import annotations

import sys

from bokeh.io import curdoc

from trace_session import TraceSession
from controller import AppController
from utils import timed


def main():
    if len(sys.argv) < 3:
        raise SystemExit("Usage: bokeh serve --show main.py --args TRACE1 TRACE2")

    print("Reading Traces")

    t1 = TraceSession(sys.argv[1])
    t2 = TraceSession(sys.argv[2])

    print("Opening Traces")

    with timed("t1 open"):
        t1.open()
    with timed("t2 open"):
        t2.open()

    print("Traces Read")

    with timed("build"):
        controller = AppController(t1, t2)
        curdoc().add_root(controller.build())
        curdoc().title = "Blup"


main()
