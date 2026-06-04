import datetime
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy

from blup_utils import natural_keys, choose_palette, apply_top_bottom

def create_empty_df() -> pd.DataFrame:
    df = pd.DataFrame({
        "thread":   pd.Series(dtype="str"),
        "function": pd.Series(dtype="str"),
        "start":    pd.Series(dtype="timedelta64[ns]"),
        "finish":   pd.Series(dtype="timedelta64[ns]"),
        "duration": pd.Series(dtype="int64"),
        "depth":    pd.Series(dtype="int")
    })
    return df

def create_empty_quanta_df() -> pd.DataFrame:
    df = pd.DataFrame({
        "thread":       pd.Series(dtype="str"),
        "quanta_idx":   pd.Series(dtype="int"),
        "quanta_start": pd.Series(dtype="timedelta64[ns]"),
        "quanta_end":   pd.Series(dtype="timedelta64[ns]"),
        "function":     pd.Series(dtype="str"),
        "proportion":   pd.Series(dtype="float64"),
        "exclusive_s":  pd.Series(dtype="float64"), 
    })
    return df

def compute_depth(df):
    if(max(df["depth"])>0):
        return df
    t1=datetime.datetime.now()

    threads = sorted(df["thread"].unique())

    sequences_depth = [float("nan")] * len(df)

    for thread in threads:
        filtered_df=df.loc[df["thread"]==thread]
        stack = []

        df_indices, start_ts, finish_ts = (
            list(filtered_df.index),
            list(filtered_df["start"]),
            list(filtered_df["finish"]),
        )

        stack.append((df_indices[0], start_ts[0], finish_ts[0]))

        for i in range(1, len(filtered_df)):
            curr_df_index, curr_start_ts, curr_finish_ts = (
                df_indices[i],
                start_ts[i],
                finish_ts[i],
            )

            # Search for a sequence whose finish_timestamp ends after curr_start_ts
            # This sequence is the one that called the current sequence
            i = len(stack)-1
            stack_df_index, stack_start_ts, stack_finish_ts = stack[i]
            while(i > -1 and curr_start_ts >= stack_finish_ts):
                  i -= 1
                  stack_df_index, stack_start_ts, stack_finish_ts = stack[i]
                  del stack[i+1]

            stack.append((curr_df_index, curr_start_ts, curr_finish_ts))
            sequences_depth[curr_df_index] = len(stack)

    df["depth"] = sequences_depth
    t2=datetime.datetime.now()
    d=t2-t1
    print("Compute depth took "+str(d))
    return df

def process_trace(df) -> tuple[pd.DataFrame, list, list]:
    threads   = sorted(df["thread"].unique(), key=natural_keys)
    functions = sorted(df["function"].unique(), key=natural_keys)

    df["start"]    = df["start"].astype("timedelta64[ns]")
    df["finish"]   = df["finish"].astype("timedelta64[ns]")
    df["duration"] = pd.to_timedelta(df["duration"])

    df = df.sort_values(["start", "finish"], ascending=[True, False])
    df = df.reset_index(drop=True)

    df = compute_depth(df)
    return df, threads, functions

def read_trace_pallas(filename: str, max_depth: int) -> pd.DataFrame:
    import pallas_trace as pallas

    trace: pallas.Trace = pallas.open_trace(filename)
    sequences = []

    for archive in trace.archives:
        for thread in archive.threads:
            thread_name = trace.locations[thread.id].name
            thread_reader: pallas.ThreadReader = thread.reader()

            while not thread_reader.isEndOfTrace():
                (token, iteration) = thread_reader.pollCurToken()

                # Compute current call depth from the callstack
                depth = len([
                    t for (t, _) in thread_reader.callstack[:-1]
                    if isinstance(t, pallas.Sequence)
                    and t.type == pallas.SequenceType.SEQUENCE_BLOCK
                ])

                if depth > max_depth:
                    while thread_reader.exitIfEndOfBlock(True, True):
                        pass
                    thread_reader.moveToNextToken(False, False)
                    continue

                if not isinstance(token, pallas.Sequence):
                    thread_reader.moveToNextToken(True, True)
                    continue

                thread_reader.moveToNextToken(True, False)

                if token.type == pallas.SequenceType.SEQUENCE_BLOCK:
                    s = {}
                    s["start"]    = token.timestamps[iteration]
                    s["duration"] = token.durations[iteration]
                    s["finish"]   = s["start"] + s["duration"]
                    s["thread"]   = thread_name
                    s["function"] = token.guessName()
                    s["depth"]    = depth
                    sequences.append(s)

                thread_reader.moveToNextToken(True, True)

    df = pd.DataFrame(sequences)
    empty_df = create_empty_df()
    df = pd.concat([empty_df, df]).fillna(0)
    df = df.astype(empty_df.dtypes)
    return df

def read_trace_otf2(trace_name) -> pd.DataFrame:
    import otf2
    sequences=[]
    ongoing_sequences={}

    with otf2.reader.open(trace_name) as trace:
        for location, event in trace.events:
            if isinstance(event, otf2.events.Enter):
                s = {}
                s["start"]=event.time
                s["duration"]=0
                s["finish"]=0
                s["thread"]=location.name
                s["function"]=event.region.name

                if (not location.name in ongoing_sequences) or (len(ongoing_sequences[location.name])==0):
                    ongoing_sequences[location.name]=[s]
                    s["depth"]=0
                else:
                    s["depth"]=ongoing_sequences[location.name][-1]["depth"]+1
                    ongoing_sequences[location.name].append(s)
            elif isinstance(event, otf2.events.Leave):
                if ongoing_sequences.get(location.name):
                    s = ongoing_sequences[location.name][-1]
                    del ongoing_sequences[location.name][-1]
                    s["finish"]   = event.time
                    s["duration"] = s["finish"] - s["start"]
                    sequences.append(s)

    df=pd.DataFrame(sequences)
    empty_df=create_empty_df()
    df=pd.concat([empty_df, df]).fillna(0)

    expected_dtypes=empty_df.dtypes
    df = df.astype(expected_dtypes)

    return df

def read_trace(file_path, max_depth=100):
    t1 = datetime.datetime.now()
    file_name, file_extension = os.path.splitext(file_path)

    df = pd.DataFrame()
    if file_extension == ".csv":
        raise NotImplementedError
        # df = read_trace_csv(file_path)
    elif file_extension == ".pallas":
        df = read_trace_pallas(file_path, max_depth)
    elif file_extension == ".otf2":
        df = read_trace_otf2(file_path)
    else:
        raise NotImplementedError

    t2 = datetime.datetime.now()
    d = t2 - t1
    print(f"Trace loaded in {d} seconds")

    return df

def build_quanta_df(
    df: pd.DataFrame,
    active_threads: list[str],
    t_max: pd.Timedelta,
    n_quanta: int,
) -> pd.DataFrame:
    t_min  = pd.Timedelta(0)
    q_dur  = (t_max - t_min) / n_quanta
    rows: list[dict] = []

    for thread in active_threads:
        tdf = df[df["thread"] == thread]
        if tdf.empty:
            continue

        for q in range(n_quanta):
            qs = t_min + q * q_dur
            qe = qs + q_dur

            overlap = tdf[(tdf["start"] < qe) & (tdf["finish"] > qs)].copy()
            if overlap.empty:
                continue

            overlap["eff_start"]  = overlap["start"].clip(lower=qs,  upper=qe) # type: ignore
            overlap["eff_finish"] = overlap["finish"].clip(lower=qs, upper=qe) # type: ignore
            overlap = overlap.dropna(subset=["depth"])

            events: list[tuple] = []
            for _, row in overlap.iterrows():
                d = int(row["depth"])
                events.append((row["eff_start"],  1, "enter", row["function"], d))
                events.append((row["eff_finish"], 0, "leave", row["function"], d))
            events.sort(key=lambda e: (e[0], e[1]))

            active:    dict[int, str]    = {}
            func_time: dict[str, float]  = defaultdict(float)
            prev_t = qs

            for t, _, typ, func, depth in events:
                if t > prev_t and active:
                    top_func = active[max(active)]
                    func_time[top_func] += (t - prev_t) / pd.Timedelta("1s")
                if typ == "enter":
                    active[depth] = func
                else:
                    active.pop(depth, None)
                prev_t = t

            total = sum(func_time.values())
            if total == 0:
                continue

            for func, exc_s in func_time.items():
                rows.append({
                    "thread":       thread,
                    "quanta_idx":   q,
                    "quanta_start": qs,
                    "quanta_end":   qe,
                    "function":     func,
                    "proportion":   exc_s / total,
                    "exclusive_s":  exc_s,
                })

    return pd.DataFrame(rows) if rows else create_empty_quanta_df()

class BlupTrace:
    df: pd.DataFrame = create_empty_df()
    threads: list    = []
    functions: list  = []

    def __init__(self, file_path=None):
        if (file_path is not None):
            df_read = read_trace(file_path)
            self.df, self.threads, self.functions = (
                process_trace(df_read)
            )

            counts = (
                self.df.groupby(["function", "thread"])
                .size()
                .reset_index(name="count")
                .sort_values(["function", "thread"])
            )
            total_per_func = counts.groupby("function")["count"].sum()
            print(f"\n[blup] Trace loaded: {file_path}")
            print(f"[blup] {len(self.functions)} functions, {len(self.threads)} threads, "
                f"{len(self.df)} total call records\n")
            print(f"{'Function':<40} {'Total':>8}  per-thread breakdown")
            print("-" * 72)
            for func in sorted(self.functions):
                thread_counts = counts[counts["function"] == func]
                breakdown = "  ".join(
                    f"{row['thread']}:{row['count']}"
                    for _, row in thread_counts.iterrows()
                )
                print(f"{func:<40} {int(total_per_func[func]):>8}  {breakdown}")
            print()

            self._func_stats_cache: dict[str, dict] = {}
            self._mean_subtree_cache: dict[tuple[str, str], pd.DataFrame] = {}
            self._quanta_cache: dict[tuple, pd.DataFrame] = {}

    def get_durations(self, func: str) -> np.ndarray:
        mask = self.df["function"] == func
        return (
            self.df.loc[mask, "duration"]
                .to_numpy(dtype=np.float64) / 1e9
        )

    def get_calls_in_duration_range(
            self, func: str, low_s: float, high_s: float
    ) -> list[int]:
        low_td  = pd.Timedelta(low_s,  unit="s")
        high_td = pd.Timedelta(high_s, unit="s")
        mask = (
            (self.df["function"] == func) &
            (self.df["duration"] >= low_td) &
            (self.df["duration"] <= high_td)
        )
        return list(self.df[mask].index)

    def get_call_instances(self, func: str, thread: str) -> list[int]:
        mask = (
            (self.df["function"] == func) &
            (self.df["thread"] == thread)
        )
        return list(self.df[mask].sort_values("start").index) # type: ignore

    def get_instance_count(self, func: str, thread: str) -> int:
        return len(self.get_call_instances(func, thread))

    def get_median_instance(self, func: str, thread: str) -> int | None:
        instances = self.get_call_instances(func, thread)
        if not instances:
            return None
        durations = self.df.loc[instances, "duration"].to_numpy(dtype=np.float64)
        median_dur = np.median(durations)
        return instances[int(np.argmin(np.abs(durations - median_dur)))]

    def get_call_subtree(self, idx: int) -> pd.DataFrame:
        row = self.df.loc[idx]
        mask = (
            (self.df["thread"] == row["thread"]) &
            (self.df["start"]  >= row["start"])  &
            (self.df["finish"] <= row["finish"])
        )
        sub = self.df[mask].copy()
        sub["depth"] = sub["depth"] - int(row["depth"]) # type: ignore
        return sub

    def build_subtree_mask(self, root_indices: list[int]) -> pd.Series:
        mask = pd.Series(False, index=self.df.index)
        for idx in root_indices:
            row = self.df.loc[idx]
            submask = (
                (self.df["thread"]  == row["thread"]) &
                (self.df["start"]   >= row["start"])  &
                (self.df["finish"]  <= row["finish"])
            )
            mask = mask | submask
        return mask

    def build_mean_subtree(self, func: str, thread: str) -> pd.DataFrame:
        cache_key = (func, thread)
        if cache_key in self._mean_subtree_cache:
            return self._mean_subtree_cache[cache_key]

        instances = self.get_call_instances(func, thread)
        if not instances:
            return pd.DataFrame()

        root_durations_ns: list[int] = []
        node_props: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
        path_to_func:  dict[tuple, str]            = {}
        path_to_depth: dict[tuple, int]            = {}
        parent_of:     dict[tuple, tuple | None]   = {}

        for idx in instances:
            root_row = self.df.loc[idx]
            root_dur_ns = root_row["duration"].value
            if root_dur_ns <= 0:
                continue
            root_durations_ns.append(root_dur_ns)

            sub = self.get_call_subtree(idx).copy()
            root_start_val = sub["start"].min().value
            sub["_s"] = sub["start"].apply(lambda x: x.value) - root_start_val
            sub["_f"] = sub["finish"].apply(lambda x: x.value) - root_start_val
            sub["_d"] = sub["duration"].apply(lambda x: x.value)
            sub = sub.sort_values("_s").reset_index(drop=True)

            stack: list[tuple[tuple, int]] = []   # (path, depth)
            row_path: list[tuple | None] = [None] * len(sub)

            for i, row in sub.iterrows():
                dep = int(row["depth"])
                fn  = str(row["function"])
                while stack and stack[-1][1] >= dep:
                    stack.pop()
                parent_path: tuple | None = stack[-1][0] if stack else None
                path = parent_path + (fn,) if parent_path is not None else (fn,)
                row_path[i] = path        # type: ignore
                path_to_func[path] = fn
                path_to_depth[path] = dep
                parent_of[path] = parent_path
                stack.append((path, dep))

            path_first_start: dict[tuple, int] = {}
            path_total_dur:   dict[tuple, int] = {}

            for i, row in sub.iterrows():
                path = row_path[i]  # type: ignore
                if path is None:
                    continue
                s = int(row["_s"])
                dur = int(row["_d"])
                if path not in path_first_start or s < path_first_start[path]:
                    path_first_start[path] = s
                path_total_dur[path] = path_total_dur.get(path, 0) + dur

            seen: set[tuple] = set()
            for path, total_dur in path_total_dur.items():
                parent = parent_of.get(path)
                if parent is None:
                    node_props[path].append((0.0, 1.0))
                else:
                    parent_dur = path_total_dur.get(parent, 0)
                    if parent_dur <= 0:
                        node_props[path].append((0.0, 0.0))
                    else:
                        parent_start  = path_first_start.get(parent, 0)
                        child_start   = path_first_start[path]
                        offset_prop   = (child_start - parent_start) / parent_dur
                        dur_prop      = total_dur / parent_dur
                        node_props[path].append((offset_prop, dur_prop))
                seen.add(path)

            for path in node_props:
                if path not in seen:
                    node_props[path].append((0.0, 0.0))

        if not root_durations_ns:
            return pd.DataFrame()

        n_valid = len(root_durations_ns)
        mean_root_dur_ns = int(np.mean(root_durations_ns))

        for path in node_props:
            while len(node_props[path]) < n_valid:
                node_props[path].append((0.0, 0.0))

        abs_pos: dict[tuple, tuple[int, int]] = {}
        rows_out: list[dict] = []
        max_dep = max(path_to_depth.values(), default=0)

        for depth in range(max_dep + 1):
            for path, dep in path_to_depth.items():
                if dep != depth:
                    continue
                parent = parent_of[path]
                props  = node_props.get(path, [])

                if depth == 0:
                    abs_start  = 0
                    abs_finish = mean_root_dur_ns
                    frequency  = 1.0
                else:
                    if parent not in abs_pos:
                        continue
                    par_start, par_finish = abs_pos[parent]
                    par_dur = par_finish - par_start
                    if par_dur <= 0:
                        continue

                    mean_offset = float(np.mean([o for o, _dur in props]))
                    mean_dur    = float(np.mean([_dur for _o, _dur in props]))
                    frequency   = sum(1 for _o, _dur in props if _dur > 0) / n_valid

                    if mean_dur <= 0:
                        continue

                    abs_start  = par_start + int(mean_offset * par_dur)
                    abs_finish = abs_start + int(mean_dur    * par_dur)

                abs_pos[path] = (abs_start, abs_finish)
                rows_out.append({
                    "function":  path_to_func[path],
                    "thread":    thread,
                    "depth":     depth,
                    "start":     abs_start,
                    "finish":    abs_finish,
                    "duration":  abs_finish - abs_start,
                    "frequency": frequency,
                })

        if not rows_out:
            return pd.DataFrame()

        result = pd.DataFrame(rows_out)
        result["start"]     = result["start"].astype("int64")
        result["finish"]    = result["finish"].astype("int64")
        result["duration"]  = result["duration"].astype("int64")
        result["depth"]     = result["depth"].astype("int")
        result["frequency"] = result["frequency"].astype("float64")

        self._mean_subtree_cache[cache_key] = result
        return result

    def get_quanta_df(
        self,
        active_threads: list[str],
        t_max: pd.Timedelta,
        n_quanta: int,
    ) -> pd.DataFrame:
        key = (tuple(active_threads), t_max.value, n_quanta)
        if key not in self._quanta_cache:
            origin = self.df["start"].min()
            df_rel = self.df.copy()
            df_rel["start"]  = self.df["start"]  - origin
            df_rel["finish"] = self.df["finish"]  - origin
            self._quanta_cache[key] = build_quanta_df(
                df_rel, active_threads, t_max, n_quanta
            )
        return self._quanta_cache[key]

    def get_function_stats(self, func: str) -> dict | None:
        if func in self._func_stats_cache:
            return self._func_stats_cache[func]

        d = self.get_durations(func)
        if len(d) < 2:
            return None

        x_min = max(d.min(), 1e-12)
        stats = {
            "n":          len(d),
            "mean":       d.mean(),
            "std":        d.std(ddof=1),
            "p90":        np.percentile(d, 90),
            "skew":       float(scipy.stats.skew(d)),
            "kurtosis":   float(scipy.stats.kurtosis(d)),
            "contention": float(np.sum((d - x_min) / x_min)),
            "time_total": d.sum(),
        }
        self._func_stats_cache[func] = stats
        return stats

    def clear_stats_cache(self):
        self._func_stats_cache.clear()

    def clear_mean_subtree_cache(self):
        self._mean_subtree_cache.clear()

    def clear_quanta_cache(self):
        self._quanta_cache.clear()

    def clear_all_caches(self):
        self.clear_stats_cache()
        self.clear_mean_subtree_cache()
        self.clear_quanta_cache()

class TraceComparison:

    def __init__(self, t1: BlupTrace, t2: BlupTrace, min_calls: int = 5):
        self.t1 = t1
        self.t2 = t2
        self.min_calls = min_calls
        self._diff_cache: dict[str, dict]  = {}   # func -> diff stats dict
        self._score_df:   pd.DataFrame | None = None

    def get_diff_stats(self, func: str) -> dict | None:
        if func in self._diff_cache:
            return self._diff_cache[func]

        s1 = self.t1.get_function_stats(func)
        s2 = self.t2.get_function_stats(func)
        if s1 is None or s2 is None:
            return None
        if s1["n"] < self.min_calls or s2["n"] < self.min_calls:
            return None

        d1 = self.t1.get_durations(func)
        d2 = self.t2.get_durations(func)

        result = {
            "function":            func,
            **{f"{k}1": v for k, v in s1.items()},
            **{f"{k}2": v for k, v in s2.items()},
            "diff_mean_abs":       s2["mean"]       - s1["mean"],
            "diff_mean_rel":      (s2["mean"]       - s1["mean"])       / max(s1["mean"],       1e-12),
            "diff_p90_abs":        s2["p90"]        - s1["p90"],
            "diff_p90_rel":       (s2["p90"]        - s1["p90"])        / max(s1["p90"],        1e-12),
            "diff_contention_rel":(s2["contention"] - s1["contention"]) / max(s1["contention"], 1e-12),
            "ks":         scipy.stats.ks_2samp(d1, d2).statistic,
            "wasserstein":scipy.stats.wasserstein_distance(d1, d2),
        }
        self._diff_cache[func] = result
        return result

    @property
    def score_df(self) -> pd.DataFrame:
        if self._score_df is None:
            self._score_df = self._compute_score_df()
        return self._score_df

    @property
    def functions_scored(self) -> list[str]:
        df = self.score_df
        return list(df["function"]) if not df.empty else []

    def clear_all_cache(self):
        self._diff_cache.clear()
        self._score_df = None
        self.t1.clear_stats_cache()
        self.t2.clear_stats_cache()

    def _compute_score_df(self, top_n: int = 30) -> pd.DataFrame:
        all_functions = sorted(set(self.t1.functions) | set(self.t2.functions))
        rows = [self.get_diff_stats(f) for f in all_functions]
        rows = [r for r in rows if r is not None]
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # scoring weights
        df["base_score"] = (
            0.5 * df["diff_mean_rel"].abs() +
            0.5 * df["diff_contention_rel"].abs()
        )
        grand_total = (df["time_total1"] + df["time_total2"]).sum() + 1e-12
        df["time_frac"] = (df["time_total1"] + df["time_total2"]) / grand_total
        df["score"]     = df["base_score"] * df["time_frac"]

        return df.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

def prepare_display_df(
    t1: BlupTrace,
    t2: BlupTrace,
    active_threads: list[str] | None = None,
    func_filter: str | None = None,
    func_filter_mode: str = "off",   # "off" | "highlight" | "only"
    stack_mode: str = "diverge",
    duration_filter: tuple[str, float, float] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    all_threads   = sorted(set(t1.threads)   | set(t2.threads),   key=natural_keys)
    all_functions = sorted(set(t1.functions) | set(t2.functions), key=natural_keys)

    used_palette = choose_palette(all_functions)
    color_map = {f: used_palette[i % len(used_palette)]
                 for i, f in enumerate(all_functions)}

    if active_threads is None:
        active_threads = all_threads

    depth_step = 0.1
    max_depth = max(
        t1.df[t1.df["thread"].isin(active_threads)]["depth"].max(),
        t2.df[t2.df["thread"].isin(active_threads)]["depth"].max(),
    )
    max_depth = int(max_depth) if not pd.isna(max_depth) else 0

    results = []
    for raw_df, label, trace_obj in [(t1.df, "Trace 1", t1), (t2.df, "Trace 2", t2)]:
        df = raw_df[raw_df["thread"].isin(active_threads)].copy()
        df["trace"] = label

        # translate all timestamps to align to zero
        origin = df["start"].min()
        df["start"]    = pd.to_timedelta(df["start"]    - origin, unit="ns")
        df["finish"]   = pd.to_timedelta(df["finish"]   - origin, unit="ns")
        df["duration"] = pd.to_timedelta(df["duration"],          unit="ns")

        # set function color palette
        df["color"] = df["function"].map(color_map)

        # set Y-center coords (categorical position)
        df["center"] = df["thread"].apply(
            lambda t: len(active_threads) - 0.5 - active_threads.index(t)
        )

        # add top/bottom coords for flame graph (depth already computed in BlupTrace)
        apply_top_bottom(df, label, stack_mode, depth_step, max_depth)

        # set alpha / function filter
        df["alpha"] = 1.0
        if func_filter_mode != "off" and func_filter is not None:
            is_func = df["function"] == func_filter
            if func_filter_mode == "highlight":
                df["alpha"] = 0.05
                df.loc[is_func, "alpha"] = 1.0
            elif func_filter_mode == "only":
                df["alpha"] = 0.0
                df.loc[is_func, "alpha"] = 1.0

        if duration_filter is not None:
            dur_func, low_s, high_s = duration_filter
            root_idxs    = trace_obj.get_calls_in_duration_range(dur_func, low_s, high_s)
            visible_mask = trace_obj.build_subtree_mask(root_idxs)
            df["alpha"] = visible_mask.reindex(df.index, fill_value=False).map(
                {True: 1.0, False: 0.0}
            )


        df = df.sort_values(["start", "finish"], ascending=[True, False])
        df = df.reset_index(drop=True)
        results.append(df)

    return results[0], results[1], all_threads, all_functions

def prepare_quanta_display_df(
    t1: BlupTrace,
    t2: BlupTrace,
    active_threads: list[str],
    n_quanta: int = 200,
    stack_order: str = "global"   # "global" | "local"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_functions = sorted(set(t1.functions) | set(t2.functions), key=natural_keys)
    palette       = choose_palette(all_functions)
    color_map     = {f: palette[i % len(palette)] for i, f in enumerate(all_functions)}

    t_max = max(
        (t1.df["finish"] - t1.df["start"].min()).max(),
        (t2.df["finish"] - t2.df["start"].min()).max(),
    )

    results = []
    for trace_obj, label in [(t1, "Trace 1"), (t2, "Trace 2")]:
        prop_df = trace_obj.get_quanta_df(active_threads, t_max, n_quanta)
        if prop_df.empty:
            results.append(pd.DataFrame())
            continue

        if stack_order == "global":
            global_totals = (
                prop_df.groupby("function")["exclusive_s"]
                       .sum()
                       .sort_values(ascending=False)
            )
            func_rank = {f: i for i, f in enumerate(global_totals.index)}
        else:
            func_rank = None

        HALF    = 0.45
        PADDING = 0.02
        quads:  list[dict] = []

        for (thread, q_idx), grp in prop_df.groupby(["thread", "quanta_idx"]):
            center  = len(active_threads) - 0.5 - active_threads.index(thread) # type: ignore
            qs      = grp["quanta_start"].iloc[0]
            qe      = grp["quanta_end"].iloc[0]

            if func_rank is not None:
                grp = grp.sort_values(
                    "function",
                    key=lambda s: s.map(lambda f: func_rank.get(f, len(func_rank))) # type: ignore
                )
            else:
                grp = grp.sort_values("proportion", ascending=False)

            cumsum  = 0.0
            for _, row in grp.iterrows():
                p = row["proportion"]
                if label == "Trace 1":
                    top, bottom = (center - PADDING - cumsum * HALF,
                                   center - PADDING - (cumsum + p) * HALF)
                else:
                    bottom, top = (center + PADDING + cumsum * HALF,
                                   center + PADDING + (cumsum + p) * HALF)
                quads.append({
                    "left":        qs,
                    "right":       qe,
                    "top":         top,
                    "bottom":      bottom,
                    "color":       color_map[row["function"]],
                    "function":    row["function"],
                    "thread":      thread,
                    "proportion":  p,
                    "exclusive_s": row["exclusive_s"],
                })
                cumsum += p

        results.append(pd.DataFrame(quads))

    return results[0], results[1]

