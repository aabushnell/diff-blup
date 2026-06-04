from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict, defaultdict
from typing import Optional, Literal

import numpy as np
import pallas_trace as pallas

from utils import timed


SnapshotMode = Literal["fast", "exact"]

@dataclass(frozen=True, order=True)
class TokenRef:
    token_type:         int
    token_id:           int

    @property
    def key(self) -> str:
        return f"{self.token_type}:{self.token_id}"

@dataclass(frozen=True)
class TokenMetadata:
    ref:                TokenRef
    display_name:       str

@dataclass(frozen=True)
class ThreadMetadata:
    thread_id:          int
    thread_name:        str
    archive_id:         int
    start_ns:           int
    end_ns:             int

@dataclass(frozen=True)
class TraceMetadata:
    path:               str
    start_ns:           int
    end_ns:             int
    threads:            tuple[ThreadMetadata, ...]
    thread_name_to_id:  dict[str, int]
    thread_id_to_name:  dict[int, str]
    tokens:             tuple[TokenMetadata, ...]
    token_ref_to_meta:  dict[TokenRef, TokenMetadata]
    token_key_to_name:  dict[str, str]

@dataclass(frozen=True)
class TokenSummary:
    token_type:         int
    token_id:           int
    display_name:       str
    incl_total_ns:      int
    excl_total_ns:      int
    call_count:         int
    thread_ids:         tuple[int, ...]

@dataclass(frozen=True)
class TraceSummary:
    tokens:             tuple[TokenSummary, ...]
    top_tokens:         tuple[str, ...]

@dataclass(frozen=True)
class NodeDescriptor:
    thread_id:          int
    thread_name:        str
    token_id:           int
    iteration:          int
    depth:              int
    start_ns:           int
    end_ns:             int
    duration_ns:        int
    exc_duration_ns:    int
    function:           str

@dataclass(frozen=True)
class WindowQuery:
    thread_ids:         tuple[int, ...]
    t0_ns:              int
    t1_ns:              int
    max_depth:          int
    func_filter:        Optional[str] = None

@dataclass(frozen=True)
class QuantaBaseQuery:
    thread_ids:         tuple[int, ...]
    bin_edges_ns:       tuple[int, ...]
    mode:               SnapshotMode = "fast"

@dataclass(frozen=True)
class QuantaQuery:
    thread_ids:         tuple[int, ...]
    bin_edges_ns:       tuple[int, ...]
    mode:               SnapshotMode = "fast"
    top_k:              Optional[int] = None
    include_other:      bool = True
    function_order:     Optional[tuple[str, ...]] = None

@dataclass
class SpanBundle:
    start_ns:           np.ndarray
    end_ns:             np.ndarray
    duration_ns:        np.ndarray
    exc_duration_ns:    np.ndarray
    depth:              np.ndarray
    thread_id:          np.ndarray
    thread_name:        np.ndarray
    function:           np.ndarray
    descriptors:        list[NodeDescriptor]

@dataclass
class QuantaBundle:
    bin_start_ns:       np.ndarray
    bin_end_ns:         np.ndarray
    thread_id:          np.ndarray
    thread_name:        np.ndarray
    token_type:         np.ndarray
    token_id:           np.ndarray
    token_key:          np.ndarray
    token_display_name: np.ndarray
    exclusive_ns:       np.ndarray
    proportion:         np.ndarray

class Cache:
    def __init__(self, capacity: int = 32):
        self.capacity = capacity
        self._data = OrderedDict()

    def get(self, key):
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key, value):
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)

    def clear(self):
        self._data.clear()

def token_int_id(tok) -> int:
    try:
        return int(tok.id)
    except Exception:
        pass
    try:
        return int(tok.id.id)
    except Exception:
        pass
    return -1

def sequence_block_depth(reader: pallas.ThreadReader) -> int:
    return len([
        t for (t, _) in reader.callstack[:-1]
        if isinstance(t, pallas.Sequence)
        and t.type == pallas.SequenceType.SEQUENCE_BLOCK
    ])

def empty_span_bundle() -> SpanBundle:
    empty_i64 = np.array([], dtype=np.int64)
    empty_obj = np.array([], dtype=object)
    return SpanBundle(
        start_ns =          empty_i64,
        end_ns =            empty_i64.copy(),
        duration_ns =       empty_i64.copy(),
        exc_duration_ns =   empty_i64.copy(),
        depth =             empty_i64.copy(),
        thread_id =         empty_i64.copy(),
        thread_name =       empty_obj,
        function =          empty_obj.copy(),
        descriptors =       [],
    )

def normalize_span_rows(rows: list[NodeDescriptor]) -> SpanBundle:
    if not rows:
        return empty_span_bundle()

    start_ns = np.array([r.start_ns for r in rows], dtype=np.int64)
    end_ns = np.array([r.end_ns for r in rows], dtype=np.int64)
    duration_ns = np.array([r.duration_ns for r in rows], dtype=np.int64)
    exclusive_duration_ns = np.array([r.exc_duration_ns for r in rows], dtype=np.int64)
    depth = np.array([r.depth for r in rows], dtype=np.int64)
    thread_id = np.array([r.thread_id for r in rows], dtype=np.int64)
    thread_name = np.array([r.thread_name for r in rows], dtype=object)
    function = np.array([r.function for r in rows], dtype=object)

    order = np.lexsort((-end_ns, start_ns))
    return SpanBundle(
        start_ns =          start_ns[order],
        end_ns =            end_ns[order],
        duration_ns =       duration_ns[order],
        exc_duration_ns =   exclusive_duration_ns[order],
        depth =             depth[order],
        thread_id =         thread_id[order],
        thread_name =       thread_name[order],
        function =          function[order],
        descriptors =       [rows[i] for i in order],
    )

def empty_quanta_bundle() -> QuantaBundle:
    empty_i64 = np.array([], dtype=np.int64)
    empty_f64 = np.array([], dtype=np.float64)
    empty_obj = np.array([], dtype=object)
    return QuantaBundle(
        bin_start_ns =  empty_i64,
        bin_end_ns =    empty_i64.copy(),
        thread_id =     empty_i64.copy(),
        thread_name =   empty_obj,
        function =      empty_obj.copy(),
        exclusive_ns =  empty_i64.copy(),
        proportion =    empty_f64,
    )

def normalize_quanta_rows(
    rows: list[tuple[int, int, int, str, str, int, float]]
) -> QuantaBundle:
    if not rows:
        return empty_quanta_bundle()

    bin_start_ns = np.array([r[0] for r in rows], dtype=np.int64)
    bin_end_ns = np.array([r[1] for r in rows], dtype=np.int64)
    thread_id = np.array([r[2] for r in rows], dtype=np.int64)
    thread_name = np.array([r[3] for r in rows], dtype=object)
    function = np.array([r[4] for r in rows], dtype=object)
    exclusive_ns = np.array([r[5] for r in rows], dtype=np.int64)
    proportion = np.array([r[6] for r in rows], dtype=np.float64)

    order = np.lexsort((function, thread_name, bin_start_ns))
    return QuantaBundle(
        bin_start_ns =  bin_start_ns[order],
        bin_end_ns =    bin_end_ns[order],
        thread_id =     thread_id[order],
        thread_name =   thread_name[order],
        function =      function[order],
        exclusive_ns =  exclusive_ns[order],
        proportion =    proportion[order],
    )


class TraceSession:
    def __init__(
        self,
        path: str,
        *,
        default_summary_bins: int = 200,
        max_summary_cache: int = 16,
        max_quanta_cache: int = 32,
        max_window_cache: int = 32,
        max_occ_cache: int = 64,
        max_subtree_cache: int = 32,
    ) -> None:
        self.path = path
        self.default_summary_bins = int(default_summary_bins)

        self._trace: Optional[pallas.Trace] = None
        self._meta: Optional[TraceMetadata] = None
        self._summary: Optional[TraceSummary] = None

        self._summary_cache = Cache(max_summary_cache)
        self._quanta_cache = Cache(max_quanta_cache)
        self._window_cache = Cache(max_window_cache)
        self._occ_cache = Cache(max_occ_cache)
        self._subtree_cache = Cache(max_subtree_cache)

    # ---------- lifecycle ----------

    def open(self) -> None:
        if self._trace is not None:
            return
        with timed("pallas.open_trace"):
            self._trace = pallas.open_trace(self.path)
        with timed("build_metadata"):
            self._meta = self.build_metadata()
        with timed("summarize_functions"):
            self._summary = self.summarize_tokens(mode="fast", top_k=32)

    def close(self) -> None:
        self._trace = None
        self._meta = None
        self._summary = None
        self.clear_caches()

    def clear_caches(self) -> None:
        self._summary_cache.clear()
        self._quanta_cache.clear()
        self._window_cache.clear()
        self._occ_cache.clear()
        self._subtree_cache.clear()

    # ---------- metadata ----------

    @property
    def meta(self) -> TraceMetadata:
        if self._meta is None:
            self.open()
        assert self._meta is not None
        return self._meta

    @property
    def summary(self) -> TraceSummary:
        if self._summary is None:
            self.open()
        assert self._summary is not None
        return self._summary

    def build_metadata(self) -> TraceMetadata:
        assert self._trace is not None

        threads: list[ThreadMetadata] = []
        for archive in self._trace.archives:
            for thread in archive.threads:
                thread_name = str(self._trace.locations[thread.id].name)
                threads.append(
                    ThreadMetadata(
                        thread_id =     int(thread.id),
                        thread_name =   thread_name,
                        archive_id =    int(archive.id),
                        start_ns =      int(thread.starting_timestamp),
                        end_ns =        int(thread.finish_timestamp),
                    )
                )

        threads.sort(key=lambda t: t.thread_name)
        start_ns = min(t.start_ns for t in threads) if threads else 0
        end_ns = max(t.end_ns for t in threads) if threads else 0

        token_res = self._trace.tokens()
        tokens = tuple(
            TokenMetadata(
                ref =           TokenRef(int(ttype), int(tid)),
                display_name =  str(name),
            )
            for ttype, tid, name in zip(token_res.token_type, token_res.token_id, token_res.display_name)
        )

        return TraceMetadata(
            path =              self.path,
            start_ns =          start_ns,
            end_ns =            end_ns,
            threads =           tuple(threads),
            thread_name_to_id = {
                t.thread_name: t.thread_id for t in threads
            },
            thread_id_to_name = {
                t.thread_id: t.thread_name for t in threads
            },
            tokens =            tokens,
            token_ref_to_meta = {
                tm.ref: tm for tm in tokens
            },
            token_key_to_name = {
                tm.ref.key: tm.display_name for tm in tokens
            },
        )

    def summarize_tokens(
        self,
        *,
        mode: SnapshotMode = "fast",
        top_k: int = 32,
    ) -> TraceSummary:
        cache_key = ("summary_tokens", mode, int(top_k))
        cached = self._summary_cache.get(cache_key)
        if cached is not None:
            return cached

        if self._trace is None:
            self.open()
        assert self._trace is not None

        incl_totals: dict[tuple[int, int], int] = defaultdict(int)
        excl_totals: dict[tuple[int, int], int] = defaultdict(int)
        call_counts: dict[tuple[int, int], int] = defaultdict(int)
        thread_ids_by_token: dict[tuple[int, int], set[int]] = defaultdict(set)

        for archive in self._trace.archives:
            for thread in archive.threads:
                tid = int(thread.id)
                for seq in thread.sequences:
                    token = seq.id
                    key = (int(token.type), int(token.id))
                    n_iter = int(seq.n_iterations)
                    if n_iter <= 0:
                        continue

                    call_counts[key] += n_iter
                    thread_ids_by_token[key].add(tid)

                    if mode == "fast":
                        incl_totals[key] += int(seq.mean_duration) * n_iter
                        excl_totals[key] += int(seq.mean_exclusive_duration) * n_iter
                    else:
                        incl_totals[key] += int(seq.durations.as_numpy_array().sum())
                        excl_totals[key] += int(seq.exclusive_durations.as_numpy_array().sum())

        keys = set(incl_totals) | set(excl_totals) | set(call_counts)
        rows = [
            TokenSummary(
                token_type=token_type,
                token_id=token_id,
                display_name=self.token_display_name(token_type, token_id),
                incl_total_ns=int(incl_totals.get((token_type, token_id), 0)),
                excl_total_ns=int(excl_totals.get((token_type, token_id), 0)),
                call_count=int(call_counts.get((token_type, token_id), 0)),
                thread_ids=tuple(sorted(thread_ids_by_token.get((token_type, token_id), set()))),
            )
            for (token_type, token_id) in keys
        ]

        rows.sort(
            key=lambda r: (-r.excl_total_ns, -r.incl_total_ns, r.display_name, r.token_type, r.token_id)
        )

        top_tokens = tuple(self.token_key(r.token_type, r.token_id) for r in rows[:top_k])
        summary = TraceSummary(tokens=tuple(rows), top_tokens=top_tokens)
        self._summary_cache.put(cache_key, summary)
        return summary

    # ---------- queries ----------

    def compute_quanta_base(self, query: QuantaBaseQuery) -> QuantaBundle:
        query = self.canonicalize_quanta_base_query(query)

        cache_key = ("quanta_base", query)
        cached = self._quanta_cache.get(cache_key)
        if cached is not None:
            return cached

        if self._trace is None:
            self.open()
        assert self._trace is not None

        rows: list[tuple[int, int, int, str, str, int, float]] = []
        wanted_thread_ids = set(int(t) for t in query.thread_ids)
        bin_edges = np.asarray(query.bin_edges_ns, dtype=np.int64)

        for archive in self._trace.archives:
            for thread in archive.threads:
                tid = int(thread.id)
                if tid not in wanted_thread_ids:
                    continue

                tname = self.meta.thread_id_to_name[tid]

                for i in range(len(bin_edges) - 1):
                    b0 = int(bin_edges[i])
                    b1 = int(bin_edges[i + 1])
                    if b1 <= b0:
                        continue

                    if query.mode == "fast":
                        snap = thread.getSnapshotViewFast(b0, b1)
                    else:
                        raise NotImplementedError
                        # snap = thread.getSnapshotViewByName(b0, b1)

                    by_name: dict[str, int] = defaultdict(int)
                    for key, dur in snap.items():
                        if isinstance(key, tuple):
                            name = str(key[1])
                        else:
                            name = str(key)
                        by_name[name] += int(dur)

                    total = sum(by_name.values())
                    if total <= 0:
                        continue

                    for func, exc_ns in by_name.items():
                        rows.append((
                            b0,
                            b1,
                            tid,
                            tname,
                            func,
                            int(exc_ns),
                            float(exc_ns / total),
                        ))

        bundle = normalize_quanta_rows(rows)
        self._quanta_cache.put(cache_key, bundle)
        return bundle

    def compute_quanta(self, query: QuantaQuery) -> QuantaBundle:
        query = self.canonicalize_quanta_query(query)

        cache_key = ("quanta_view", query)
        cached = self._quanta_cache.get(cache_key)
        if cached is not None:
            return cached

        base = self.compute_quanta_base(QuantaBaseQuery(
            thread_ids=query.thread_ids,
            bin_edges_ns=query.bin_edges_ns,
            mode=query.mode,
        ))

        bundle = self.reduce_quanta_bundle(
            base,
            top_k=query.top_k,
            include_other=query.include_other,
            function_order=query.function_order,
        )

        self._quanta_cache.put(cache_key, bundle)
        return bundle

    def iter_block_sequence_occurrences(
        self,
        *,
        wanted_thread_ids: Optional[set[int]] = None,
        t0_ns: Optional[int] = None,
        t1_ns: Optional[int] = None,
        max_depth: Optional[int] = None,
        func_filter: Optional[str] = None,
    ):
        if self._trace is None:
            self.open()
        assert self._trace is not None

        for archive in self._trace.archives:
            for thread in archive.threads:
                tid = int(thread.id)
                if wanted_thread_ids is not None and tid not in wanted_thread_ids:
                    continue

                thread_name = str(self.meta.thread_id_to_name[tid])
                reader = thread.reader()

                while not reader.isEndOfTrace():
                    token, iteration = reader.pollCurToken()
                    depth = sequence_block_depth(reader)

                    if max_depth is not None and depth > max_depth:
                        while reader.exitIfEndOfBlock(True, True):
                            pass
                        reader.moveToNextToken(False, False)
                        continue

                    if not isinstance(token, pallas.Sequence):
                        reader.moveToNextToken(True, True)
                        continue

                    reader.moveToNextToken(True, False)

                    if token.type != pallas.SequenceType.SEQUENCE_BLOCK:
                        reader.moveToNextToken(True, True)
                        continue

                    start_ns = int(token.timestamps[iteration])
                    duration_ns = int(token.durations[iteration])
                    exclusive_duration_ns = int(token.exclusive_durations[iteration])
                    end_ns = start_ns + duration_ns
                    function = str(token.guessName())

                    if t0_ns is not None and end_ns < t0_ns:
                        reader.moveToNextToken(True, True)
                        continue
                    if t1_ns is not None and start_ns > t1_ns:
                        reader.moveToNextToken(True, True)
                        continue
                    if func_filter is not None and function != func_filter:
                        reader.moveToNextToken(True, True)
                        continue

                    yield NodeDescriptor(
                        thread_id =         tid,
                        thread_name =       thread_name,
                        token_id =          token_int_id(token),
                        iteration =         int(iteration),
                        depth =             int(depth),
                        start_ns =          start_ns,
                        end_ns =            end_ns,
                        duration_ns =       duration_ns,
                        exc_duration_ns =   exclusive_duration_ns,
                        function =          function,
                    )

                    reader.moveToNextToken(True, True)

    def query_function_occurrences(
        self,
        func: str,
        thread_id: Optional[int] = None,
        *,
        max_depth: Optional[int] = None,
    ) -> list[NodeDescriptor]:
        key = ("occ", func, thread_id, max_depth)
        cached = self._occ_cache.get(key)
        if cached is not None:
            return cached

        wanted = None if thread_id is None else {int(thread_id)}
        rows = list(self.iter_block_sequence_occurrences(
            wanted_thread_ids =     wanted,
            max_depth =             max_depth,
            func_filter =           func,
        ))

        rows.sort(key=lambda r: (r.thread_name, r.start_ns, -r.end_ns, r.iteration))
        self._occ_cache.put(key, rows)
        return rows

    def get_instance_count(self, func: str, thread_id: int) -> int:
        return len(self.query_function_occurrences(func, thread_id))

    def get_nth_occurrence(
        self,
        func: str,
        thread_id: int,
        n: int,
    ) -> Optional[NodeDescriptor]:
        occs = self.query_function_occurrences(func, thread_id)
        if n < 0 or n >= len(occs):
            return None
        return occs[n]

    def get_median_occurrence(
        self,
        func: str,
        thread_id: int,
    ) -> Optional[NodeDescriptor]:
        occs = self.query_function_occurrences(func, thread_id)
        if not occs:
            return None
        durations = np.array([o.duration_ns for o in occs], dtype=np.int64)
        median = np.median(durations)
        idx = int(np.argmin(np.abs(durations - median)))
        return occs[idx]

    def query_window(self, query: WindowQuery) -> SpanBundle:
        cached = self._window_cache.get(query)
        if cached is not None:
            return cached

        rows = list(self.iter_block_sequence_occurrences(
            wanted_thread_ids =     set(int(t) for t in query.thread_ids),
            t0_ns =                 int(query.t0_ns),
            t1_ns =                 int(query.t1_ns),
            max_depth =             int(query.max_depth),
            func_filter =           query.func_filter,
        ))

        bundle = normalize_span_rows(rows)
        self._window_cache.put(query, bundle)
        return bundle

    def query_subtree(
        self,
        root: NodeDescriptor,
        *,
        max_depth: Optional[int] = None,
        normalize_time: bool = False,
    ) -> SpanBundle:
        key = ("subtree", root, max_depth, normalize_time)
        cached = self._subtree_cache.get(key)
        if cached is not None:
            return cached

        if max_depth is None:
            effective_max_depth = 10**9
        else:
            effective_max_depth = root.depth + int(max_depth)

        base = self.query_window(WindowQuery(
            thread_ids=(root.thread_id,),
            t0_ns=root.start_ns,
            t1_ns=root.end_ns,
            max_depth=effective_max_depth,
            func_filter=None,
        ))

        rows: list[NodeDescriptor] = []
        for d in base.descriptors:
            if d.thread_id != root.thread_id:
                continue
            if d.start_ns < root.start_ns:
                continue
            if d.end_ns > root.end_ns:
                continue
            if max_depth is not None and d.depth > root.depth + max_depth:
                continue

            start_ns = d.start_ns - root.start_ns if normalize_time else d.start_ns
            end_ns = d.end_ns - root.start_ns if normalize_time else d.end_ns

            rows.append(NodeDescriptor(
                thread_id=d.thread_id,
                thread_name=d.thread_name,
                token_id=d.token_id,
                iteration=d.iteration,
                depth=d.depth - root.depth,
                start_ns=start_ns,
                end_ns=end_ns,
                duration_ns=d.duration_ns,
                exc_duration_ns=d.exc_duration_ns,
                function=d.function,
            ))

        bundle = normalize_span_rows(rows)
        self._subtree_cache.put(key, bundle)
        return bundle

    # ------- helpers -------

    def token_key(self, token_type: int, token_id: int) -> str:
        return f"{token_type}:{token_id}"

    def token_display_name(self, token_type: int, token_id: int) -> str:
        ref = TokenRef(int(token_type), int(token_id))
        meta = self.meta.token_ref_to_meta.get(ref)
        if meta is not None:
            return meta.display_name
        return f"<token {ref.key}>"


    def canonicalize_quanta_base_query(self, query: QuantaBaseQuery) -> QuantaBaseQuery:
        return QuantaBaseQuery(
            thread_ids=tuple(sorted(int(t) for t in query.thread_ids)),
            bin_edges_ns=tuple(int(x) for x in query.bin_edges_ns),
            mode=query.mode,
        )

    def canonicalize_quanta_query(self, query: QuantaQuery) -> QuantaQuery:
        return QuantaQuery(
            thread_ids=tuple(sorted(int(t) for t in query.thread_ids)),
            bin_edges_ns=tuple(int(x) for x in query.bin_edges_ns),
            mode=query.mode,
            top_k=None if query.top_k is None else int(query.top_k),
            include_other=bool(query.include_other),
            function_order=None if query.function_order is None else tuple(str(f) for f in query.function_order),
        )

    def group_quanta_rows(
        self,
        bundle: QuantaBundle,
    ) -> dict[tuple[int, int, int, str], list[tuple[str, int]]]:
        grouped: dict[tuple[int, int, int, str], list[tuple[str, int]]] = defaultdict(list)
        for b0, b1, tid, tname, func, exc_ns in zip(
            bundle.bin_start_ns,
            bundle.bin_end_ns,
            bundle.thread_id,
            bundle.thread_name,
            bundle.function,
            bundle.exclusive_ns,
        ):
            key = (int(b0), int(b1), int(tid), str(tname))
            grouped[key].append((str(func), int(exc_ns)))
        return grouped

    def global_quanta_function_totals(
        self,
        bundle: QuantaBundle,
    ) -> dict[str, int]:
        totals: dict[str, int] = defaultdict(int)
        for func, exc_ns in zip(bundle.function, bundle.exclusive_ns):
            totals[str(func)] += int(exc_ns)
        return totals

    def reduce_quanta_bundle(
        self,
        bundle: QuantaBundle,
        *,
        top_k: Optional[int],
        include_other: bool,
        function_order: Optional[tuple[str, ...]] = None,
    ) -> QuantaBundle:
        if top_k is None:
            return bundle

        if top_k <= 0:
            return empty_quanta_bundle()

        grouped = self.group_quanta_rows(bundle)

        if function_order is not None:
            rank = {str(f): i for i, f in enumerate(function_order)}
            sort_key_fn = lambda item: (rank.get(item[0], len(rank)), -item[1], item[0])
        else:
            totals = self.global_quanta_function_totals(bundle)
            sort_key_fn = lambda item: (-totals.get(item[0], 0), item[0])

        rows: list[tuple[int, int, int, str, str, int, float]] = []

        for (b0, b1, tid, tname), items in grouped.items():
            total = sum(exc_ns for _, exc_ns in items)
            if total <= 0:
                continue

            items_sorted = sorted(items, key=sort_key_fn)
            kept = items_sorted[:top_k]
            other_sum = sum(exc_ns for _, exc_ns in items_sorted[top_k:])

            if include_other and other_sum > 0:
                kept = kept + [("OTHER", other_sum)]

            for func, exc_ns in kept:
                rows.append((
                    int(b0),
                    int(b1),
                    int(tid),
                    str(tname),
                    str(func),
                    int(exc_ns),
                    float(exc_ns / total),
                ))

        return normalize_quanta_rows(rows)

