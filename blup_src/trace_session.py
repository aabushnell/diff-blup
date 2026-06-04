from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import pallas_trace as pallas

from data_model import *
from utils import timed

def sequence_block_depth(reader: pallas.ThreadReader) -> int:
    return len([
        t for (t, _) in reader.callstack[:-1]
        if isinstance(t, pallas.Sequence)
        and t.type == pallas.SequenceType.SEQUENCE_BLOCK
    ])

class TraceSession:
    def __init__(
        self,
        path: str,
        *,
        max_summary_cache: int = 16,
        max_quanta_cache: int = 32,
        max_span_cache: int = 32,
        max_occ_cache: int = 64,
        max_subtree_cache: int = 32,
    ) -> None:
        self.path = path

        self._trace: Optional[pallas.Trace] = None
        self._meta: Optional[TraceInfo] = None
        self._summary: Optional[TraceSummary] = None

        self._summary_cache = DataCache(max_summary_cache)
        self._quanta_cache = DataCache(max_quanta_cache)
        self._span_cache = DataCache(max_span_cache)
        self._occ_cache = DataCache(max_occ_cache)
        self._subtree_cache = DataCache(max_subtree_cache)

    # ---------- lifecycle ----------

    def open(self) -> None:
        if self._trace is not None:
            return

        with timed("pallas.open_trace"):
            self._trace = pallas.open_trace(self.path)

        with timed("build_metadata"):
            self._meta = self.build_metadata()

        assert self._meta is not None
        with timed("validate_trace"):
            self.validate_trace(self._meta)

        summary_query = SummaryQuery(fidelity="fast", top_k=32)
        with timed("summarize_functions"):
            self._summary = self.summarize_tokens(summary_query)

    def close(self) -> None:
        self._trace = None
        self._meta = None
        self._summary = None
        self.clear_caches()

    def clear_caches(self) -> None:
        self._summary_cache.clear()
        self._quanta_cache.clear()
        self._span_cache.clear()
        self._subtree_cache.clear()

    # ---------- metadata ----------

    @property
    def meta(self) -> TraceInfo:
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

    def build_metadata(self) -> TraceInfo:
        assert self._trace is not None

        thread_rows: list[tuple[int, str, int, int]] = []
        for archive in self._trace.archives:
            for thread in archive.threads:
                tid = int(thread.id)
                tname = str(self._trace.locations[thread.id].name)
                t0 = int(thread.starting_timestamp)
                t1 = int(thread.finish_timestamp)
                thread_rows.append((tid, tname, t0, t1))

        thread_rows.sort(key=lambda x: x[1])
        thread_ids = np.array([r[0] for r in thread_rows], dtype=np.int64)
        thread_names = np.array([r[1] for r in thread_rows], dtype=object)

        token_res = self._trace.tokens()
        token_types = np.asarray(token_res.token_type, dtype=np.uint8)
        token_ids = np.asarray(token_res.token_id, dtype=np.int64)
        token_names = np.asarray([str(x) for x in token_res.display_name], dtype=object)

        return TraceInfo(
            path            = self.path,
            start_ns        = min(r[2] for r in thread_rows),
            end_ns          = max(r[3] for r in thread_rows),
            thread_ids      = thread_ids,
            thread_names    = thread_names,
            token_types     = token_types,
            token_ids       = token_ids,
            token_names     = token_names,
            thread_name_to_id = {
                str(name): int(tid)
                for tid, name in zip(thread_ids, thread_names)
            },
            thread_id_to_name = {
                int(tid): str(name)
                for tid, name in zip(thread_ids, thread_names)
            },
            token_key_to_name = {
                as_token_key(int(ttype), int(tid)): str(name)
                for ttype, tid, name in zip(token_types, token_ids, token_names)
            },
        )

    def validate_trace(self, meta: TraceInfo) -> None:
        thread_ids = [int(x) for x in meta.thread_ids]
        thread_names = [str(x) for x in meta.thread_names]
        token_keys = [
            as_token_key(int(ttype), int(tid))
            for ttype, tid in zip(meta.token_types, meta.token_ids)
        ]
        token_names = [str(x) for x in meta.token_names]

        def search_dupl_items(xs):
            counts = Counter(xs)
            return [x for x, n in counts.items() if n > 1]

        dup_thread_ids = search_dupl_items(thread_ids)
        if dup_thread_ids:
            raise RuntimeError(
                f"trace validation failed: duplicate thread ids: {dup_thread_ids[:8]}"
            )

        dup_thread_names = search_dupl_items(thread_names)
        if dup_thread_names:
            raise RuntimeError(
                f"trace validation failed: duplicate thread names: {dup_thread_names[:8]}"
            )

        dup_token_keys = search_dupl_items(token_keys)
        if dup_token_keys:
            raise RuntimeError(
                f"trace validation failed: duplicate token keys: {dup_token_keys[:8]}"
            )

        if len(meta.thread_name_to_id) != len(thread_ids):
            raise RuntimeError("trace validation failed: thread_name_to_id is non-injective")

        if len(meta.thread_id_to_name) != len(thread_ids):
            raise RuntimeError("trace validation failed: thread_id_to_name is non-injective")

        if len(meta.token_key_to_name) != len(token_keys):
            raise RuntimeError("trace validation failed: token_key_to_name is non-injective")

        for key, name in zip(token_keys, token_names):
            mapped = meta.token_key_to_name.get(key)
            if mapped != name:
                raise RuntimeError(
                    f"trace validation failed: inconsistent token name for key {key!r}: "
                    f"{mapped!r} != {name!r}"
                )

    def summarize_tokens(self, query: SummaryQuery) -> TraceSummary:
        cache_key = ("summary_tokens", query)
        cached = self._summary_cache.get(cache_key)
        if cached is not None:
            return cached

        if self._trace is None:
            self.open()
        assert self._trace is not None

        incl_totals: dict[tuple[int, int], int] = defaultdict(int)
        excl_totals: dict[tuple[int, int], int] = defaultdict(int)
        call_counts: dict[tuple[int, int], int] = defaultdict(int)
        tids_by_token: dict[tuple[int, int], set[int]] = defaultdict(set)

        for archive in self._trace.archives:
            for thread in archive.threads:
                tid = int(thread.id)
                for seq in thread.sequences:
                    token = seq.id
                    token_type = int(token.type)
                    token_id = int(token.id)
                    key = (token_type, token_id)

                    n_iter = int(seq.n_iterations)
                    if n_iter <= 0:
                        continue

                    call_counts[key] += n_iter
                    tids_by_token[key].add(tid)

                    if query.fidelity == "fast":
                        incl_totals[key] += int(seq.mean_duration) * n_iter
                        excl_totals[key] += int(seq.mean_exclusive_duration) * n_iter
                    else:
                        incl_totals[key] += int(seq.durations.as_numpy_array().sum())
                        excl_totals[key] += int(seq.exclusive_durations.as_numpy_array().sum())

        keys = set(incl_totals) | set(excl_totals) | set(call_counts)
        rows = [
            TokenSummary(
                token_type      = token_type,
                token_id        = token_id,
                incl_total_ns   = int(incl_totals.get((token_type, token_id), 0)),
                excl_total_ns   = int(excl_totals.get((token_type, token_id), 0)),
                call_count      = int(call_counts.get((token_type, token_id), 0)),
                thread_ids      = tuple(sorted(tids_by_token.get((token_type, token_id), set()))),
            )
            for (token_type, token_id) in keys
        ]
        rows.sort(
            key=lambda r: (-r.excl_total_ns, -r.incl_total_ns, r.call_count, r.token_type, r.token_id)
        )
        top_tokens = tuple(as_token_key(r.token_type, r.token_id) for r in rows[: query.top_k])

        summary = TraceSummary(
            fidelity    = query.fidelity,
            tokens      = tuple(rows),
            top_tokens  = top_tokens,
        )
        self._summary_cache.put(cache_key, summary)
        return summary

    # ---------- queries ----------

    def query_quanta(self, query: QuantaQuery) -> QuantaBundle:
        query = canonicalize_quanta_query(query)

        cache_key = ("quanta", query)
        cached = self._quanta_cache.get(cache_key)
        if cached is not None:
            return cached

        if self._trace is None:
            self.open()
        assert self._trace is not None

        if query.fidelity != "fast":
            raise NotImplementedError("query_quanta: only fidelity='fast' is supported currently")

        tids = np.asarray(query.thread_ids, dtype=np.uint32)
        bins = np.asarray(query.bin_edges_ns, dtype=np.uint64)

        if tids.size == 0 or bins.size < 2:
            bundle = empty_quanta_bundle(query.fidelity)
            self._quanta_cache.put(cache_key, bundle)
            return bundle

        with timed("trace.calc_quanta_base"):
            raw = self._trace.calc_quanta_base(tids, bins, query.fidelity)

        with timed("normalize_quanta_result"):
            bundle = normalize_quanta_result(raw, query.fidelity)

        with timed("limit_quanta_top_k"):
            if query.top_k is not None:
                bundle = self.limit_quanta_top_k(bundle, query.top_k)

        self._quanta_cache.put(cache_key, bundle)
        return bundle

    def query_spans(self, query: SpanQuery) -> SpanBundle:
        query = canonicalize_span_query(query)

        cache_key = ("spans", query)
        cached = self._span_cache.get(cache_key)
        if cached is not None:
            return cached

        if self._trace is None:
            self.open()
        assert self._trace is not None

        req_thread_ids = set(query.thread_ids)
        token_filter = query.token

        rows: list[tuple[int, int, int, int, int, int, int, int, int]] = []
        for archive in self._trace.archives:
            for thread in archive.threads:
                tid = int(thread.id)
                if tid not in req_thread_ids:
                    continue

                reader = thread.reader()

                while not reader.isEndOfTrace():
                    token, iteration = reader.pollCurToken()
                    depth = sequence_block_depth(reader)

                    if query.max_depth is not None and depth > query.max_depth:
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

                    token_type = int(token.type)
                    token_id = token_int_id(token)

                    if token_filter is not None and (token_type, token_id) != token_filter:
                        reader.moveToNextToken(True, True)
                        continue

                    start_ns = int(token.timestamps[iteration])
                    dur_ns = int(token.durations[iteration])
                    excl_ns = int(token.exclusive_durations[iteration])
                    end_ns = start_ns + dur_ns

                    if end_ns < query.t0_ns:
                        reader.moveToNextToken(True, True)
                        continue
                    if start_ns > query.t1_ns:
                        reader.moveToNextToken(True, True)
                        continue

                    rows.append((
                        tid,
                        token_type,
                        token_id,
                        int(iteration),
                        int(depth),
                        start_ns,
                        end_ns,
                        dur_ns,
                        excl_ns,
                    ))

                    reader.moveToNextToken(True, True)

        bundle = normalize_span_rows(rows, fidelity="exact")
        self._span_cache.put(cache_key, bundle)
        return bundle

    def query_occurrences(self, query: OccurrenceQuery) -> tuple[NodeRef, ...]:
        query = canonicalize_occurrence_query(query)

        cache_key = ("occ", query)
        cached = self._occ_cache.get(cache_key)
        if cached is not None:
            return cached

        if self._trace is None:
            self.open()
        assert self._trace is not None

        if query.fidelity != "exact":
            raise NotImplementedError("query_occurrences: only fidelity='exact' is supported currently")

        wanted_thread_ids = set(query.thread_ids)
        want_token = query.token

        rows: list[NodeRef] = []
        for archive in self._trace.archives:
            for thread in archive.threads:
                tid = int(thread.id)
                if tid not in wanted_thread_ids:
                    continue

                reader = thread.reader()

                while not reader.isEndOfTrace():
                    token, iteration = reader.pollCurToken()
                    depth = sequence_block_depth(reader)

                    if query.max_depth is not None and depth > query.max_depth:
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

                    token_type = int(token.type)
                    token_id = token_int_id(token)

                    if (token_type, token_id) != want_token:
                        reader.moveToNextToken(True, True)
                        continue

                    start_ns = int(token.timestamps[iteration])
                    dur_ns = int(token.durations[iteration])
                    end_ns = start_ns + dur_ns

                    if query.t0_ns is not None and end_ns < query.t0_ns:
                        reader.moveToNextToken(True, True)
                        continue
                    if query.t1_ns is not None and start_ns > query.t1_ns:
                        reader.moveToNextToken(True, True)
                        continue

                    rows.append(
                        NodeRef(
                            thread_id   = tid,
                            token_type  = token_type,
                            token_id    = token_id,
                            iteration   = int(iteration),
                            depth       = int(depth),
                            start_ns    = start_ns,
                            end_ns      = end_ns,
                        )
                    )

                    reader.moveToNextToken(True, True)

        rows.sort(key=lambda r: (r.thread_id, r.start_ns, -r.end_ns, r.iteration))

        if query.mode == "all":
            out = tuple(rows)
        elif query.mode == "first":
            out = tuple(rows[:1])
        elif query.mode == "last":
            out = tuple(rows[-1:]) if rows else ()
        elif query.mode == "nth":
            if query.index is None:
                raise ValueError("query_occurrences(mode='nth') requires index")
            out = (rows[query.index],) if 0 <= query.index < len(rows) else ()
        elif query.mode == "median":
            if not rows:
                out = ()
            else:
                durations = np.array([r.end_ns - r.start_ns for r in rows], dtype=np.int64)
                median = np.median(durations)
                idx = int(np.argmin(np.abs(durations - median)))
                out = (rows[idx],)
        else:
            raise ValueError(f"unknown occurrence mode: {query.mode}")

        self._occ_cache.put(cache_key, out)
        return out

    def query_subtree(self, query: SubtreeQuery) -> SpanBundle:
        query = canonicalize_subtree_query(query)

        cache_key = ("subtree", query)
        cached = self._subtree_cache.get(cache_key)
        if cached is not None:
            return cached

        root = query.root
        if query.max_depth is None:
            span_max_depth = None
        else:
            span_max_depth = root.depth + query.max_depth

        base = self.query_spans(
            SpanQuery(
                thread_ids  = (root.thread_id,),
                t0_ns       = root.start_ns,
                t1_ns       = root.end_ns,
                fidelity    = query.fidelity,
                max_depth   = span_max_depth,
                token       = None,
            )
        )

        keep = (
            (base.thread_id == root.thread_id) &
            (base.start_ns >= root.start_ns) &
            (base.end_ns <= root.end_ns)
        )

        if query.max_depth is not None:
            keep &= (base.depth <= root.depth + query.max_depth)

        idx = np.nonzero(keep)[0]
        if idx.size == 0:
            bundle = empty_span_bundle(base.fidelity)
            self._subtree_cache.put(cache_key, bundle)
            return bundle

        bundle = subset_span_bundle(base, idx, fidelity=base.fidelity)

        bundle.depth = bundle.depth - int(root.depth)
        if query.normalize_time:
            bundle.start_ns = bundle.start_ns - int(root.start_ns)
            bundle.end_ns = bundle.end_ns - int(root.start_ns)

        self._subtree_cache.put(cache_key, bundle)
        return bundle

    # ------- helpers -------

    def limit_quanta_top_k(self, bundle: QuantaBundle, top_k: int) -> QuantaBundle:
        if top_k <= 0:
            return empty_quanta_bundle(bundle.fidelity)

        totals: dict[tuple[int, int], int] = defaultdict(int)
        for ttype, tid, excl in zip(bundle.token_type, bundle.token_id, bundle.excl_ns):
            totals[(int(ttype), int(tid))] += int(excl)

        groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for i, (b0, b1, tid) in enumerate(zip(bundle.start_ns, bundle.end_ns, bundle.thread_id)):
            groups[(int(b0), int(b1), int(tid))].append(i)

        keep: list[int] = []
        for _, idxs in groups.items():
            idxs.sort(
                key=lambda i: (
                    -totals[(int(bundle.token_type[i]), int(bundle.token_id[i]))],
                    -int(bundle.excl_ns[i]),
                    int(bundle.token_type[i]),
                    int(bundle.token_id[i]),
                )
            )
            keep.extend(idxs[:top_k])

        if not keep:
            return empty_quanta_bundle(bundle.fidelity)

        keep_arr = np.array(sorted(keep), dtype=np.int64)
        return subset_quanta_bundle(bundle, keep_arr)
