"""Bounded process-local reuse of pure derived frames; never a data store."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from threading import Lock

import pandas as pd

from .cleaning import SensorCleaningPipeline


class CleanedFrameCache:
    """Reuse exact inputs across UI readers and return independent frame copies.

    Callers must read and validate current source rows and batch options first.
    No database access or cleaning runs under the cache lock. Concurrent misses
    may compute twice; failed computations never enter the cache.
    """

    def __init__(self) -> None:
        self._frames: OrderedDict[str, tuple[pd.DataFrame, int]] = OrderedDict()
        self._lock = Lock()
        self._max_entries = 2
        self._max_bytes = 64 * 1024 * 1024
        self._size_bytes = 0

    def load(
        self, namespace: str, pipeline: SensorCleaningPipeline, prepared: pd.DataFrame
    ) -> pd.DataFrame:
        """Clean or copy a result keyed by all values, options, policy and profile."""
        # Keep the customer/legacy workflow on its existing uncached path.
        if pipeline.processing_version == "legacy":
            return pipeline.clean(prepared).frame.copy()
        header = json.dumps(
            [namespace, pipeline.processing_sha256,
             list(prepared.columns), [str(dtype) for dtype in prepared.dtypes]],
            ensure_ascii=False,
        ).encode("utf-8")
        digest = hashlib.sha256(header)
        digest.update(pd.util.hash_pandas_object(prepared, index=True).values.tobytes())
        key = digest.hexdigest()
        with self._lock:
            cached = self._frames.get(key)
            if cached is not None:
                self._frames.move_to_end(key)
                return cached[0].copy(deep=True)

        frame = pipeline.clean(prepared).frame.copy(deep=True)
        size = int(frame.memory_usage(index=True, deep=True).sum())
        if size <= self._max_bytes:
            with self._lock:
                previous = self._frames.pop(key, None)
                if previous is not None:
                    self._size_bytes -= previous[1]
                self._frames[key] = (frame, size)
                self._size_bytes += size
                self._frames.move_to_end(key)
                while (
                    len(self._frames) > self._max_entries
                    or self._size_bytes > self._max_bytes
                ):
                    _, (_, removed_size) = self._frames.popitem(last=False)
                    self._size_bytes -= removed_size
                return frame.copy(deep=True)
        return frame.copy(deep=True)


cleaned_frame_cache = CleanedFrameCache()
