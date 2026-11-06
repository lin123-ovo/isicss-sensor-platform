"""Session-owned background search with cooperative cancellation."""
from __future__ import annotations
import logging
from contextvars import copy_context
from threading import Event, Lock, Thread
from typing import Any
from .regional_weight_search import SearchCancelled

logger = logging.getLogger(__name__)


class RegionalSearchJob:
    """Run a workflow search without calling Streamlit from the worker."""
    def __init__(self, workflow: Any, frame: Any, context: Any, source_sha: str,
                 strategy: str, split: Any, fingerprint: str) -> None:
        self.fingerprint = fingerprint
        self._cancel = Event()
        self._lock = Lock()
        self._state: dict[str, Any] = {'status': 'running', 'phase': '准备', 'done': 0, 'total': 1}
        def progress(phase: str, done: int, total: int) -> None:
            with self._lock:
                self._state.update(phase=phase, done=done, total=total)
        def run() -> None:
            try:
                result = workflow.search_regional_weights(frame, context, source_sha, strategy, split,
                                                          progress=progress, cancelled=self._cancel.is_set)
                if self._cancel.is_set(): raise SearchCancelled('已取消比例搜索。')
                with self._lock: self._state.update(status='complete', result=result)
            except SearchCancelled:
                with self._lock: self._state.update(status='cancelled', message='比例搜索已取消。')
            except Exception as error:
                logger.exception('Regional search failed')
                with self._lock: self._state.update(status='failed', message=f'比例搜索失败：{error}')
        worker_context = copy_context()
        self._thread = Thread(target=worker_context.run, args=(run,), name='regional-weight-search', daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

    def snapshot(self) -> dict[str, Any]:
        with self._lock: return self._state.copy()
