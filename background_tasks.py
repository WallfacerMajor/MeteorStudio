"""Bounded, replaceable background work for MeteorStudio UI controllers."""

from __future__ import annotations

import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar


ResultT = TypeVar("ResultT")


class TaskCancelledError(RuntimeError):
    """Raised at a cooperative boundary when a background job was cancelled."""


@dataclass(eq=False)
class CancellationToken:
    """Cooperative cancellation token owned by one scheduler generation."""

    channel: str
    generation: int
    _cancelled: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> None:
        self._cancelled.set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise TaskCancelledError(f"后台任务已取消：{self.channel}")


class BackgroundTaskScheduler:
    """Run background jobs through a bounded pool and suppress stale results.

    Submitting a replacement on the same channel cancels its previous token and
    attempts to cancel work that has not started. Running jobs must check their
    token at natural boundaries; even if they finish, their callbacks are never
    delivered after a newer generation takes ownership of the channel.
    """

    def __init__(self, max_workers: int = 4, thread_name_prefix: str = "meteor-task") -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)), thread_name_prefix=thread_name_prefix
        )
        # A future may finish before add_done_callback() returns.  In that
        # case concurrent.futures invokes the callback synchronously while
        # submit() still owns this lock, so it must be re-entrant.
        self._lock = threading.RLock()
        self._generations: dict[str, int] = {}
        self._tokens: dict[str, CancellationToken] = {}
        self._current_futures: dict[str, Future] = {}
        self._all_futures: dict[Future, tuple[str, CancellationToken, bool]] = {}
        self._closed = False

    def submit(
        self,
        channel: str,
        function: Callable[[CancellationToken], ResultT],
        *,
        on_result: Callable[[ResultT], None] | None = None,
        on_error: Callable[[Exception, str], None] | None = None,
        replace: bool = True,
        retain_current: bool = True,
    ) -> CancellationToken:
        with self._lock:
            if self._closed:
                raise RuntimeError("后台任务调度器已经关闭")
            if replace:
                self._cancel_locked(channel)
            generation = self._generations.get(channel, 0) + 1
            self._generations[channel] = generation
            token = CancellationToken(channel, generation)
            self._tokens[channel] = token

            def run() -> None:
                if token.cancelled:
                    return
                try:
                    result = function(token)
                except Exception as exc:
                    details = traceback.format_exc()
                    if on_error is not None and self.is_current(token):
                        on_error(exc, details)
                    return
                if on_result is not None and self.is_current(token):
                    on_result(result)

            future = self._executor.submit(run)
            self._current_futures[channel] = future
            self._all_futures[future] = (channel, token, retain_current)
            future.add_done_callback(self._forget_future)
            return token

    def _forget_future(self, future: Future) -> None:
        with self._lock:
            record = self._all_futures.pop(future, None)
            if record is None:
                return
            channel, token, retain_current = record
            if self._current_futures.get(channel) is future:
                self._current_futures.pop(channel, None)
            if not retain_current and self._tokens.get(channel) is token:
                self._tokens.pop(channel, None)

    def _cancel_locked(self, channel: str) -> None:
        token = self._tokens.pop(channel, None)
        if token is not None:
            token.cancel()
        future = self._current_futures.pop(channel, None)
        if future is not None:
            future.cancel()

    def cancel(self, channel: str) -> None:
        with self._lock:
            self._cancel_locked(channel)

    def cancel_prefix(self, prefix: str) -> None:
        with self._lock:
            channels = [channel for channel in self._tokens if channel.startswith(prefix)]
            for channel in channels:
                self._cancel_locked(channel)

    def is_current(self, token: CancellationToken) -> bool:
        with self._lock:
            return not self._closed and self._tokens.get(token.channel) is token and not token.cancelled

    def active_count(self, prefix: str | None = None) -> int:
        with self._lock:
            return sum(
                not future.done() and (prefix is None or channel.startswith(prefix))
                for future, (channel, _token, _retain_current) in self._all_futures.items()
            )

    def tracked_channel_count(self, prefix: str | None = None) -> int:
        with self._lock:
            return sum(prefix is None or channel.startswith(prefix) for channel in self._tokens)

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def shutdown(self, wait: bool = False) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for token in self._tokens.values():
                token.cancel()
            # cancel() may synchronously invoke _forget_future and mutate this
            # dictionary. Snapshot both futures and all generation tokens.
            for future, (_channel, token, _retain) in list(self._all_futures.items()):
                token.cancel()
                future.cancel()
            self._tokens.clear()
            self._current_futures.clear()
        self._executor.shutdown(wait=wait, cancel_futures=True)
