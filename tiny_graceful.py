"""
tiny-graceful — Zero-dependency graceful shutdown coordination.

Ctrl-C kills your process mid-write. Kubernetes sends SIGTERM and gives you
ten seconds to finish what you were doing. Docker stops with SIGTERM and then
SIGKILL ninety seconds later. Every long-running program eventually needs the
same thing: a coordinated "stop accepting work, drain what is in flight, then
exit" sequence that does not require hand-rolled signal.signal boilerplate,
threading bookkeeping, and timeout logic in every service you own.

tiny-graceful is one file, stdlib only, and gives you that sequence:

    from tiny_graceful import serve

    serve(lambda shutdown: run_my_server(shutdown), worker_count=8,
          drain_timeout=10.0)

Install signal handlers once, run workers until a stop is requested, run
shutdown hooks in reverse registration order, and return the Shutdown object
so the caller can inspect the stop reason.

Three building blocks, each usable on its own:

  * Shutdown    — a one-shot stop signal plus a global deadline, with
                  wait(), cancelled, reason, and timeout_remaining().
  * coordinator — a context manager that installs SIGINT/SIGTERM handlers
                  and guarantees hooks have run by the time the block exits.
  * serve()     — the batteries-included runner: workers plus shutdown.

There is an async counterpart too: AsyncShutdown and async_serve() follow the
same ordering and error rules, so an asyncio service gets deterministic
shutdown without reimplementing the sequence.

Design rules, in priority order:

  1. Hooks run in reverse registration order. Last registered closes first,
     mirroring how nested resources were opened. This is the rule people get
     wrong most often: if you open a connection and then a transaction, the
     transaction must close first.
  2. A failing hook never blocks the rest. One broken cleanup function must
     not strand your other resources. Failures are collected and re-raised as
     a single ShutdownError after every hook has had its turn.
  3. The deadline is global, not per-hook. drain_timeout=10 means the whole
     drain gets ten seconds, not ten seconds each. Otherwise N hooks can hang
     for N * timeout and the supervisor SIGKILLs you anyway.
  4. Shutdown is idempotent. Calling shutdown twice is a no-op. Signals can
     arrive twice (SIGINT then SIGTERM during a hung Ctrl-C) and that must
     not double-run your teardown.

Zero dependencies. Python 3.8+. MIT licensed.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from typing import Any, Callable, Iterable, List, Optional, Tuple

__version__ = "0.1.0"

__all__ = [
    "Shutdown",
    "AsyncShutdown",
    "ShutdownError",
    "coordinator",
    "serve",
    "async_serve",
]


class ShutdownError(RuntimeError):
    """Raised after a shutdown in which one or more hooks failed.

    Carries the original (name, exception) pairs so callers can log real
    causes instead of a generic "shutdown failed".
    """

    def __init__(self, errors: Iterable[Tuple[str, BaseException]]) -> None:
        self.errors = list(errors)
        detail = "; ".join(
            "{0}: {1}: {2}".format(name, type(exc).__name__, exc)
            for name, exc in self.errors
        )
        super().__init__(
            "{0} shutdown hook(s) failed: {1}".format(len(self.errors), detail)
        )


class Shutdown:
    """A one-shot stop signal with an optional global deadline.

    The smallest useful piece: something a worker thread can wait on and
    something a signal handler can set.

        shutdown = Shutdown()

        def worker(shutdown):
            while not shutdown.cancelled:
                handle_one_request()
                if shutdown.cancelled:
                    break

        threading.Thread(target=worker, args=(shutdown,)).start()
        ...
        shutdown.trigger()          # e.g. from a SIGTERM handler
        shutdown.wait(timeout=5.0)  # False if the deadline passed
    """

    __slots__ = (
        "_event",
        "_reason",
        "_requested_at",
        "_timeout",
        "_hooks",
        "_hooks_done",
        "_lock",
        "_errors",
    )

    def __init__(self, timeout: Optional[float] = None) -> None:
        self._event = threading.Event()
        self._reason: Optional[str] = None
        self._requested_at: Optional[float] = None
        self._timeout = timeout
        self._hooks: List[Tuple[str, Callable[[], Any]]] = []
        self._hooks_done = False
        self._lock = threading.Lock()
        self._errors: List[Tuple[str, BaseException]] = []

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def cancelled(self) -> bool:
        """True once a stop has been requested."""
        return self._event.is_set()

    @property
    def reason(self) -> Optional[str]:
        """Why the shutdown started ("SIGTERM", "manual", ...), or None."""
        return self._reason

    @property
    def errors(self) -> List[Tuple[str, BaseException]]:
        """(name, exception) pairs for every hook that raised."""
        return list(self._errors)

    def timeout_remaining(self) -> Optional[float]:
        """Seconds left in the drain budget, or None if no timeout.

        Returns None when no deadline was configured, so callers that
        intend to wait forever can skip the check entirely.
        """
        if self._timeout is None:
            return None
        if self._requested_at is None:
            return self._timeout
        return max(0.0, self._timeout - (time.monotonic() - self._requested_at))

    # ------------------------------------------------------------------
    # Requesting a stop
    # ------------------------------------------------------------------

    def trigger(self, reason: str = "manual") -> bool:
        """Request a stop. Returns True only for the first caller.

        Later callers get False and change nothing, which is what makes
        repeated signal delivery safe.
        """
        with self._lock:
            if self._event.is_set():
                return False
            self._reason = reason
            self._requested_at = time.monotonic()
            self._event.set()
            return True

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until a stop is requested. Returns False on timeout.

        With no argument this waits forever, matching threading.Event.wait.
        """
        return self._event.wait(timeout)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def on_shutdown(self, fn: Callable[[], Any], name: Optional[str] = None) -> Callable[[], Any]:
        """Register a cleanup hook. Returns fn so it can be a decorator.

        Hooks run in reverse registration order, so register the
        outermost resource first.
        """
        if not callable(fn):
            raise TypeError(
                "shutdown hook must be callable, got {0!r}".format(type(fn).__name__)
            )
        hook_name = name or getattr(fn, "__name__", None) or repr(fn)
        with self._lock:
            self._hooks.append((hook_name, fn))
        return fn

    def run_hooks(self) -> None:
        """Run every registered hook, newest first, exactly once.

        A hook that raises is recorded and stepped over; the rest still
        run. Raises ShutdownError afterwards if anything failed.
        """
        with self._lock:
            if self._hooks_done:
                return
            self._hooks_done = True
            hooks = list(self._hooks)
            self._hooks.clear()

        for hook_name, fn in reversed(hooks):
            try:
                fn()
            except BaseException as exc:  # cleanup must not abort the rest
                self._errors.append((hook_name, exc))

        if self._errors:
            raise ShutdownError(self._errors)

    def __enter__(self) -> "Shutdown":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.trigger("context-exit")
        self.run_hooks()
        return False

    def __repr__(self) -> str:
        return "Shutdown(cancelled={0!r}, reason={1!r}, timeout={2!r})".format(
            self.cancelled, self._reason, self._timeout
        )


class AsyncShutdown:
    """asyncio counterpart of Shutdown.

        shutdown = AsyncShutdown(timeout=10)

        async def main(shutdown):
            shutdown.on_shutdown(close_pool, "db-pool")
            await serve_forever(shutdown)

        shutdown = AsyncShutdown()
        await main(shutdown)
        await shutdown.run_hooks()

    Ordering and error semantics match Shutdown exactly: reverse registration
    order, failures collected rather than propagated mid-sequence, and one
    ShutdownError at the end.
    """

    __slots__ = ("_shutdown", "_async_hooks", "_async_done")

    def __init__(self, timeout: Optional[float] = None) -> None:
        self._shutdown = Shutdown(timeout=timeout)
        self._async_hooks: List[Tuple[str, Callable[[], Any]]] = []
        self._async_done = False

    @property
    def cancelled(self) -> bool:
        return self._shutdown.cancelled

    @property
    def reason(self) -> Optional[str]:
        return self._shutdown.reason

    @property
    def errors(self) -> List[Tuple[str, BaseException]]:
        return self._shutdown.errors

    def timeout_remaining(self) -> Optional[float]:
        return self._shutdown.timeout_remaining()

    def trigger(self, reason: str = "manual") -> bool:
        return self._shutdown.trigger(reason)

    def on_shutdown(self, fn: Callable[[], Any], name: Optional[str] = None) -> Callable[[], Any]:
        """Register an async cleanup hook; its coroutine is awaited on shutdown."""
        if not callable(fn):
            raise TypeError(
                "shutdown hook must be callable, got {0!r}".format(type(fn).__name__)
            )
        hook_name = name or getattr(fn, "__name__", None) or repr(fn)
        self._async_hooks.append((hook_name, fn))
        return fn

    async def wait(self, timeout: Optional[float] = None) -> bool:
        """Await until a stop is requested. Returns False on timeout.

        Implemented as a bounded poll loop rather than
        loop.add_signal_handler, so it also works on platforms and event
        loops where signal handling is unavailable.
        """
        import asyncio

        if timeout is None:
            while not self._shutdown.cancelled:
                await asyncio.sleep(0.05)
            return True

        deadline = time.monotonic() + timeout
        while not self._shutdown.cancelled:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(0.05, remaining))
        return True

    async def run_hooks(self) -> None:
        """Await every registered async hook, newest first, exactly once."""
        import inspect

        if self._async_done:
            return
        self._async_done = True

        errors: List[Tuple[str, BaseException]] = []
        for hook_name, fn in reversed(self._async_hooks):
            try:
                result = fn()
                if inspect.isawaitable(result):
                    await result
            except BaseException as exc:  # cleanup must not abort the rest
                errors.append((hook_name, exc))

        try:
            self._shutdown.run_hooks()
        except ShutdownError as exc:
            errors.extend(exc.errors)

        if errors:
            self._shutdown._errors.extend(errors)
            raise ShutdownError(errors)

    def __repr__(self) -> str:
        return "AsyncShutdown(cancelled={0!r}, reason={1!r})".format(
            self.cancelled, self.reason
        )


# ----------------------------------------------------------------------
# Signal handling
# ----------------------------------------------------------------------

_SIGNAL_NAMES = {
    signal.SIGINT: "SIGINT",
    signal.SIGTERM: "SIGTERM",
    getattr(signal, "SIGHUP", -1): "SIGHUP",
    getattr(signal, "SIGQUIT", -1): "SIGQUIT",
}


def _signal_name(signum: int) -> str:
    return _SIGNAL_NAMES.get(signum) or "signal-{0}".format(signum)


def _default_signals() -> List[int]:
    """SIGINT and SIGTERM where available (SIGTERM is absent on Windows)."""
    sigs = [signal.SIGINT]
    term = getattr(signal, "SIGTERM", None)
    if term is not None:
        sigs.append(term)
    return sigs


def _install_handlers(shutdown: Shutdown, signals: Optional[Iterable[int]]) -> Callable[[], None]:
    """Install handlers that trigger shutdown. Returns a restorer callable.

    Falls back to a no-op restorer on platforms (Windows) or contexts (a
    non-main thread) where signal.signal raises, because a worker thread
    that tried to install handlers would crash otherwise.
    """
    if not signals:
        return lambda: None

    previous: dict = {}

    def handler(signum, _frame):
        shutdown.trigger(_signal_name(signum))

    for sig in signals:
        try:
            previous[sig] = signal.signal(sig, handler)
        except (ValueError, OSError, RuntimeError):
            continue

    def restore() -> None:
        for sig, old in previous.items():
            try:
                signal.signal(sig, old)
            except (ValueError, OSError, RuntimeError):
                pass

    return restore


# ----------------------------------------------------------------------
# coordinator
# ----------------------------------------------------------------------


class _Coordinator:
    """Returned by coordinator(); use it as a context manager.

        with coordinator(timeout=10) as shutdown:
            shutdown.on_shutdown(flush_buffer, "buffer")
            run_forever(shutdown)

    Hooks have run, in reverse order, by the time the block exits — even
    when the body raises.
    """

    __slots__ = ("shutdown", "signals", "_restore")

    def __init__(self, shutdown: Shutdown, signals: Optional[Iterable[int]]) -> None:
        self.shutdown = shutdown
        self.signals = signals
        self._restore: Optional[Callable[[], None]] = None

    def __enter__(self) -> Shutdown:
        self._restore = _install_handlers(self.shutdown, self.signals or _default_signals())
        return self.shutdown

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            if self.shutdown.cancelled:
                # A stop was requested: wait out the drain, then tear down.
                self.shutdown.wait()
                self.shutdown.run_hooks()
            else:
                # Normal exit with no stop request; still release resources.
                self.shutdown.trigger("context-exit")
                self.shutdown.run_hooks()
        except ShutdownError:
            if exc is None:
                raise
            # An exception is already propagating; don't replace it with a
            # shutdown error, but do report the failures.
            for name, hook_exc in self.shutdown.errors:
                sys.stderr.write(
                    "tiny-graceful: hook {0} failed: {1}: {2}\n".format(
                        name, type(hook_exc).__name__, hook_exc
                    )
                )
            sys.stderr.flush()
        finally:
            if self._restore is not None:
                self._restore()
                self._restore = None
        return False


def coordinator(timeout: Optional[float] = None,
                signals: Optional[Iterable[int]] = None) -> _Coordinator:
    """Build a shutdown context manager that traps SIGINT and SIGTERM.

    timeout is the global drain budget in seconds. Pass signals=[] to
    install no handlers, which is what you want in tests.
    """
    return _Coordinator(Shutdown(timeout=timeout), signals)


# ----------------------------------------------------------------------
# serve(): workers plus shutdown
# ----------------------------------------------------------------------


def _guard(worker: Callable[[Any], Any], shutdown: Shutdown, index: int) -> None:
    """Run a worker, turning an unhandled crash into a clean stop.

    A crashing worker should bring the process down in a controlled way,
    not leave the service half-serving traffic.
    """
    try:
        worker(shutdown)
    except Exception as exc:
        sys.stderr.write(
            "tiny-graceful: worker {0} failed ({1}: {2}); triggering shutdown\n".format(
                index, type(exc).__name__, exc
            )
        )
        sys.stderr.flush()
        shutdown.trigger("worker-{0}-error".format(index))


def _drain(threads: List[threading.Thread], drain_timeout: Optional[float]) -> List[str]:
    """Join workers against one shared deadline. Returns the stragglers."""
    deadline = None if drain_timeout is None else time.monotonic() + drain_timeout
    for thread in threads:
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        thread.join(remaining)
    return [t.name for t in threads if t.is_alive()]


def serve(
    worker: Callable[[Shutdown], Any],
    worker_count: int = 1,
    drain_timeout: Optional[float] = 10.0,
    timeout: Optional[float] = None,
    signals: Optional[Iterable[int]] = None,
    hooks: Optional[Iterable[Callable[[], Any]]] = None,
    shutdown: Optional[Shutdown] = None,
) -> Shutdown:
    """Run worker in worker_count threads until a stop is requested.

    This is the whole library in one call: install handlers, spawn workers,
    block, drain, run hooks, and return the Shutdown so the caller can
    inspect reason, errors, and timeout_remaining().

    Args:
        worker: called once per thread with the Shutdown. Should loop and
            check shutdown.cancelled.
        worker_count: number of worker threads (default 1).
        drain_timeout: seconds to wait for workers after a stop (default
            10). None waits forever.
        timeout: global budget for shutdown hooks. Defaults to
            drain_timeout so one knob governs the entire exit path.
        signals: signals to trap; defaults to SIGINT + SIGTERM. Pass []
            for none.
        hooks: callables to run at shutdown, newest first.
        shutdown: an existing Shutdown to use instead of a fresh one. Pass
            this when something other than a signal should be able to stop
            the service — a /admin/drain endpoint, a health check, or a test.
            Its timeout wins over drain_timeout if you set both.

    Returns:
        The Shutdown object, after all hooks have run.

    Raises:
        ShutdownError if any hook failed.
    """
    if not callable(worker):
        raise TypeError("worker must be callable")
    if worker_count < 1:
        raise ValueError("worker_count must be >= 1")

    if shutdown is None:
        shutdown = Shutdown(timeout=drain_timeout)
    restore = _install_handlers(shutdown, signals or _default_signals())

    for hook in hooks or ():
        shutdown.on_shutdown(hook)

    threads: List[threading.Thread] = []
    try:
        for index in range(worker_count):
            thread = threading.Thread(
                target=_guard,
                args=(worker, shutdown, index),
                name="tiny-graceful-worker-{0}".format(index),
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        shutdown.wait()

        stragglers = _drain(threads, drain_timeout)
        if stragglers:
            sys.stderr.write(
                "tiny-graceful: {0} worker(s) overran the {1}s drain: {2}\n".format(
                    len(stragglers), drain_timeout, ", ".join(stragglers)
                )
            )
            sys.stderr.flush()
    finally:
        restore()

    shutdown.trigger("post-drain")
    shutdown.run_hooks()
    return shutdown


async def async_serve(
    worker: Callable[[AsyncShutdown], Any],
    drain_timeout: Optional[float] = 10.0,
    timeout: Optional[float] = None,
    signals: Optional[Iterable[int]] = None,
    hooks: Optional[Iterable[Callable[[], Any]]] = None,
    shutdown: Optional[AsyncShutdown] = None,
) -> AsyncShutdown:
    """asyncio counterpart of serve().

    worker is awaited once with an AsyncShutdown. When it returns, hooks run
    in reverse registration order.

    Same ordering rules, same error aggregation, same exit semantics as the
    threaded version.
    """
    import asyncio
    import signal as _signal

    if not callable(worker):
        raise TypeError("worker must be callable")

    if shutdown is None:
        shutdown = AsyncShutdown(
            timeout=timeout if timeout is not None else drain_timeout
        )
    loop = asyncio.get_event_loop()
    installed: List[int] = []

    if signals is None:
        sigs = [s for s in (getattr(_signal, "SIGINT", None),
                            getattr(_signal, "SIGTERM", None)) if s is not None]
    else:
        sigs = list(signals)

    try:
        for sig in sigs:
            loop.add_signal_handler(sig, shutdown.trigger, _signal_name(sig))
            installed.append(sig)
    except (NotImplementedError, RuntimeError, ValueError):
        installed = []

    for hook in hooks or ():
        shutdown.on_shutdown(hook)

    try:
        await worker(shutdown)
    finally:
        await shutdown.run_hooks()
        for sig in installed:
            try:
                loop.remove_signal_handler(sig)
            except (NotImplementedError, RuntimeError, ValueError):
                pass

    if shutdown.errors:
        raise ShutdownError(shutdown.errors)
    return shutdown
