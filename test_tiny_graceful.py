"""Tests for tiny-graceful.

Covers ordering, error aggregation, idempotency, deadlines, worker draining,
and the async path. Signal delivery itself is exercised by raising the signal
from a background thread in one test; the rest drive the API directly so the
suite stays fast and deterministic.
"""

import asyncio
import os
import signal
import sys

# Python 3.8/3.9 do not expose sys.stdlib_module_names (added in 3.10).
# Top-level stdlib modules the package is allowed to import.
_STDLIB_FALLBACK = {
    "abc", "argparse", "asyncio", "base64", "collections", "contextlib",
    "copy", "dataclasses", "datetime", "enum", "functools", "hashlib",
    "inspect", "io", "itertools", "json", "logging", "os", "pathlib",
    "queue", "random", "re", "signal", "socket", "string", "subprocess",
    "sys", "tempfile", "textwrap", "threading", "time", "traceback", "types",
    "typing", "uuid", "warnings", "weakref", "__future__",
}
import threading
import time

import pytest

from tiny_graceful import (
    AsyncShutdown,
    Shutdown,
    ShutdownError,
    async_serve,
    coordinator,
    serve,
)


# ----------------------------------------------------------------------
# Shutdown basics
# ----------------------------------------------------------------------


class TestShutdownBasics:
    def test_starts_uncancelled(self):
        s = Shutdown()
        assert s.cancelled is False
        assert s.reason is None
        assert s.errors == []

    def test_trigger_sets_cancelled_and_reason(self):
        s = Shutdown()
        assert s.trigger("SIGTERM") is True
        assert s.cancelled is True
        assert s.reason == "SIGTERM"

    def test_trigger_is_idempotent(self):
        s = Shutdown()
        assert s.trigger("first") is True
        assert s.trigger("second") is False
        assert s.reason == "first", "second trigger must not overwrite the reason"

    def test_wait_returns_true_after_trigger(self):
        s = Shutdown()
        threading.Timer(0.05, s.trigger).start()
        assert s.wait(timeout=2.0) is True

    def test_wait_returns_false_on_timeout(self):
        s = Shutdown()
        started = time.monotonic()
        assert s.wait(timeout=0.1) is False
        assert time.monotonic() - started < 1.0

    def test_wait_no_args_blocks_until_triggered(self):
        s = Shutdown()
        threading.Timer(0.05, s.trigger).start()
        assert s.wait() is True

    def test_timeout_remaining_none_without_deadline(self):
        assert Shutdown().timeout_remaining() is None

    def test_timeout_remaining_counts_down_after_trigger(self):
        s = Shutdown(timeout=5.0)
        s.trigger()
        remaining = s.timeout_remaining()
        assert remaining is not None
        assert 0 < remaining <= 5.0

    def test_timeout_remaining_full_budget_before_trigger(self):
        assert Shutdown(timeout=7.0).timeout_remaining() == 7.0

    def test_timeout_remaining_never_negative(self):
        s = Shutdown(timeout=0.05)
        s.trigger()
        time.sleep(0.12)
        assert s.timeout_remaining() == 0.0

    def test_repr_is_informative(self):
        s = Shutdown(timeout=3.0)
        s.trigger("manual")
        assert "Shutdown(" in repr(s)
        assert "manual" in repr(s)


# ----------------------------------------------------------------------
# Hooks
# ----------------------------------------------------------------------


class TestHooks:
    def test_hooks_run(self):
        s = Shutdown()
        calls = []
        s.on_shutdown(lambda: calls.append("a"))
        s.run_hooks()
        assert calls == ["a"]

    def test_hooks_run_in_reverse_registration_order(self):
        s = Shutdown()
        calls = []
        s.on_shutdown(lambda: calls.append(1))
        s.on_shutdown(lambda: calls.append(2))
        s.on_shutdown(lambda: calls.append(3))
        s.run_hooks()
        assert calls == [3, 2, 1], "last registered must close first"

    def test_on_shutdown_returns_fn_for_decorator_use(self):
        s = Shutdown()

        @s.on_shutdown
        def cleanup():
            calls.append("cleanup")

        calls = []
        s.run_hooks()
        assert calls == ["cleanup"]
        assert callable(cleanup)

    def test_hook_name_defaults_to_function_name(self):
        s = Shutdown()
        s.on_shutdown(lambda: None, name="explicit")
        s.on_shutdown(lambda: None)
        s.run_hooks()
        assert s.errors == []

    def test_run_hooks_is_idempotent(self):
        s = Shutdown()
        calls = []
        s.on_shutdown(lambda: calls.append(1))
        s.run_hooks()
        s.run_hooks()
        s.run_hooks()
        assert calls == [1], "hooks must run exactly once"

    def test_failing_hook_does_not_block_others(self):
        s = Shutdown()
        calls = []

        def boom():
            raise ValueError("nope")

        s.on_shutdown(lambda: calls.append("first"))
        s.on_shutdown(boom)
        s.on_shutdown(lambda: calls.append("third"))

        with pytest.raises(ShutdownError):
            s.run_hooks()

        assert calls == ["third", "first"], "both surviving hooks still ran"

    def test_shutdown_error_carries_original_exceptions(self):
        s = Shutdown()

        def boom():
            raise KeyError("missing")

        s.on_shutdown(boom, name="db-close")
        with pytest.raises(ShutdownError) as excinfo:
            s.run_hooks()

        assert len(excinfo.value.errors) == 1
        name, exc = excinfo.value.errors[0]
        assert name == "db-close"
        assert isinstance(exc, KeyError)
        assert "db-close" in str(excinfo.value)

    def test_all_hooks_fail_still_reports_every_one(self):
        s = Shutdown()

        def make(n):
            def hook():
                raise RuntimeError("fail-{0}".format(n))
            return hook

        for i in range(3):
            s.on_shutdown(make(i), name="hook-{0}".format(i))

        with pytest.raises(ShutdownError) as excinfo:
            s.run_hooks()
        assert len(excinfo.value.errors) == 3

    def test_non_callable_hook_rejected(self):
        s = Shutdown()
        with pytest.raises(TypeError):
            s.on_shutdown("not callable")

    def test_context_manager_runs_hooks_on_exit(self):
        calls = []
        with Shutdown() as s:
            s.on_shutdown(lambda: calls.append("done"))
        assert calls == ["done"]

    def test_context_manager_runs_hooks_even_on_exception(self):
        calls = []
        with pytest.raises(ValueError):
            with Shutdown() as s:
                s.on_shutdown(lambda: calls.append("cleanup"))
                raise ValueError("body blew up")
        assert calls == ["cleanup"], "cleanup must survive a raising body"


# ----------------------------------------------------------------------
# coordinator
# ----------------------------------------------------------------------


class TestCoordinator:
    def test_runs_hooks_on_exit(self):
        calls = []
        with coordinator(timeout=5) as s:
            s.on_shutdown(lambda: calls.append("bye"))
            s.trigger("test")
        assert calls == ["bye"]

    def test_hooks_run_without_explicit_trigger(self):
        calls = []
        with coordinator(timeout=5, signals=[]) as s:
            s.on_shutdown(lambda: calls.append("auto"))
        assert calls == ["auto"], "normal exit must still release resources"

    def test_signal_handler_triggers_shutdown(self):
        with coordinator(timeout=5) as s:
            def fire():
                time.sleep(0.05)
                os.kill(os.getpid(), signal.SIGTERM)

            threading.Thread(target=fire, daemon=True).start()
            assert s.wait(timeout=3.0) is True
            assert s.reason == "SIGTERM"

    def test_signals_empty_installs_nothing(self):
        with coordinator(timeout=5, signals=[]) as s:
            assert s.cancelled is False

    def test_restores_previous_handlers(self):
        original = signal.getsignal(signal.SIGTERM)
        with coordinator(timeout=5):
            assert signal.getsignal(signal.SIGTERM) is not original
        assert signal.getsignal(signal.SIGTERM) is original


# ----------------------------------------------------------------------
# serve()
# ----------------------------------------------------------------------


class TestServe:
    def test_runs_worker_and_stops_on_trigger(self):
        ran = threading.Event()

        def worker(shutdown):
            ran.set()
            while not shutdown.cancelled:
                time.sleep(0.01)

        result = _serve_with_stop(worker, worker_count=2)
        assert ran.is_set()
        assert result.cancelled is True

    def test_uses_caller_supplied_shutdown(self):
        sh = Shutdown()

        def worker(shutdown):
            assert shutdown is sh
            while not shutdown.cancelled:
                time.sleep(0.01)

        result = _serve_with_stop(worker, worker_count=1, shutdown=sh)
        assert result is sh

    def test_runs_requested_number_of_workers(self):
        lock = threading.Lock()
        started = []

        def worker(shutdown):
            with lock:
                started.append(threading.current_thread().name)
            while not shutdown.cancelled:
                time.sleep(0.01)

        _serve_with_stop(worker, worker_count=4, stop_after=0.2)
        assert len(started) == 4
        assert len(set(started)) == 4, "each worker gets its own thread"

    def test_hooks_run_in_reverse_order(self):
        order = []

        def worker(shutdown):
            while not shutdown.cancelled:
                time.sleep(0.01)

        _serve_with_stop(
            worker,
            worker_count=1,
            hooks=[lambda: order.append("a"), lambda: order.append("b")],
            stop_after=0.15,
        )
        assert order == ["b", "a"]

    def test_hooks_run_after_drain(self):
        order = []

        def worker(shutdown):
            while not shutdown.cancelled:
                time.sleep(0.01)
            order.append("worker-stopped")

        def hook():
            order.append("hook-ran")

        _serve_with_stop(worker, worker_count=1, hooks=[hook], stop_after=0.15)
        assert order == ["worker-stopped", "hook-ran"], (
            "drain must finish before hooks run"
        )

    def test_drain_timeout_is_reported_not_fatal(self):
        def slow_worker(shutdown):
            while not shutdown.cancelled:
                time.sleep(0.01)
            time.sleep(5)  # ignores the stop

        started = time.monotonic()
        result = _serve_with_stop(
            slow_worker, worker_count=1, drain_timeout=0.3, stop_after=0.1
        )
        elapsed = time.monotonic() - started
        assert elapsed < 3.0, "must not wait for the full 5s sleep"
        assert result.cancelled is True

    def test_worker_crash_triggers_clean_shutdown(self):
        def crashing_worker(shutdown):
            raise RuntimeError("worker exploded")

        sh = Shutdown()
        result = _serve_with_stop(
            crashing_worker, worker_count=1, stop_after=3.0, shutdown=sh
        )
        assert result.cancelled is True
        assert "worker-0-error" in (result.reason or "")

    def test_bad_worker_count_rejected(self):
        with pytest.raises(ValueError):
            serve(lambda s: None, worker_count=0, signals=[])

    def test_non_callable_worker_rejected(self):
        with pytest.raises(TypeError):
            serve("not callable", signals=[])

    def test_non_callable_hook_rejected(self):
        sh = Shutdown()
        sh.trigger()
        with pytest.raises(TypeError):
            serve(lambda s: None, signals=[], hooks=["not callable"], shutdown=sh)

    def test_returns_shutdown_instance(self):
        def worker(shutdown):
            return None

        result = _serve_with_stop(worker, worker_count=1)
        assert isinstance(result, Shutdown)


def _serve_with_stop(worker, worker_count=1, drain_timeout=2.0, hooks=None,
                     stop_after=0.1, shutdown=None):
    """Run serve() on a helper thread and trigger the stop after stop_after.

    The stop is delivered through a caller-supplied Shutdown rather than a
    real signal: serve() must be called from the main thread to install
    signal handlers, and raising SIGTERM here would take down pytest.
    """
    sh = shutdown if shutdown is not None else Shutdown()
    box = {}

    def target():
        box["result"] = serve(
            worker,
            worker_count=worker_count,
            drain_timeout=drain_timeout,
            hooks=hooks,
            shutdown=sh,
        )

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    time.sleep(stop_after)
    sh.trigger("test-trigger")
    thread.join(timeout=10)
    assert not thread.is_alive(), "serve() did not return"
    assert "result" in box, "serve() did not return a Shutdown"
    return box["result"]


# ----------------------------------------------------------------------
# AsyncShutdown
# ----------------------------------------------------------------------


def run(coro):
    return asyncio.run(coro)


class TestAsyncShutdown:
    def test_starts_uncancelled(self):
        s = AsyncShutdown()
        assert s.cancelled is False

    def test_trigger_and_reason(self):
        s = AsyncShutdown()
        assert s.trigger("SIGINT") is True
        assert s.cancelled is True
        assert s.reason == "SIGINT"

    def test_trigger_idempotent(self):
        s = AsyncShutdown()
        s.trigger("first")
        assert s.trigger("second") is False
        assert s.reason == "first"

    def test_async_hooks_run_in_reverse_order(self):
        order = []

        async def main():
            s = AsyncShutdown()
            s.on_shutdown(lambda: order.append(1))
            s.on_shutdown(lambda: order.append(2))
            s.on_shutdown(lambda: order.append(3))
            await s.run_hooks()

        run(main())
        assert order == [3, 2, 1]

    def test_async_hook_coroutines_are_awaited(self):
        done = []

        async def cleanup():
            await asyncio.sleep(0)
            done.append("cleaned")

        async def main():
            s = AsyncShutdown()
            s.on_shutdown(cleanup, "async-cleanup")
            await s.run_hooks()

        run(main())
        assert done == ["cleaned"], "coroutine hooks must actually be awaited"

    def test_sync_hooks_work_on_async_shutdown(self):
        done = []

        async def main():
            s = AsyncShutdown()
            s.on_shutdown(lambda: done.append("sync"))
            await s.run_hooks()

        run(main())
        assert done == ["sync"]

    def test_failing_async_hook_does_not_block_others(self):
        done = []

        async def boom():
            raise ValueError("bad")

        async def main():
            s = AsyncShutdown()
            s.on_shutdown(lambda: done.append("first"))
            s.on_shutdown(boom, "boom")
            s.on_shutdown(lambda: done.append("third"))
            with pytest.raises(ShutdownError):
                await s.run_hooks()

        run(main())
        assert done == ["third", "first"]

    def test_run_hooks_idempotent(self):
        calls = []

        async def main():
            s = AsyncShutdown()
            s.on_shutdown(lambda: calls.append(1))
            await s.run_hooks()
            await s.run_hooks()

        run(main())
        assert calls == [1]

    def test_wait_returns_true_after_trigger(self):
        async def main():
            s = AsyncShutdown()
            asyncio.get_event_loop().call_later(0.05, s.trigger)
            return await s.wait(timeout=2.0)

        assert run(main()) is True

    def test_wait_returns_false_on_timeout(self):
        async def main():
            return await AsyncShutdown().wait(timeout=0.1)

        assert run(main()) is False


class TestAsyncServe:
    def test_worker_runs_and_hooks_fire(self):
        seen = []

        async def worker(shutdown):
            seen.append(shutdown.cancelled)

        async def main():
            await async_serve(
                worker,
                hooks=[lambda: seen.append("hook")],
                signals=[],
            )

        run(main())
        assert seen == [False, "hook"]

    def test_hooks_run_in_reverse_order(self):
        order = []

        async def worker(shutdown):
            return None

        async def main():
            await async_serve(
                worker,
                hooks=[lambda: order.append("a"), lambda: order.append("b")],
                signals=[],
            )

        run(main())
        assert order == ["b", "a"]

    def test_failing_hook_raises_shutdown_error(self):
        async def worker(shutdown):
            return None

        def boom():
            raise RuntimeError("teardown failed")

        async def main():
            with pytest.raises(ShutdownError):
                await async_serve(worker, hooks=[boom], signals=[])

        run(main())

    def test_non_callable_worker_rejected(self):
        async def main():
            with pytest.raises(TypeError):
                await async_serve("not callable", signals=[])

        run(main())

    def test_worker_can_trigger_stop(self):
        seen = []

        async def worker(shutdown):
            shutdown.trigger("done")
            seen.append(shutdown.reason)

        async def main():
            return await async_serve(worker, signals=[])

        result = run(main())
        assert seen == ["done"]
        assert result.reason == "done"


# ----------------------------------------------------------------------
# Package hygiene
# ----------------------------------------------------------------------


class TestModule:
    def test_version_is_exported(self):
        import tiny_graceful

        assert tiny_graceful.__version__ == "0.1.0"
        assert "Shutdown" in tiny_graceful.__all__

    def test_has_no_third_party_imports(self):
        """The whole pitch is stdlib-only; assert it stays true."""
        import ast
        import pathlib

        # sys.stdlib_module_names only exists on Python 3.10+; the package
        # itself supports 3.8+, so fall back to a known-stdlib list there.
        if hasattr(sys, "stdlib_module_names"):
            stdlib = set(sys.stdlib_module_names)
        else:
            stdlib = _STDLIB_FALLBACK
        source = pathlib.Path(__file__).with_name("tiny_graceful.py")
        tree = ast.parse(source.read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
        # __future__ is a compiler directive, not a real dependency.
        imported.discard("__future__")
        outside = sorted(imported - stdlib)
        assert outside == [], "non-stdlib imports found: {0}".format(outside)
