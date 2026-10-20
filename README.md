# tiny-graceful

> **Zero-dependency graceful shutdown for Python services.**
> Signal handling, reverse-order cleanup hooks, and bounded drains — one file, stdlib only.

```bash
pip install tiny-graceful
# or: copy tiny_graceful.py into your project (MIT licensed)
```

## The problem

Every long-running program eventually gets told to stop. Ctrl-C sends `SIGINT`.
Kubernetes sends `SIGTERM` and gives you ten seconds. Docker sends `SIGTERM`, then
`SIGKILL` ninety seconds later. The handling you need is always the same:

> stop accepting work, drain what is in flight, close resources in the right order, exit

Most teams write that from scratch, badly, per service. The bugs are always the same
four. **tiny-graceful fixes all four and is one file.**

```python
from tiny_graceful import serve

def worker(shutdown):
    while not shutdown.cancelled:
        handle_request()

serve(
    worker,
    hooks=[flush_metrics, close_pool, close_db],
    drain_timeout=10.0,
)
```

That is the whole integration. Handlers installed, workers spawned, Ctrl-C or `SIGTERM`
drained with a ten-second budget, hooks run newest-first, process exits.

## The four bugs this fixes

**1. Cleanup in the wrong order.** Open a connection, then a transaction on it. Close the
connection first and the transaction explodes. Hooks run in **reverse registration order**
here, always — register outermost first, exactly as you opened them.

```python
shutdown.on_shutdown(close_db)      # registered first
shutdown.on_shutdown(rollback_tx)   # runs first, closes correctly
```

**2. One broken cleanup strands the rest.** If `close_db()` raises, your connection pool
never closes and you leak a socket. Here a failing hook is **recorded and stepped over**;
every other hook still runs, and one `ShutdownError` is raised at the end carrying the
original exceptions.

```python
try:
    shutdown.run_hooks()
except ShutdownError as e:
    for name, exc in e.errors:
        logger.error("shutdown hook %s failed: %r", name, exc)
```

**3. Deadlines that are per-hook instead of global.** Ten hooks × a 10-second timeout is
100 seconds — the supervisor `SIGKILL`s you at 30. Here `drain_timeout` is **one budget for the
whole exit path**: workers and hooks share it.

**4. Double teardown from a second signal.** You Ctrl-C, it hangs, you hit Ctrl-C again —
now `SIGINT` *and* `SIGTERM` and your teardown runs twice. `Shutdown.trigger()` is
one-shot: only the first caller wins, later ones get `False` and change nothing.

## Install

```bash
pip install tiny-graceful
```

Or vendor the single file:

```bash
curl -O https://raw.githubusercontent.com/hussain-alsaibai/tiny-graceful/main/tiny_graceful.py
```

## Quick start

### serve() — workers plus shutdown

```python
from tiny_graceful import serve, ShutdownError

def worker(shutdown):
    while not shutdown.cancelled:
        conn = next_connection()
        try:
            serve_request(conn)
        finally:
            conn.close()          # per-request cleanup is still your job

def flush_metrics():
    prometheus_client.REGISTRY.collect()

try:
    result = serve(worker, worker_count=8, drain_timeout=10.0,
                   hooks=[flush_metrics, close_pool])
    print("stopped because:", result.reason)   # 'SIGTERM', 'SIGINT', ...
except ShutdownError as e:
    for name, exc in e.errors:
        print("hook", name, "failed:", exc)
```

### Shutdown — the building block

Use it directly when you own the threads and only want the stop signal plus hook ordering.

```python
from tiny_graceful import Shutdown

shutdown = Shutdown(timeout=10.0)

@shutdown.on_shutdown
def flush():
    ...

def worker():
    while not shutdown.cancelled:
        do_work()
        remaining = shutdown.timeout_remaining()
        if remaining is not None and remaining < 1.0:
            break          # wrap up early so the hook still has time to run

worker_threads = [threading.Thread(target=worker) for _ in range(4)]
shutdown.wait()                       # blocks until a signal arrives
for t in worker_threads:
    t.join(timeout=10)
shutdown.run_hooks()
```

### coordinator() — handlers as a context manager

```python
from tiny_graceful import coordinator

with coordinator(timeout=10) as shutdown:
    shutdown.on_shutdown(flush_buffer, "buffer")
    run_forever(shutdown)
# hooks have definitely run by the time the block exits, even if the body raised
```

### async_serve() — asyncio

Same ordering rules, same error aggregation.

```python
import asyncio
from tiny_graceful import async_serve

async def main(shutdown):
    server = await start_server()
    shutdown.on_shutdown(server.close, "http-server")
    async with server:
        await asyncio.Event().wait()

shutdown = asyncio.run(async_serve(main, drain_timeout=10.0))
```

### Stopping without a signal

``serve()` accepts a `Shutdown` you already own, which is how you wire a drain endpoint or a
test into the same exit path:

```python
from tiny_graceful import serve, Shutdown

shutdown = Shutdown(timeout=10.0)

# ... somewhere in your HTTP handler:
#   shutdown.trigger("admin-drain")

serve(worker, shutdown=shutdown, hooks=[close_pool])
```

## API

| Object | Purpose |
|--------|---------|
| `serve(worker, ...)` | Threads + shutdown in one call. Returns the `Shutdown`. |
| `coordinator(timeout, signals)` | Context manager that traps SIGINT/SIGTERM. |
| `async_serve(worker, ...)` | asyncio version of `serve()`. |
| `Shutdown(timeout)` | Stop signal, deadline, hook registry. |
| `AsyncShutdown(timeout)` | asyncio variant. |
| `ShutdownError` | Carries `(name, exception)` for every failed hook. |

`serve()` and `async_serve()` accept:

| Argument | Default | Meaning |
|----------|---------|---------|
| `worker` | — | Called per thread (or awaited once) with the shutdown object. |
| `worker_count` | `1` | Worker threads. |
| `drain_timeout` | `10.0` | Seconds to wait for workers. `None` waits forever. |
| `timeout` | `drain_timeout` | Global budget for shutdown hooks. |
| `signals` | SIGINT + SIGTERM | `[]` installs nothing (use in tests). |
| `hooks` | `()` | Cleanup callables, run newest-first. |
| `shutdown` | fresh `Shutdown` | Supply your own to trigger stops from code. |

`Shutdown` exposes `cancelled`, `reason`, `errors`, `wait(timeout)`,
`trigger(reason)`, `timeout_remaining()`, `on_shutdown(fn, name)`, and
`run_hooks()`.

## Notes on behaviour

- **Signal handlers only install on the main thread.** `signal.signal` raises elsewhere
  and on Windows for signals the platform lacks. tiny-graceful skips those signals
  instead of crashing — your worker threads never install handlers.
- **Overrunning workers are reported, not killed.** If a worker ignores the stop past
  `drain_timeout`, `serve()` writes the straggler thread names to stderr and returns.
  Workers are daemon threads, so the process still exits; you get the diagnostic.
- **A crashing worker triggers a clean shutdown** rather than leaving the service half
  up. The stop reason becomes `worker-N-error`.
- **Async signal handling degrades gracefully.** `async_serve()` prefers
  `loop.add_signal_handler` and falls back to a bounded poll loop where that is
  unavailable.

## Requirements

Python 3.8+, standard library only. No dependencies, ever. There is a test that
parses this module's AST and fails if a non-stdlib import ever appears.

## Testing

```bash
python -m pytest test_tiny_graceful.py -q
```

55 tests covering hook ordering, error aggregation, idempotency, deadlines, worker
draining, crash propagation, and the async path.

## License

MIT © OpenClaw
