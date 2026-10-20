"""Runnable demo for tiny-graceful.

Run it and press Ctrl-C (or send SIGTERM). You will see:

  1. workers start,
  2. the stop reason is captured,
  3. workers drain,
  4. hooks run newest-first.

    python example.py
    # in another shell:
    #   kill -TERM @@PID@@
"""

import threading
import time

from tiny_graceful import serve


def make_worker(worker_id):
    def worker(shutdown):
        n = 0
        while not shutdown.cancelled:
            n += 1
            print("worker {0}: handled {1}".format(worker_id, n))
            time.sleep(0.5)
        print("worker {0}: stop requested, finishing current item".format(worker_id))
    return worker


def open_pool():
    print("hook: open pool (registered first -> closes last)")


def acquire_conn():
    print("hook: acquire connection (registered later -> released first)")


def release_conn():
    print("hook: release connection")


def flush_metrics():
    print("hook: flush metrics")


def close_pool():
    print("hook: close pool")


def _run_both(shutdown):
    """Fan two workers out of one managed thread, so each prints its own id."""
    threads = [
        threading.Thread(target=make_worker(i), args=(shutdown,), daemon=True)
        for i in (1, 2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def main():
    print("starting; press Ctrl-C to trigger a graceful shutdown\n")
    result = serve(
        _run_both,
        worker_count=1,
        drain_timeout=5.0,
        hooks=[open_pool, acquire_conn, release_conn, flush_metrics, close_pool],
    )
    print("\nstopped. reason={0!r} errors={1}".format(result.reason, result.errors))


if __name__ == "__main__":
    main()
