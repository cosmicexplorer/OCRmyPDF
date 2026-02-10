# SPDX-FileCopyrightText: 2022 James R. Barlow
# SPDX-License-Identifier: MPL-2.0
"""OCRmyPDF's multiprocessing/multithreading abstraction layer."""

from __future__ import annotations

import functools
import logging
import logging.handlers
import multiprocessing
import multiprocessing.queues
import os
import queue
import signal
import sys
import threading
import time
from abc import abstractmethod, abstractproperty
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import (
    Executor as StdlibExecutor,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from contextlib import suppress
from enum import Enum, auto
from typing import TYPE_CHECKING

from rich.console import Console as RichConsole

from ocrmypdf import Executor, ExecutorBase, WorkloadKind, SharedLock, hookimpl
from ocrmypdf._logging import RichLoggingHandler
from ocrmypdf._progressbar import ProgressBar, RichProgressBar
from ocrmypdf.exceptions import InputFileError, CancelRunningTasksMixin
from ocrmypdf.helpers import remove_all_log_handlers

class _WaitBehavior(Enum):
    BLOCK_INDEFINITELY = auto()
    TIMEOUT_MAX = auto()


class _CancelBehavior(Enum):
    CANCEL_RUNNING_TASKS = auto()
    NO_CANCELLATION = auto()


if TYPE_CHECKING:
    from types import TracebackType
    from typing import Any, ClassVar, Generic, Protocol, TypeAlias, TypeVar, runtime_checkable
    from typing_extensions import Self

    @runtime_checkable
    class Queue[T](Protocol):
        def empty(self) -> bool:
            """Whether the queue has nothing in it."""

        def get(self) -> T | None:
            """Wait to retrieve the next value."""

        def put_nowait(self, val: T | None) -> None:
            """Put a new value into the queue, without any synchronization."""

    assert issubclass(multiprocessing.queues.Queue, Queue)
    assert issubclass(queue.Queue, Queue)

    UserInit: TypeAlias = Callable[[], None]
    WorkerInit: TypeAlias = Callable[[Queue[logging.LogRecord], UserInit, int], None]

    E = TypeVar('E', bound=BaseException)

    @runtime_checkable
    class StandardExecutorGeneration[
        Q: Queue[logging.LogRecord],
        Exe: StdlibExecutor,
    ](Protocol):
        def __call__(
            self,
            max_workers: int,
            initializer: WorkerInit,
            initargs: tuple[Q, UserInit, int],
        ) -> Exe:
            pass

        def perform_shutdown_strategy(
            self,
            inner: Exe,
            listener: threading.Thread,
            q: Q,
            wait: _WaitBehavior,
            cancel: _CancelBehavior,
            timeout_max: int | float | None,
        ) -> None:
            pass

    @runtime_checkable
    class LockGenerator[Lock: SharedLock](Protocol):
        def __call__(self) -> Lock:
            pass

    @runtime_checkable
    class QueueGenerator[Q: Queue[logging.LogRecord]](Protocol):
        def __call__(self, n: int) -> Q:
            pass

    assert isinstance(type[ThreadPoolExecutor], StandardExecutorGeneration)
    assert isinstance(type[ProcessPoolExecutor], StandardExecutorGeneration)

    @runtime_checkable
    class ExecutorPlatform[
        Lock: SharedLock,
        Q: Queue[logging.LogRecord],
        Exe: StdlibExecutor,
    ](Executor[Lock], Protocol):
        def make_queue(self) -> Q:
            pass

        def make_lock(self) -> Lock:
            pass

        def make_exe(
            self,
            *,
            max_workers: int,
            worker_initializer: Callable,
        ) -> Exe:
            pass


_basic_logger = logging.getLogger(__name__)


class _ThreadedExecutorWrapper:
    __slots__ = ('_ty',)

    def __init__(self):
        self._ty = ThreadPoolExecutor

    def __call__(
        self,
        max_workers: int,
        initializer: WorkerInit,
        initargs: tuple['queue.Queue', UserInit, int],
    ) -> ThreadPoolExecutor:
        return self._ty(
            max_workers=max_workers,
            initializer=initializer,
            initargs=initargs,
        )

    _per_thread_wait_interval = 0.05 # 50ms

    def perform_shutdown_strategy(
        self,
        inner: ThreadPoolExecutor,
        listener: threading.Thread,
        q: queue.Queue,
        wait: _WaitBehavior,
        cancel: _CancelBehavior,
        timeout_max: int | float | None,
    ) -> None:
        match cancel:
            case _CancelBehavior.CANCEL_RUNNING_TASKS:
                cancel_futures = True
            case _CancelBehavior.NO_CANCELLATION:
                cancel_futures = False
        q.shutdown(immediate=cancel_futures)

        match wait:
            case _WaitBehavior.BLOCK_INDEFINITELY:
                assert timeout_max is None, timeout_max
                _basic_logger.warn(f"blocking indefinitely upon this thread pool: %s (queue %r)",
                                   inner, q)
                # Nothing changed from the stdlib implementation.
                inner.shutdown(wait=True, cancel_futures=cancel_futures)
                # Block indefinitely on the listener thread with the other end of the queue!
                listener.join()
                # Block indefinitely on the queue to free up!
                q.join()

            case _WaitBehavior.TIMEOUT_MAX:
                assert timeout_max is not None
                assert timeout_max > 0, timeout_max
                if timeout_max < self._per_thread_wait_interval:
                    _basic_logger.info(
                        'the total requested wait time (given %s) '
                        'will be increased to match the wait interval %s',
                        timeout_max, self._per_thread_wait_interval,
                    )
                    timeout_max = self._per_thread_wait_interval

                beg = time.monotonic()

                # NB: We add our own "listener" thread to the start of this collection, so it is
                #     always nonempty.
                remaining_threads = [listener]
                # Manipulate private stdlib thread pool state:
                if inner._shutdown_lock.acquire(blocking=True, timeout=timeout_max):
                    # Set the private internal shutdown flag.
                    inner._shutdown = True

                    # Manipulate the shared queue in very stateful and unexpected ways!
                    work_queue = inner._work_queue
                    if cancel_futures:
                        # Drain all work items from the queue, and then cancel their
                        # associated futures.
                        while True:
                            try:
                                work_item = work_queue.get_nowait()
                            except queue.Empty:
                                break
                            if work_item is not None:
                                work_item.future.cancel()
                    # Send a subsequent wake-up notif to any *other* threads that may be blocking on
                    # this queue:
                    work_queue.put(None) # type: ignore[arg-type]
                    # We hold the lock, so synchronous .put() is allowed!

                    # Copy over the list of active threads.
                    remaining_threads.extend(inner._threads)
                else:
                    raise RuntimeError(
                        f"timed out blocking on shutdown lock for thread pool {inner!r}")

                # We should only get here if we successfully acquired the lock within the timeout!
                assert inner._shutdown_lock.locked(), inner._shutdown_lock
                inner._shutdown_lock.release()
                # We always have our own listener thread at the very least.
                assert len(remaining_threads) > 0, remaining_threads

                # Now repeatedly wait for the specified interval, until the duration is past.
                while (elapsed := time.monotonic() - beg) < timeout_max:
                    try:
                        cur_thread = remaining_threads.pop()
                    except IndexError:
                        # No more waiting for threads!
                        break
                    # Wait for a bit!
                    cur_thread.join(timeout=self._per_thread_wait_interval)
                    # If it's still out there, then push it back!
                    if cur_thread.is_alive():
                        remaining_threads.append(cur_thread)
                else:
                    assert elapsed >= timeout_max, (elapsed, timeout_max)
                    raise TimeoutError(
                        f"waited {elapsed!r} seconds "
                        f"for intervals of {self._per_thread_wait_interval!r}; "
                        f"this outlasted the specified maximum timeout {timeout_max!r} !"
                    )

                # We should be here after breaking out of the while loop!
                assert len(remaining_threads) == 0, remaining_threads
                # NB: Since the listener should have the other end of this queue, it should
                #     be empty here!
                assert q.empty(), q


class _IPCExecutorWrapper:
    __slots__ = ('_ty',)

    def __init__(self):
        self._ty = ProcessPoolExecutor

    def __call__(
        self,
        max_workers: int,
        initializer: WorkerInit,
        initargs: tuple['multiprocessing.queues.Queue', UserInit, int],
    ) -> ProcessPoolExecutor:
        return self._ty(
            max_workers=max_workers,
            initializer=initializer,
            initargs=initargs,
        )

    _per_process_wait_interval = 0.1 # 100ms

    def perform_shutdown_strategy(
        self,
        inner: ProcessPoolExecutor,
        listener: threading.Thread,
        q: multiprocessing.queues.Queue,
        wait: _WaitBehavior,
        cancel: _CancelBehavior,
        timeout_max: int | float | None,
    ) -> None:
        match cancel:
            case _CancelBehavior.CANCEL_RUNNING_TASKS:
                cancel_futures = True
            case _CancelBehavior.NO_CANCELLATION:
                cancel_futures = False

        match wait:
            case _WaitBehavior.BLOCK_INDEFINITELY:
                assert timeout_max is None, timeout_max
                _basic_logger.warn(f"blocking indefinitely upon this process pool: %s (queue %r)",
                                   inner, q)
                # Nothing changed from the stdlib implementation.
                inner.shutdown(wait=True, cancel_futures=cancel_futures)
                # Block indefinitely on the listener thread with the other end of the queue!
                listener.join()
                # Block indefinitely on the queue to free up!
                q.join_thread()

            case _WaitBehavior.TIMEOUT_MAX:
                assert timeout_max is not None
                assert timeout_max > 0, timeout_max
                if timeout_max < self._per_process_wait_interval:
                    _basic_logger.info(
                        'the total requested wait time (given %s) '
                        'will be increased to match the wait interval %s',
                        timeout_max, self._per_process_wait_interval,
                    )
                    timeout_max = self._per_process_wait_interval

                beg = time.monotonic()

                # NB: We add our own "listener" thread to the start of this collection, so it is
                #     always nonempty.
                remaining_threads: list[threading.Thread] = [listener]
                remaining_processes = {}
                rq = None

                # Manipulate private stdlib multiprocessing pool state:
                if inner._shutdown_lock.acquire(blocking=True, timeout=timeout_max):
                    # Set the private internal shutdown flags.
                    inner._cancel_pending_futures = cancel_futures
                    inner._shutdown_thread = True

                    # Obtain handle to manager thread.
                    if (manager_thread := inner._executor_manager_thread) is not None:
                        # Add queue manager thread to our list of handles to join.
                        remaining_threads.append(manager_thread) # type: ignore[arg-type]
                        inner._executor_manager_thread = None    # type: ignore[assignment]
                    if (wakeup := inner._executor_manager_thread_wakeup) is not None:
                        # Wake up queue manager thread.
                        wakeup.wakeup()

                    # Recover the internal queues, and nullify them in the source.
                    if (cq := inner._call_queue) is not None: # type: ignore[attr-defined]
                        # This is not exactly a future -- no need to wait. Drop it now!
                        # Semantics of this at https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue.cancel_join_thread
                        cq.cancel_join_thread()
                        cq.close()
                        cq.join_thread()
                    inner._call_queue = None # type: ignore[attr-defined]

                    if (rq := inner._result_queue) is not None:
                        # This one however *does* contain futures.
                        rq.close()
                    inner._result_queue = None # type: ignore[assignment]

                    # Now recover the process handles.
                    if procs := inner._processes:
                        remaining_processes.update(procs.copy()) # type: ignore[attr-defined]
                    inner._processes = None                      # type: ignore[assignment]

                    inner._executor_manager_thread_wakeup = None # type: ignore[assignment]

                else:
                    raise RuntimeError(
                        f"timed out blocking on shutdown lock for process pool {inner!r}")

                # We should only get here if we successfully acquired the lock within the timeout!
                assert inner._shutdown_lock.locked(), inner._shutdown_lock
                inner._shutdown_lock.release()
                # We always have our own listener thread at the very least.
                assert len(remaining_threads) > 0, remaining_threads

                if cancel_futures:
                    for key, proc in remaining_processes.items():
                        # Determine if process is already exited/closed out.
                        try:
                            if not proc.is_alive():
                                del remaining_processes[key]
                                continue
                        except ValueError:
                            del remaining_processes[key]
                            continue

                        # Send initial SIGTERM to enable graceful exit.
                        try:
                            proc.terminate()
                        except ProcessLookupError:
                            # The process just ended before our signal!
                            del remaining_processes[key]
                            continue

                # Now repeatedly wait for the specified interval, until the duration is past.
                while (elapsed := time.monotonic() - beg) < timeout_max:
                    # Determine any processes which have exited since last time.
                    for key, proc in remaining_processes.items():
                        try:
                            if not proc.is_alive():
                                del remaining_processes[key]
                                continue
                        except ValueError:
                            del remaining_processes[key]
                            continue

                    # Now onto the threads!
                    try:
                        cur_thread = remaining_threads.pop()
                    except IndexError:
                        # No more waiting for threads!
                        break
                    # Wait for a bit!
                    cur_thread.join(timeout=self._per_process_wait_interval)
                    # If it's still out there, then push it back!
                    if cur_thread.is_alive():
                        remaining_threads.append(cur_thread)
                else:
                    assert elapsed >= timeout_max, (elapsed, timeout_max)
                    if cancel_futures:
                        for key, proc in remaining_processes.items():
                            try:
                                if not proc.is_alive():
                                    del remaining_processes[key]
                                    continue
                            except ValueError:
                                del remaining_processes[key]
                                continue

                            # Send SIGKILL if cancellation is requested and timeout is past.
                            try:
                                proc.kill()
                            except ProcessLookupError:
                                del remaining_processes[key]
                                continue

                    raise TimeoutError(
                        f"waited {elapsed!r} seconds "
                        f"for intervals of {self._per_process_wait_interval!r}; "
                        f"this outlasted the specified maximum timeout {timeout_max!r} !"
                    )

                if cancel_futures:
                    for key, proc in remaining_processes.items():
                        try:
                            if not proc.is_alive():
                                del remaining_processes[key]
                                continue
                        except ValueError:
                            del remaining_processes[key]
                            continue

                        # Send SIGKILL if cancellation is requested and timeout is past.
                        try:
                            proc.kill()
                        except ProcessLookupError:
                            del remaining_processes[key]
                            continue

                if rq is not None:
                    if (elapsed := time.monotonic() - beg) < timeout_max:
                        if cancel_futures:
                            rq.cancel_join_thread() # type: ignore[attr-defined]
                    else:
                        # Have *finally* overrun the timeout!
                        assert elapsed >= timeout_max, (elapsed, timeout_max)
                        rq.cancel_join_thread() # type: ignore[attr-defined]
                        raise TimeoutError(
                            f"waited {elapsed!r} seconds "
                            f"for intervals of {self._per_process_wait_interval!r}; "
                            f"this outlasted the specified maximum timeout {timeout_max!r} !"
                        )
                    rq.join_thread() # type: ignore[attr-defined]

                # NB: Since the listener should have the other end of this queue, it should
                #     be empty here!
                assert q.empty(), q


def log_listener(q: Queue[logging.LogRecord]) -> None:
    """Listen to the worker processes and forward the messages to logging.

    For simplicity this is a thread rather than a process. Only one process
    should actually write to sys.stderr or whatever we're using, so if this is
    made into a process the main application needs to be directed to it.

    See:
    https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes
    """
    while True:
        try:
            record = q.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)
        except Exception:  # pylint: disable=broad-except
            import traceback  # pylint: disable=import-outside-toplevel

            print("Logging problem", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)



def process_sigbus(*args):
    """Handle SIGBUS signal at the worker level."""
    raise InputFileError("A worker process lost access to an input file")


def process_sigint(*args):
    """Handle SIGBUS signal at the worker level."""
    raise KeyboardInterrupt('simulated SIGINT')


def process_init(
    q: Queue[logging.LogRecord], user_init: UserInit, loglevel: int
) -> None:
    """Initialize a process pool worker."""
    # Ignore SIGINT (our parent process will kill us gracefully)
    signal.signal(signal.SIGINT, process_sigint)

    # Install SIGBUS handler (so our parent process can abort somewhat gracefully)
    with suppress(AttributeError):  # Windows and Cygwin do not have SIGBUS
        # Windows and Cygwin do not have pthread_sigmask or SIGBUS
        signal.signal(signal.SIGBUS, process_sigbus)

    # Remove any log handlers inherited from the parent process
    root = logging.getLogger()
    remove_all_log_handlers(root)

    # Set up our single log handler to forward messages to the parent
    root.setLevel(loglevel)
    root.addHandler(logging.handlers.QueueHandler(q))

    user_init()
    return


def thread_init(
    q: Queue[logging.LogRecord], user_init: UserInit, loglevel: int
) -> None:
    """Begin a thread pool worker."""
    del q  # unused but required argument
    del loglevel  # unused but required argument
    # As a thread, block SIGBUS so the main thread deals with it...
    with suppress(AttributeError):
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGBUS, signal.SIGINT})

    user_init()
    return


@functools.cache
def setup_executor(workload: WorkloadKind) -> StandardExecutorGeneration:
    # Check if semaphore support is available, and if not, fall back to using threads.
    match workload:
        case WorkloadKind.MORE_MESSAGING:
            if _IPCExecutor._try_lock_type() is not None:
                return _IPCExecutor._executor_class()
            # We currently expect inter-thread locks to always be available.
            _basic_logger.info(
                "ipc semaphore unavailable for workload %s (falling back to %s)",
                workload,
                WorkloadKind.MORE_SHARED_DATA,
            )
            return setup_executor(WorkloadKind.MORE_SHARED_DATA)
        case WorkloadKind.MORE_SHARED_DATA:
            if _ThreadedExecutor._try_lock_type() is not None:
                return _ThreadedExecutor._executor_class()
            _basic_logger.info(
                "in-memory semaphore unavailable for workload %s (no fallback)",
                workload,
            )
    raise RuntimeError(
        f"could not produce executor for given workload type {workload!r}"
    )


class _ConcurrentExecutorBase(ExecutorBase):
    """Standard OCRmyPDF concurrent task executor."""

    @abstractmethod
    def make_queue(self) -> Queue[logging.LogRecord]:
        """Internal shared queue."""

    @abstractmethod
    def make_lock(self) -> SharedLock:
        """Generate an instance of this class's shared locking apparatus."""

    @abstractmethod
    def make_exe(
        self,
        *,
        max_workers: int,
        worker_initializer: Callable[[], None],
    ) -> StdlibExecutor:
        """Internal executor type."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pool_lock = self.make_lock()
        self._inner_executor: StdlibExecutor | None = None
        self._log_queue: Queue[logging.LogRecord] | None = None
        self._listener: threading.Thread | None = None

    @property
    def pool_lock(self) -> SharedLock:
        return self._pool_lock

    def _initialize_workers(
        self,
        *,
        max_workers: int,
        worker_initializer: Callable,
    ) -> None:
        assert self._log_queue is None
        self._log_queue = self.make_queue()
        assert self._inner_executor is None
        self._inner_executor = self.make_exe(
            max_workers=max_workers,
            worker_initializer=worker_initializer,
        )

    def __enter__(self) -> Self:
        assert self._log_queue is not None
        assert isinstance(self._inner_executor, StdlibExecutor)

        assert self._listener is None

        # Regardless of whether we use_threads for worker processes, the log_listener
        # must be a thread. Make sure we create the listener after the worker pool,
        # so that it does not get forked into the workers.
        # If use_threads is False, we are currently guilty of creating a thread before
        # forking on Linux, which is not recommended. However, we take a big
        # performance hit in pdfinfo if we can't fork. Long term solution is to
        # replace most of this with an asyncio implementation, and probably to
        # migrate some of pdfinfo into C++ or Rust.
        self._listener = threading.Thread(target=log_listener, args=(self._log_queue,))
        self._listener.start()

        # NB: We want to call the parent locking logic *after* starting the background thread.
        super().__enter__()

        assert isinstance(self._inner_executor, StdlibExecutor)
        self._inner_executor.__enter__()

        return self

    @abstractmethod
    def _forward_shutdown_strategy(
        self,
        inner: StdlibExecutor,
        listener: threading.Thread,
        q: Queue[logging.LogRecord],
        exc_type: type[E] | None,
        exc_val: E | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        pass

    def __exit__(
        self,
        exc_type: type[E] | None,
        exc_val: E | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        assert self._inner_executor is not None
        assert self._log_queue is not None
        assert self._listener is not None

        # Terminate log listener. This will never block, so we can call it immediately in all cases.
        self._log_queue.put_nowait(None)
        # NB: This is equivalent to having released the queue!

        if exc_val is None:
            assert exc_type is None and exc_tb is None

            # NB: Because this is the success case, we actually do *not* want to release the pool
            #     lock etc yet!
            assert not self._forward_shutdown_strategy(
                inner=self._inner_executor,
                listener=self._listener,
                q=self._log_queue,
                exc_type=exc_type,
                exc_val=exc_val,
                exc_tb=exc_tb,
            )
            # We're done waiting! Now release the shared resources, and prepare for
            # a future execution!
            assert not super().__exit__(exc_type, exc_val, exc_tb)

            # Since we have returned successfully, we can wait for the listener thread to exit.
            # (If an exception occurs, we don't try to join, in case it deadlocks.)
            assert self._log_queue.empty(), self._log_queue
            assert not self._listener.is_alive(), self._listener

            # Reset the context-specific state. This technically enables reuse upon successful exit.
            self._inner_executor = None
            self._log_queue = None
            self._listener = None
            return None

        assert exc_type is not None and exc_tb is not None
        # Unlike in the success case, we do not clobber the context-specific state, in case it
        # becomes needed to recover or log the failure.

        # NB: the internal executor will always be the most problematic resource to release.
        #     Let's make sure to release the pool lock and close off the progress bar before turning
        #     our attention to the complex machinations of the wrapped executor with user code.
        if super().__exit__(exc_type, exc_val, exc_tb):
            return True

        return self._forward_shutdown_strategy(
            inner=self._inner_executor,
            listener=self._listener,
            q=self._log_queue,
            exc_type=exc_type,
            exc_val=exc_val,
            exc_tb=exc_tb,
        )

    def _execute(
        self,
        *,
        task: Callable,
        task_arguments: Iterable,
        task_finished: Callable,
    ) -> None:
        assert isinstance(self._inner_executor, StdlibExecutor)
        futures = [self._inner_executor.submit(task, *args) for args in task_arguments]
        for future in as_completed(futures):
            result = future.result()
            task_finished(result, self.pbar)


class _ParallelismFrameworkExecutor[
    Lock: SharedLock,
    Q: Queue[logging.LogRecord],
    Exe: StdlibExecutor,
](_ConcurrentExecutorBase):
    @staticmethod
    @abstractmethod
    def _try_lock_type() -> LockGenerator[Lock] | None:
        """Try to access the base type definition for the shared lock."""


    def make_lock(self) -> Lock:
        return self.__class__._try_lock_type()() # type: ignore[misc]

    @staticmethod
    @abstractmethod
    def _queue_type() -> QueueGenerator[Q]:
        pass

    def make_queue(self) -> Q:
        return self.__class__._queue_type()(-1)

    @staticmethod
    @abstractmethod
    def _executor_class() -> StandardExecutorGeneration[Q, Exe]:
        """Return a concrete type definition implementing the stdlib ``Executor`` interface."""

    @staticmethod
    @abstractmethod
    def _worker_init(q: Q, user_init: UserInit, loglevel: int) -> None:
        """Begin a pool worker."""

    def make_exe(
        self,
        *,
        max_workers: int,
        worker_initializer: Callable[[], None],
    ) -> Exe:
        cls = self.__class__._executor_class()
        return cls(
            max_workers=max_workers,
            initializer=self.__class__._worker_init, # type: ignore[arg-type]
            initargs=(self._log_queue, worker_initializer, logging.getLogger("").level), # type: ignore[arg-type]
        )

    @classmethod
    def specialize(
        cls,
        workload: WorkloadKind,
        *,
        pbar_class: type[ProgressBar] | None = None,
    ) -> Self:
        """Now need to implement this in *shared* form!"""
        raise NotImplementedError(f"cannot specialize based on workload {workload!r}")


class _IPCExecutor(_ParallelismFrameworkExecutor[
    'multiprocessing.synchronize.SemLock', # type: ignore[type-var]
    'multiprocessing.queues.Queue',
    'ProcessPoolExecutor',

]):
    @staticmethod
    def _queue_type() -> QueueGenerator['multiprocessing.queues.Queue']:
        return multiprocessing.Queue # type: ignore[return-value]

    @staticmethod
    @functools.cache
    def _try_lock_type() -> LockGenerator['multiprocessing.synchronize.SemLock'] | None: # type: ignore[type-var]
        """Check if inter-process semaphore support is available.

        Some execution environments like AWS Lambda and Termux do not support
        semaphores.
        """
        try:
            # pylint: disable=import-outside-toplevel
            from multiprocessing.synchronize import Lock

            return multiprocessing.Lock
        except ImportError:
            return None

    @staticmethod
    @functools.cache
    def _executor_class() -> StandardExecutorGeneration[
        'multiprocessing.queues.Queue',
        'ProcessPoolExecutor',
    ]:
        return _IPCExecutorWrapper()

    @staticmethod
    def _worker_init(
        q: 'multiprocessing.queues.Queue', user_init: UserInit, loglevel: int
    ) -> None:
        process_init(q, user_init, loglevel)


class _ThreadedExecutor(_ParallelismFrameworkExecutor[
    'threading.Lock',           # type: ignore[type-var]
    'queue.Queue',
    'ThreadPoolExecutor',
]):
    @staticmethod
    def _queue_type() -> QueueGenerator['queue.Queue']:
        return queue.Queue      # type: ignore[return-value]

    @staticmethod
    @functools.cache
    def _try_lock_type() -> LockGenerator['threading.Lock'] | None: # type: ignore[type-var]
        """Check if in-memory semaphore (a standard lock) is available."""
        try:
            # pylint: disable=import-outside-toplevel
            from threading import Lock

            return Lock
        except ImportError:
            return None

    @staticmethod
    @functools.cache
    def _executor_class() -> StandardExecutorGeneration[
        'queue.Queue',
        'ThreadPoolExecutor'
    ]:
        return _ThreadedExecutorWrapper()

    @staticmethod
    def _worker_init(q: 'queue.Queue', user_init: UserInit, loglevel: int) -> None:
        thread_init(q, user_init, loglevel)


class StandardExecutor:
    def __init__(
        self,
        *args,
        impl: type[_ParallelismFrameworkExecutor],
        **kwargs,
    ) -> None:
        self._instance = impl(*args, **kwargs)
        # Delay until the construction of the executor to allow for any in-process pytest patching.
        self._executing_within_pytest = bool(os.environ.get("PYTEST_CURRENT_TEST", ""))

    _maximum_timeout_wait = 2.0 # 2000ms = 1 second

    @property
    def pool_lock(self) -> SharedLock:
        return self._instance.pool_lock

    def _forward_shutdown_strategy(
        self,
        inner: StdlibExecutor,
        listener: threading.Thread,
        q: Queue[logging.LogRecord],
        exc_type: type[E] | None,
        exc_val: E | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        handler = self._instance.__class__._executor_class()

        if exc_type is None:
            assert exc_val is None and exc_tb is None
            handler.perform_shutdown_strategy(
                inner=inner,
                listener=listener,
                q=q,
                wait=_WaitBehavior.BLOCK_INDEFINITELY,
                cancel=_CancelBehavior.NO_CANCELLATION,
                timeout_max=None,
            )
            return None

        assert exc_type is not None and exc_val is not None and exc_tb is not None

        if self._executing_within_pytest:
            # NB: Normally, we shutdown without waiting for other child workers on error, because
            #     there is no point in waiting for them (their results will be discarded).
            #
            #     But if we are running in pytest, we want everything to exit as cleanly as possible
            #     so that we're likely to get more useful error messages.
            if issubclass(exc_type, Exception):  # (not KeyboardInterrupt)
                handler.perform_shutdown_strategy(
                    inner=inner,
                    listener=listener,
                    q=q,
                    wait=_WaitBehavior.BLOCK_INDEFINITELY,
                    cancel=_CancelBehavior.NO_CANCELLATION,
                    timeout_max=None,
                )
                return None

        if issubclass(exc_type, CancelRunningTasksMixin):
            handler.perform_shutdown_strategy(
                inner=inner,
                listener=listener,
                q=q,
                wait=_WaitBehavior.TIMEOUT_MAX,
                cancel=_CancelBehavior.CANCEL_RUNNING_TASKS,
                timeout_max=self.__class__._maximum_timeout_wait,
            )
            return None

        handler.perform_shutdown_strategy(
            inner=inner,
            listener=listener,
            q=q,
            wait=_WaitBehavior.TIMEOUT_MAX,
            cancel=_CancelBehavior.NO_CANCELLATION,
            timeout_max=self.__class__._maximum_timeout_wait,
        )
        return None

    @classmethod
    def specialize(
        cls,
        workload: WorkloadKind | None = None,
        *,
        pbar_class: type[ProgressBar] | None = None,
    ) -> Self:
        return cls(
            impl=setup_executor(workload or WorkloadKind.MORE_MESSAGING), # type: ignore[arg-type]
            pbar_class=pbar_class,
        )

    def make_queue(self) -> 'multiprocessing.queues.Queue | queue.Queue':
        return self._instance.make_queue()

    def make_lock(self) -> 'multiprocessing.synchronize.SemLock | threading.Lock':
        return self._instance.make_lock()

    def make_exe(
        self,
        *,
        max_workers: int,
        worker_initializer: Callable,
    ) -> 'ProcessPoolExecutor | ThreadPoolExecutor':
        return self._instance.make_exe(
            max_workers=max_workers,
            worker_initializer=worker_initializer,
        )

    def __enter__(self) -> Self:
        self._instance.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[E] | None,
        exc_val: E | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        return self._instance.__exit__(exc_type, exc_val, exc_tb)

    def __call__[T](
        self,
        *,
        max_workers: int,
        progress_kwargs: Mapping[str, Any],
        worker_initializer: Callable | None = None,
        task: Callable[..., T] | None = None,
        task_arguments: Iterable | None = None,
        task_finished: Callable[[T, ProgressBar], None] | None = None,
    ) -> None:
        self._instance.__call__(
            max_workers=max_workers,
            progress_kwargs=progress_kwargs,
            worker_initializer=worker_initializer,
            task=task,
            task_arguments=task_arguments,
            task_finished=task_finished,
        )


@hookimpl
def get_executor(progressbar_class) -> Executor:
    """Return the default executor."""
    return StandardExecutor.specialize(pbar_class=progressbar_class)


RICH_CONSOLE = RichConsole(stderr=True)


@hookimpl
def get_progressbar_class():
    """Return the default progress bar class."""

    def partial_RichProgressBar(*args, **kwargs):
        return RichProgressBar(*args, **kwargs, console=RICH_CONSOLE)

    return partial_RichProgressBar


@hookimpl
def get_logging_console():
    """Return the default logging console handler."""
    return RichLoggingHandler(console=RICH_CONSOLE)
