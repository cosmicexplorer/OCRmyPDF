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
from abc import abstractmethod, abstractproperty
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import (
    Executor as StdlibExecutor,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from contextlib import suppress
from typing import TYPE_CHECKING

from rich.console import Console as RichConsole

from ocrmypdf import Executor, ExecutorBase, WorkloadKind, SharedLock, hookimpl
from ocrmypdf._logging import RichLoggingHandler
from ocrmypdf._progressbar import ProgressBar, RichProgressBar
from ocrmypdf.exceptions import InputFileError
from ocrmypdf.helpers import remove_all_log_handlers

if TYPE_CHECKING:
    from types import TracebackType
    from typing import Any, Protocol, TypeAlias, TypeVar, runtime_checkable
    from typing_extensions import Self

    @runtime_checkable
    class Queue[T](Protocol[T]):
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
    class ExecutorPlatform[
        Lock: SharedLock,
        Q: Queue[logging.LogRecord],
        Exe: StdlibExecutor,
    ](Executor[Lock]):
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


def process_init(q: Queue[logging.LogRecord], user_init: UserInit, loglevel: int) -> None:
    """Initialize a process pool worker."""
    # Ignore SIGINT (our parent process will kill us gracefully)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

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


def thread_init(q: Queue[logging.LogRecord], user_init: UserInit, loglevel: int) -> None:
    """Begin a thread pool worker."""
    del q  # unused but required argument
    del loglevel  # unused but required argument
    # As a thread, block SIGBUS so the main thread deals with it...
    with suppress(AttributeError):
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGBUS, signal.SIGINT})

    user_init()
    return


@functools.cache
def setup_executor(workload: WorkloadKind) -> type[_ParallelismFrameworkExecutor]:
    # Check if semaphore support is available, and if not, fall back to using threads.
    match workload:
        case WorkloadKind.MORE_MESSAGING:
            if lock_type := _IPCExecutor._try_lock_type():
                return _IPCExecutor
            # We currently expect inter-thread locks to always be available.
            _basic_logger.info("ipc semaphore unavailable for workload %s (falling back to %s)",
                               workload, WorkloadKind.MORE_SHARED_DATA)
            return setup_executor(WorkloadKind.MORE_SHARED_DATA)
        case WorkloadKind.MORE_SHARED_DATA:
            if lock_type := _ThreadedExecutor._try_lock_type():
                return _ThreadedExecutor
            _basic_logger.info("in-memory semaphore unavailable for workload %s (no fallback)",
                               workload)
    raise RuntimeError(f"could not produce executor for given workload type {workload!r}")


class _ConcurrentExecutorBase[
    Lock: SharedLock,
    Q: Queue[logging.LogRecord],
    Exe: StdlibExecutor
](ExecutorBase[Lock]):
    """Standard OCRmyPDF concurrent task executor."""
    @abstractmethod
    def make_queue(self) -> Q:
        """Internal shared queue."""

    @abstractmethod
    def make_lock(self) -> Lock:
        """Generate an instance of this class's shared locking apparatus."""

    @abstractmethod
    def make_exe(
        self,
        *,
        max_workers: int,
        worker_initializer: Callable,
    ) -> Exe:
        """Internal executor type."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pool_lock = self.make_lock()
        self._inner_executor: Exe | None = None
        self._log_queue: Q | None = None
        self._listener: threading.Thread | None = None

    @property
    def pool_lock(self) -> Lock:
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
        assert isinstance(self._log_queue, Queue)
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
        self._listener = threading.Thread(target=log_listener, args=(self.queue,))
        listener.start()

        # NB: We want to call the parent locking logic *after* starting the background thread.
        super().__enter__()

        assert isinstance(self._inner_executor, StdlibExecutor)
        self._inner_executor.__enter__()

        return self

    def __exit__(
        self,
        exc_type: type[E] | None,
        exc_val: E | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        assert self._log_queue is not None
        assert self._listener is not None

        # Terminate log listener. This will never block, so we can call it immediately in all cases.
        self._log_queue.put_nowait(None)

        if exc_val is None:
            assert exc_type is None and exc_tb is None
            assert not self._inner_executor.__exit__(exc_type, exc_val, exc_tb)
            self._inner_executor = None
            assert not super().__exit__(exc_type, exc_val, exc_tb)

            # Since we have returned successfully, we can wait for the listener thread to exit.
            # (If an exception occurs, we don't try to join, in case it deadlocks.)
            self._listener.join()

            # Reset the context-specific state. This technically enables reuse upon successful exit.
            self._log_queue = None
            self._listener = None
            return

        assert exc_type is not None and exc_tb is not None
        # Unlike in the success case, we do not clobber the context-specific state, in case it
        # becomes needed to recover or log the failure.
        if self._inner_executor.__exit__(exc_type, exc_val, exc_tb):
            return True
        return super().__exit__(exc_type, exc_val, exc_tb)

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
    Exe: StdlibExecutor
](_ConcurrentExecutorBase[Lock, Q, Exe]):
    @staticmethod
    @abstractmethod
    def _try_lock_type() -> type[Lock] | None:
        """Try to access the base type definition for the shared lock."""

    @staticmethod
    @abstractmethod
    def _executor_class() -> type[Exe]:
        """Return a concrete type definition implementing the stdlib ``Executor`` interface."""

    @staticmethod
    @abstractmethod
    def _worker_init(q: Q, user_init: UserInit, loglevel: int) -> None:
        """Begin a pool worker."""

    def make_exe(
        self,
        *,
        max_workers: int,
        worker_initializer: Callable,
    ) -> Exe:
        assert isinstance(self._log_queue, Queue)
        cls = self.__class__._executor_class()
        return cls(
            max_workers=max_workers,
            initializer=self.__class__._worker_init,
            initargs=(self._log_queue, worker_initializer, logging.getLogger("").level),
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
    'multiprocessing.Lock',
    'multiprocessing.queues.Queue',
    'ProcessPoolExecutor',
]):
    @staticmethod
    def _queue_type() -> type['multiprocessing.queues.Queue']:
        return multiprocessing.queues.Queue

    def make_queue(self) -> 'multiprocessing.queues.Queue':
        return self.__class__._queue_type(-1)

    @staticmethod
    @functools.cache
    def _try_lock_type() -> type['multiprocessing.Lock'] | None:
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

    def make_lock(self) -> 'multiprocessing.Lock':
        return self.__class__._try_lock_type()()

    @staticmethod
    def _executor_class() -> type[ProcessPoolExecutor]:
        return ProcessPoolExecutor

    @staticmethod
    def _worker_init(q: 'multiprocessing.queues.Queue', user_init: UserInit, loglevel: int) -> None:
        process_init(q, user_init, loglevel)


class _ThreadedExecutor(_ParallelismFrameworkExecutor[
    'threading.Lock',
    'queue.Queue',
    'ThreadPoolExecutor',
]):
    @staticmethod
    def _queue_type() -> type['queue.Queue']:
        return queue.Queue

    def make_queue(self) -> 'queue.Queue':
        return self.__class__._queue_type(-1)

    @staticmethod
    @functools.cache
    def _try_lock_type() -> type['threading.Lock'] | None:
        """Check if in-memory semaphore (a standard lock) is available."""
        try:
            # pylint: disable=import-outside-toplevel
            from threading import Lock

            return Lock
        except ImportError:
            return None

    def make_lock(self) -> 'threading.Lock':
        return self.__class__._try_lock_type()()

    @staticmethod
    def _executor_class() -> type[ThreadPoolExecutor]:
        return ThreadPoolExecutor

    @staticmethod
    def _worker_init(q: 'queue.Queue', user_init: UserInit, loglevel: int) -> None:
        thread_init(q, user_init, loglevel)


class StandardExecutor:
    def __init__(
        self,
        *args,
        impl: type[_ParallelismFrameworkExecutor],
        **kwargs,
    ):
        self._instance = impl(*args, **kwargs)
        # Delay until the construction of the executor to allow for any in-process pytest patching.
        self._executing_within_pytest = bool(os.environ.get("PYTEST_CURRENT_TEST", ""))

    @classmethod
    def specialize(
        cls,
        workload: WorkloadKind | None = None,
        *,
        pbar_class: type[ProgressBar] | None = None,
    ) -> Self:
        return cls(
            impl=setup_executor(workload or WorkloadKind.MORE_MESSAGING),
            pbar_class=pbar_class,
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
        if exc_val is None:
            assert exc_type is None and exc_tb is None

            return super().__exit__(None, None, None)

        assert exc_type is not None and exc_tb is not None

        import pdb; pdb.set_trace()
        if self._executing_within_pytest:
            # NB: Normally, we shutdown without waiting for other child workers on error, because
            #     there is no point in waiting for them (their results will be discarded).
            #
            #     But if we are running in pytest, we want everything to exit as cleanly as possible
            #     so that we're likely to get more useful error messages.
            if issubclass(exc_type, Exception): # (not KeyboardInterrupt)
                import pdb; pdb.set_trace()


        import pdb; pdb.set_trace()
        self._inner_executor.shutdown(wait=False, cancel_futures=True)
        # Unlike in the success case, we do not clobber the context-specific state, in case it
        # becomes needed to recover or log the failure.
        return False

    def make_queue(self) -> 'multiprocessing.queues.Queue | queue.Queue':
        return self._instance.make_queue()

    def make_lock(self) -> 'multiprocessing.Lock | threading.Lock':
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

    def __call__(
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


if TYPE_CHECKING:
    assert issubclass(
        StandardExecutor,
        ExecutorPlatform[
            'multiprocessing.Lock | threading.Lock',
            'multiprocessing.queues.Queue | queue.Queue',
            'ProcessPoolExecutor | ThreadPoolExecutor',
        ]
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
