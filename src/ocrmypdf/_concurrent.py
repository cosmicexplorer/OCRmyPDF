# SPDX-FileCopyrightText: 2022 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

"""OCRmyPDF concurrency abstractions."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Callable, Iterable, Mapping
from enum import Enum, auto
from types import TracebackType
from typing import Any, ClassVar, Generic, TypeVar, Protocol, runtime_checkable
from typing_extensions import Self

from ocrmypdf._progressbar import NullProgressBar, ProgressBar

T = TypeVar('T')

E = TypeVar('E', bound=BaseException)


def _task_noop(*_args, **_kwargs) -> None:
    return


def _task_finished_noop(_result: Any, pbar: ProgressBar):
    pbar.update()


@runtime_checkable
class SharedLock(Protocol):
    def locked(self) -> bool:
        """Whether this is currently locked."""

    def __enter__(self) -> Self:
        """Enter the lock."""

    def __exit__(self, *args) -> bool | None:
        """Release the lock."""


class WorkloadKind(Enum):
    """Characterize the expected workload by frequency/urgency of data sharing."""
    MORE_MESSAGING = auto()
    MORE_SHARED_DATA = auto()


@runtime_checkable
class Executor[Lock: SharedLock](Protocol[Lock]):
    @property
    def pool_lock(self) -> Lock:
        pass

    @classmethod
    def specialize(
        cls,
        workload: WorkloadKind,
        *,
        pbar_class: type[ProgressBar] | None = None,
    ) -> Self:
        pass

    def __enter__(self) -> Self:
        pass

    def __exit__(
        self,
        exc_type: type[E] | None,
        exc_val: E | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        pass

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
        pass


class ExecutorBase[Lock: SharedLock](ABC):
    """Abstract concurrent executor."""

    def __init__(self, *args, pbar_class: type[ProgressBar] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._pbar_class: type[ProgressBar] = pbar_class or NullProgressBar
        self.pbar: ProgressBar | None = None

    @abstractproperty
    def pool_lock(self) -> Lock:
        """Return a semaphore to synchronize any shared state."""

    @classmethod
    def specialize(
        cls,
        workload: WorkloadKind,
        *,
        pbar_class: type[ProgressBar] | None = None,
    ) -> Self:
        """Set up parallel execution and progress reporting.

        Args:
            workload: If ``WorkloadKind.MORE_MESSAGING``, the workload is the sort that will
                benefit from running in a multiprocessing context (for example, it uses Python
                heavily, and parallelizing it with threads is not expected to be
                performant).
            pbar_class: An override for progress reporting, as desired.
        """
        return cls(pbar_class=pbar_class)

    def _initialize_progress_bar(self, **progress_kwargs: Any) -> None:
        assert self.pbar is None
        self.pbar = self._pbar_class(**progress_kwargs)

    @abstractmethod
    def _initialize_workers(
        self,
        *,
        max_workers: int,
        worker_initializer: Callable,
    ) -> None:
        """Initialize a set of worker threads and go to town!"""

    def __enter__(self) -> Self:
        assert self.pbar is not None
        assert isinstance(self.pbar, ProgressBar)
        self.pbar.__enter__()
        assert isinstance(self.pool_lock, SharedLock)
        self.pool_lock.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[E] | None,
        exc_val: E | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        assert self.pbar is not None
        pbar, self.pbar = self.pbar, None
        if self.pool_lock.__exit__(exc_type, exc_val, exc_tb):
            return True
        if pbar.__exit__(exc_type, exc_val, exc_tb):
            return True

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
        """Set up parallel execution and progress reporting.

        Args:
            max_workers: The maximum number of workers that should be run.
            progress_kwargs: Arguments to set up the progress bar.
            worker_initializer: Called when a worker is initialized, in the worker's
                execution context. If the child workers are processes, it must be
                possible to marshall/pickle the worker initializer.
                ``functools.partial`` can be used to bind parameters.
            task: Called when the worker starts a new task, in the worker's execution
                context. Must be possible to marshall to the worker.
            task_finished: Called when a worker finishes a task, in the parent's
                context.
            task_arguments: An iterable that generates a group of parameters for each
                task. This runs in the parent's context, but the parameters must be
                marshallable to the worker.
        """
        if not task_arguments:
            return  # Nothing to do!
        if not worker_initializer:
            worker_initializer = _task_noop
        if not task_finished:
            task_finished = _task_finished_noop
        if not task:
            task = _task_noop

        self._initialize_progress_bar(**progress_kwargs)
        self._initialize_workers(
            max_workers=max_workers,
            worker_initializer=worker_initializer,
        )
        with self:
            self._execute(
                task=task,
                task_arguments=task_arguments,
                task_finished=task_finished,
            )

    @abstractmethod
    def _execute(
        self,
        *,
        task: Callable,
        task_arguments: Iterable,
        task_finished: Callable,
    ) -> None:
        """Custom executors should override this method."""


def setup_executor(plugin_manager) -> Executor:
    pbar_class = plugin_manager.get_progressbar_class()
    return plugin_manager.get_executor(progressbar_class=pbar_class)


class SerialLock:
    __slots__ = ('_locked',)
    def __init__(self):
        self._locked = False
    def locked(self) -> bool:
        return self._locked
    def __enter__(self) -> Self:
        self._locked = True
        return self
    def __exit__(self, *args) -> bool | None:
        self._locked = False


assert issubclass(SerialLock, SharedLock)


class SerialExecutor(ExecutorBase[SerialLock]):
    """Implements a purely sequential executor using the parallel protocol.

    The current process/thread will be the worker that executes all tasks
    in order. As such, ``worker_initializer`` will never be called.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pool_lock = SerialLock()

    @property
    def pool_lock(self) -> SerialLock:
        return self._pool_lock

    def _initialize_workers(
        self,
        *,
        max_workers: int,
        worker_initializer: Callable,
    ) -> None:
        pass

    def _execute(
        self,
        *,
        task: Callable,
        task_arguments: Iterable,
        task_finished: Callable,
    ) -> None:  # pylint: disable=unused-argument
        for args in task_arguments:
            result = task(*args)
            task_finished(result, self.pbar)


assert isinstance(SerialExecutor(), Executor)
