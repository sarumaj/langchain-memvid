"""Utility functions for MemVid."""

from typing import Any, Iterable, Optional, TypeVar
from contextlib import contextmanager
from tqdm import tqdm as tqdm_base

T = TypeVar('T')


class ProgressDisplay:
    """Wrapper for tqdm progress bar that can be disabled."""

    def __init__(self, show_progress: bool = True):
        """Initialize progress display.

        Args:
            show_progress: Whether to show progress bars
        """
        self.show_progress = show_progress

    def tqdm(
        self,
        iterable: Optional[Iterable[T]] = None,
        **kwargs: Any
    ) -> Iterable[T]:
        """Wrapper for tqdm that respects show_progress setting.

        Args:
            iterable: Iterable to wrap with progress bar
            **kwargs: Additional arguments to pass to tqdm

        Returns:
            Iterable with or without progress bar
        """
        if self.show_progress:
            return tqdm_base(iterable, **kwargs)
        return iterable or []

    @contextmanager
    def progress(self, **kwargs: Any):
        """Context manager for manual progress bar updates.

        Args:
            **kwargs: Arguments to pass to tqdm

        Yields:
            tqdm instance or dummy progress bar
        """
        if self.show_progress:
            with tqdm_base(**kwargs) as pbar:
                yield pbar
                return

        # Nop progress bar that does nothing
        class NopProgress:
            def update(self, *args, **kwargs): pass
            def close(self): pass

        yield NopProgress()
