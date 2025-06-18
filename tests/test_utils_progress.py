"""Tests for progress utility functions."""

from unittest.mock import Mock, patch
from langchain_memvid.utils.progress import ProgressDisplay


class TestProgressDisplay:
    """Test cases for ProgressDisplay class."""

    def test_init_with_show_progress_true(self):
        """Test initialization with show_progress=True."""
        progress = ProgressDisplay(show_progress=True)
        assert progress.show_progress is True

    def test_init_with_show_progress_false(self):
        """Test initialization with show_progress=False."""
        progress = ProgressDisplay(show_progress=False)
        assert progress.show_progress is False

    def test_init_default_show_progress(self):
        """Test initialization with default show_progress value."""
        progress = ProgressDisplay()
        assert progress.show_progress is True

    @patch('langchain_memvid.utils.progress.tqdm_base')
    def test_tqdm_with_show_progress_true(self, mock_tqdm):
        """Test tqdm method when show_progress is True."""
        progress = ProgressDisplay(show_progress=True)
        mock_iterable = [1, 2, 3]
        mock_tqdm_instance = Mock()
        mock_tqdm.return_value = mock_tqdm_instance

        result = progress.tqdm(mock_iterable, desc="Test")

        mock_tqdm.assert_called_once_with(mock_iterable, desc="Test")
        assert result == mock_tqdm_instance

    def test_tqdm_with_show_progress_false_with_iterable(self):
        """Test tqdm method when show_progress is False with iterable."""
        progress = ProgressDisplay(show_progress=False)
        test_iterable = [1, 2, 3]

        result = list(progress.tqdm(test_iterable, desc="Test"))

        assert result == [1, 2, 3]

    def test_tqdm_with_show_progress_false_without_iterable(self):
        """Test tqdm method when show_progress is False without iterable."""
        progress = ProgressDisplay(show_progress=False)

        result = progress.tqdm(desc="Test")

        assert result == []

    def test_tqdm_with_show_progress_false_with_none_iterable(self):
        """Test tqdm method when show_progress is False with None iterable."""
        progress = ProgressDisplay(show_progress=False)

        result = progress.tqdm(None, desc="Test")

        assert result == []

    @patch('langchain_memvid.utils.progress.tqdm_base')
    def test_tqdm_passes_kwargs(self, mock_tqdm):
        """Test that tqdm passes all kwargs to the underlying tqdm function."""
        progress = ProgressDisplay(show_progress=True)
        mock_tqdm_instance = Mock()
        mock_tqdm.return_value = mock_tqdm_instance

        progress.tqdm(
            [1, 2, 3],
            desc="Test Description",
            total=100,
            unit="items",
            colour="green"
        )

        mock_tqdm.assert_called_once_with(
            [1, 2, 3],
            desc="Test Description",
            total=100,
            unit="items",
            colour="green"
        )

    @patch('langchain_memvid.utils.progress.tqdm_base')
    def test_progress_context_manager_with_show_progress_true(self, mock_tqdm):
        """Test progress context manager when show_progress is True."""
        progress = ProgressDisplay(show_progress=True)
        mock_tqdm_instance = Mock()
        mock_tqdm.return_value.__enter__.return_value = mock_tqdm_instance
        mock_tqdm.return_value.__exit__.return_value = None

        with progress.progress(desc="Test") as pbar:
            assert pbar == mock_tqdm_instance
            pbar.update(10)

        mock_tqdm.assert_called_once_with(desc="Test")
        mock_tqdm_instance.update.assert_called_once_with(10)

    def test_progress_context_manager_with_show_progress_false(self):
        """Test progress context manager when show_progress is False."""
        progress = ProgressDisplay(show_progress=False)

        with progress.progress(desc="Test") as pbar:
            # Should be able to call update without error
            pbar.update(10)
            pbar.update(20)
            pbar.close()

        # Verify it's a NopProgress instance
        assert hasattr(pbar, 'update')
        assert hasattr(pbar, 'close')
        assert callable(pbar.update)
        assert callable(pbar.close)

    def test_nop_progress_methods(self):
        """Test that NopProgress methods don't raise errors."""
        progress = ProgressDisplay(show_progress=False)

        with progress.progress() as pbar:
            # These should not raise any exceptions
            pbar.update()
            pbar.update(10)
            pbar.update(20, refresh=True)
            pbar.close()

    @patch('langchain_memvid.utils.progress.tqdm_base')
    def test_progress_context_manager_passes_kwargs(self, mock_tqdm):
        """Test that progress context manager passes all kwargs to tqdm."""
        progress = ProgressDisplay(show_progress=True)
        mock_tqdm_instance = Mock()
        mock_tqdm.return_value.__enter__.return_value = mock_tqdm_instance
        mock_tqdm.return_value.__exit__.return_value = None

        with progress.progress(
            desc="Test Description",
            total=100,
            unit="items",
            colour="blue"
        ):
            pass

        mock_tqdm.assert_called_once_with(
            desc="Test Description",
            total=100,
            unit="items",
            colour="blue"
        )

    def test_progress_context_manager_exception_handling(self):
        """Test that progress context manager handles exceptions properly."""
        progress = ProgressDisplay(show_progress=False)

        try:
            with progress.progress() as pbar:
                # Use pbar to avoid linter warning
                pbar.update(1)
                raise ValueError("Test exception")
        except ValueError:
            # Exception should be re-raised
            pass

    @patch('langchain_memvid.utils.progress.tqdm_base')
    def test_progress_context_manager_exception_handling_with_tqdm(self, mock_tqdm):
        """Test that progress context manager handles exceptions with tqdm properly."""
        progress = ProgressDisplay(show_progress=True)
        mock_tqdm_instance = Mock()
        mock_tqdm.return_value.__enter__.return_value = mock_tqdm_instance
        mock_tqdm.return_value.__exit__.return_value = None

        try:
            with progress.progress() as pbar:
                # Use pbar to avoid linter warning
                pbar.update(1)
                raise ValueError("Test exception")
        except ValueError:
            # Exception should be re-raised
            pass

        # Ensure __exit__ was called on the context manager, not the instance
        mock_tqdm.return_value.__exit__.assert_called_once()

    def test_multiple_progress_instances(self):
        """Test that multiple ProgressDisplay instances work independently."""
        progress1 = ProgressDisplay(show_progress=True)
        progress2 = ProgressDisplay(show_progress=False)

        assert progress1.show_progress is True
        assert progress2.show_progress is False

        # Test that they behave differently - compare the types of results
        result1 = progress1.tqdm([1, 2, 3])
        result2 = progress2.tqdm([1, 2, 3])

        # result1 should be a tqdm object, result2 should be a list
        assert hasattr(result1, 'update')  # tqdm object has update method
        assert isinstance(result2, list)   # list object

    def test_nop_progress_attributes(self):
        """Test that NopProgress has the expected attributes."""
        progress = ProgressDisplay(show_progress=False)

        with progress.progress() as pbar:
            # Check that it has the expected methods
            assert hasattr(pbar, 'update')
            assert hasattr(pbar, 'close')

            # Check that methods are callable
            assert callable(pbar.update)
            assert callable(pbar.close)

    def test_tqdm_with_empty_iterable(self):
        """Test tqdm with empty iterable."""
        progress = ProgressDisplay(show_progress=False)
        empty_list = []

        result = list(progress.tqdm(empty_list))

        assert result == []

    @patch('langchain_memvid.utils.progress.tqdm_base')
    def test_tqdm_with_empty_iterable_and_progress(self, mock_tqdm):
        """Test tqdm with empty iterable when progress is enabled."""
        progress = ProgressDisplay(show_progress=True)
        empty_list = []
        mock_tqdm_instance = Mock()
        mock_tqdm.return_value = mock_tqdm_instance

        result = progress.tqdm(empty_list)

        mock_tqdm.assert_called_once_with(empty_list)
        assert result == mock_tqdm_instance
