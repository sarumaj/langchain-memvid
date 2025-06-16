"""Unit tests for Retriever."""

import json
import tempfile
from unittest.mock import Mock, patch
import pytest
import cv2
import numpy as np
import time
from unittest.mock import call

from langchain_memvid.retriever import (
    Retriever,
    RetrieverConfig,
    VideoMetadata,
    RetrieverStats
)
from langchain_memvid.index import IndexManager, IndexConfig, IndexStats


@pytest.fixture
def config():
    """Create a test configuration."""
    return RetrieverConfig(
        cache_size=100,
        max_workers=2
    )


@pytest.fixture
def mock_index_manager():
    """Create a mock index manager."""
    manager = Mock(spec=IndexManager)

    # Mock embedding model
    mock_embedding_model = Mock()
    mock_embedding_model.embed_query.return_value = [0.0] * 384
    mock_embedding_model.embedding_dimension = 384

    # Set up manager attributes
    manager.embedding_model = mock_embedding_model
    manager.config = IndexConfig(index_type="Flat", nlist=100)
    manager._dimension = 384
    manager._is_loaded = False

    # Mock search results
    manager.search.return_value = [
        (0, 0.1, {"frame": 0, "text": "Test chunk 1"}),
        (1, 0.2, {"frame": 1, "text": "Test chunk 2"})
    ]

    # Mock get_chunk_by_id
    manager.get_chunk_by_id.return_value = {"frame": 0, "text": "Test chunk"}

    # Mock get_stats to return IndexStats object
    manager.get_stats.return_value = IndexStats(
        total_chunks=2,
        total_frames=2,
        index_type="Flat",
        embedding_model="test-model",
        dimension=384,
        avg_chunks_per_frame=1.0,
        config={}
    )

    return manager


@pytest.fixture
def mock_video_file():
    """Create a temporary video file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        # Create a simple video file
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video.name, fourcc, 30.0, (640, 480))

        # Write a few frames
        for _ in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

        yield temp_video.name


@pytest.fixture
def mock_index_file():
    """Create a temporary index file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_index:
        # Write some test index data
        index_data = {
            "chunks": [
                {"id": 0, "text": "Test chunk 1", "frame": 0},
                {"id": 1, "text": "Test chunk 2", "frame": 1}
            ]
        }
        temp_index.write(json.dumps(index_data).encode())
        yield temp_index.name


@pytest.fixture
def retriever(config, mock_index_manager, mock_video_file, mock_index_file):
    """Create a test retriever instance."""
    return Retriever(
        video_file=mock_video_file,
        index_file=mock_index_file,
        config=config,
        index_manager=mock_index_manager
    )


def test_retriever_initialization(retriever, config, mock_index_manager):
    """Test retriever initialization."""
    assert retriever.config == config
    assert retriever.index_manager == mock_index_manager
    assert isinstance(retriever.video_metadata, VideoMetadata)
    assert retriever.video_metadata.total_frames == 10
    assert retriever.video_metadata.fps == 30.0


def test_retriever_initialization_invalid_video():
    """Test retriever initialization with invalid video file."""
    with pytest.raises(ValueError, match="Cannot open video file"):
        Retriever(
            video_file="nonexistent.mp4",
            index_file="index.json",
            config=RetrieverConfig(),
            index_manager=Mock(spec=IndexManager)
        )


def test_search(retriever):
    """Test basic search functionality."""
    results = retriever.search("test query", top_k=2)
    assert len(results) == 2
    assert "Test chunk" in results[0]
    retriever.index_manager.search.assert_called_once_with("test query", 2)


def test_search_with_metadata(retriever):
    """Test search with metadata."""
    results = retriever.search_with_metadata("test query", top_k=2)
    assert len(results) == 2
    assert all(isinstance(result, dict) for result in results)
    assert all("text" in result for result in results)
    assert all("score" in result for result in results)
    assert all("chunk_id" in result for result in results)
    assert all("frame" in result for result in results)
    assert all("metadata" in result for result in results)


def test_get_chunk_by_id(retriever):
    """Test getting chunk by ID."""
    chunk = retriever.get_chunk_by_id(0)
    assert chunk == "Test chunk"
    retriever.index_manager.get_chunk_by_id.assert_called_once_with(0)


def test_get_chunk_by_id_not_found(retriever):
    """Test getting non-existent chunk."""
    retriever.index_manager.get_chunk_by_id.return_value = None
    chunk = retriever.get_chunk_by_id(999)
    assert chunk is None


def test_get_context_window(retriever):
    """Test getting context window."""
    chunks = retriever.get_context_window(1, window_size=1)
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_prefetch_frames(retriever):
    """Test frame prefetching."""
    # Set a low prefetch threshold to ensure prefetching happens
    retriever.config.cache_prefetch_threshold = 2
    retriever.config.prefetch_batch_size = 2

    with patch('langchain_memvid.retriever.batch_extract_and_decode') as mock_decode:
        mock_decode.return_value = {0: "test data", 1: "test data 2"}

        # Prefetch frames
        retriever.prefetch_frames([0, 1, 2])

        # Verify mock was called twice with correct batch arguments
        assert mock_decode.call_count == 2
        mock_decode.assert_has_calls([
            call(
                retriever.video_file,
                [0, 1],  # First batch
                max_workers=retriever.config.max_workers
            ),
            call(
                retriever.video_file,
                [2],     # Second batch
                max_workers=retriever.config.max_workers
            )
        ])

        # Verify cache was updated
        assert 0 in retriever._frame_cache
        assert 1 in retriever._frame_cache
        assert len(retriever._frame_cache) == 2  # Only 2 frames should be cached


def test_clear_cache(retriever):
    """Test cache clearing."""
    # Add some data to cache
    retriever._frame_cache[0] = "test data"
    assert len(retriever._frame_cache) > 0

    # Clear cache
    retriever.clear_cache()
    assert len(retriever._frame_cache) == 0


def test_get_stats(retriever):
    """Test getting retriever statistics."""
    stats = retriever.get_stats()
    assert isinstance(stats, RetrieverStats)
    assert stats.video_file == retriever.video_file
    assert stats.total_frames == retriever.video_metadata.total_frames
    assert stats.fps == retriever.video_metadata.fps
    assert stats.cache_size == len(retriever._frame_cache)
    assert stats.cache_ttl == retriever.config.cache_ttl
    assert isinstance(stats.cache_age, float)
    assert isinstance(stats.config, dict)
    assert isinstance(stats.index_stats, dict)


def test_decode_single_frame(retriever):
    """Test single frame decoding."""
    with patch('langchain_memvid.retriever.extract_and_decode_cached') as mock_decode:
        mock_decode.return_value = "test data"
        result = retriever._decode_single_frame(0)
        assert result == "test data"
        mock_decode.assert_called_once_with(retriever.video_file, 0)


def test_decode_frames_parallel(retriever):
    """Test parallel frame decoding."""
    with patch('langchain_memvid.retriever.batch_extract_and_decode') as mock_decode:
        mock_decode.return_value = {0: "test data 1", 1: "test data 2"}
        results = retriever._decode_frames_parallel([0, 1])
        assert len(results) == 2
        assert results[0] == "test data 1"
        assert results[1] == "test data 2"
        mock_decode.assert_called_once()


def test_decode_frames_parallel_with_cache(retriever):
    """Test parallel frame decoding with cache."""
    # Add some data to cache with timestamp
    retriever._frame_cache[0] = ("cached data", time.time())

    with patch('langchain_memvid.retriever.batch_extract_and_decode') as mock_decode:
        mock_decode.return_value = {1: "test data"}
        results = retriever._decode_frames_parallel([0, 1])

        # Verify results
        assert results[0] == "cached data"  # From cache
        assert results[1] == "test data"    # From mock decode
        assert len(results) == 2

        # Verify mock was called only for uncached frame
        mock_decode.assert_called_once_with(
            retriever.video_file,
            [1],  # Only frame 1 should be decoded
            max_workers=retriever.config.max_workers
        )
