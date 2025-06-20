"""Shared test fixtures and utilities for langchain-memvid tests."""

import pytest
import tempfile
import shutil
import hashlib
import random
import string
from pathlib import Path
from unittest.mock import MagicMock
from typing import List, Union, Any, NamedTuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_memvid.config import VectorStoreConfig, VideoConfig, QRCodeConfig, IndexConfig
from langchain_memvid.index import SearchResult


def custom_parametrize(
    argnames: Union[str, List[str]],
    argvalues: List[Union[tuple, Any]],
    **kwargs
) -> pytest.MarkDecorator:
    """
    Custom parametrize decorator with automatic ID generation.

    Args:
        argnames: Parameter names (string or list of strings)
        argvalues: List of parameter value tuples (or single values if only one parameter)
        **kwargs: Additional arguments passed to pytest.mark.parametrize

    Returns:
        pytest.mark.parametrize decorator with automatic ID generation.
    """
    if isinstance(argnames, str):
        argnames_list = argnames.split(",")
    else:
        argnames_list = argnames

    class TestParameter(NamedTuple):
        name: str
        value: Any

    class TestCase(NamedTuple):
        prefix: str
        parameters: List[TestParameter]

    ids = []
    for idx, argvalues_list in enumerate(argvalues):
        test_case = TestCase(
            prefix="{{idx:0{size}d}}|".format(size=len(argvalues)//10+1).format(idx=idx+1),
            parameters=[]
        )
        if len(argnames_list) == 1:
            test_case.parameters.append(TestParameter(name=argnames_list[0], value=argvalues_list))
        else:
            test_case.parameters.extend((
                TestParameter(name=name, value=value) for name, value in zip(argnames_list, argvalues_list)
            ))

        id_string = test_case.prefix + ",".join(
            "{initials}:{value}".format(
                initials=''.join(map(lambda p: p[:1].upper(), arg.name.split('_'))),
                value=arg.value
            ) for arg in test_case.parameters
        )

        ids.append(id_string)

    return pytest.mark.parametrize(argnames, argvalues, ids=ids, **kwargs)


class MockEmbeddings(Embeddings):
    """Mock embeddings for testing."""

    def __init__(self, dimension: int = 3):
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Mock embed_documents method."""
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Mock embed_query method."""
        return [0.1, 0.2, 0.3]


class BenchmarkEmbeddings(Embeddings):
    """Embeddings implementation for benchmarking with realistic vector generation."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for documents."""
        embeddings = []
        for text in texts:
            # Generate deterministic but varied embeddings based on text content
            hash_value = int(hashlib.sha256(text.encode('utf-8')).hexdigest()[:8], 16)
            random.seed(hash_value)
            embedding = [random.uniform(-1, 1) for _ in range(self.dimension)]
            # Normalize to unit vector
            norm = sum(x*x for x in embedding) ** 0.5
            embedding = [x/norm for x in embedding]
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        return self.embed_documents([text])[0]


def generate_test_texts(count: int, min_length: int = 50, max_length: int = 200) -> List[str]:
    """Generate test texts of varying lengths."""
    texts = []
    for i in range(count):
        length = random.randint(min_length, max_length)
        # Generate realistic text with words
        words = []
        for _ in range(length // 5):  # Approximate 5 chars per word
            word_length = random.randint(3, 12)
            word = ''.join(random.choices(string.ascii_lowercase, k=word_length))
            words.append(word)
        text = ' '.join(words)
        texts.append(text)
    return texts


def generate_test_documents(count: int) -> List[Document]:
    """Generate test documents."""
    texts = generate_test_texts(count)
    return [
        Document(page_content=text, metadata={"source": f"doc_{i}"})
        for i, text in enumerate(texts)
    ]


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings instance."""
    return MockEmbeddings()


@pytest.fixture
def benchmark_embeddings():
    """Create embeddings instance for benchmarking."""
    return BenchmarkEmbeddings(dimension=384)


@pytest.fixture
def video_config():
    """Create a test video configuration."""
    return VideoConfig(
        fps=30,
        resolution=(640, 480),
        codec="mp4v"
    )


@pytest.fixture
def qrcode_config():
    """Create a test QR code configuration."""
    return QRCodeConfig(
        error_correction="H",
        box_size=10,
        border=4
    )


@pytest.fixture
def index_config():
    """Create a test index configuration."""
    return IndexConfig(
        index_type="faiss",
        metric="cosine",
        nlist=2
    )


@pytest.fixture
def vector_store_config(video_config, qrcode_config):
    """Create a test vector store configuration."""
    return VectorStoreConfig(
        video=video_config,
        qrcode=qrcode_config
    )


@pytest.fixture
def mock_index_manager():
    """Create a mock index manager."""
    manager = MagicMock()

    # Mock the search_text method to return SearchResult objects
    def search_text_side_effect(query_text, k=4):
        # Call embed_query to ensure it's called
        manager.embeddings.embed_query(query_text)

        # Return mock SearchResult objects
        return [
            SearchResult(
                text="test1",
                source="doc1",
                category="test_category",
                similarity=0.8,
                metadata={
                    "text": "test1",
                    "id": 0,
                    "source": "doc1",
                    "category": "test_category",
                    "metadata_hash": hashlib.sha256("test1".encode('utf-8')).hexdigest()
                }
            ),
            SearchResult(
                text="test2",
                source="doc2",
                category="test_category",
                similarity=0.6,
                metadata={
                    "text": "test2",
                    "id": 1,
                    "source": "doc2",
                    "category": "test_category",
                    "metadata_hash": hashlib.sha256("test2".encode('utf-8')).hexdigest()
                }
            )
        ]

    manager.search_text.side_effect = search_text_side_effect

    # Mock the get_metadata method to return metadata based on document IDs
    def get_metadata_side_effect(doc_ids):
        metadata_map = {
            0: {
                "text": "test1",
                "id": 0,
                "source": "doc1",
                "category": "test_category",
                "metadata_hash": hashlib.sha256("test1".encode('utf-8')).hexdigest()
            },
            1: {
                "text": "test2",
                "id": 1,
                "source": "doc2",
                "category": "test_category",
                "metadata_hash": hashlib.sha256("test2".encode('utf-8')).hexdigest()
            }
        }

        # Return metadata for each requested ID
        return [metadata_map.get(doc_id, {}) for doc_id in doc_ids]

    manager.get_metadata.side_effect = get_metadata_side_effect

    # Mock embeddings
    manager.embeddings = MagicMock()
    manager.embeddings.embed_query.return_value = [0.1, 0.2, 0.3]

    return manager


@pytest.fixture
def test_texts():
    """Create test texts for testing."""
    return [f"test text {i}" for i in range(10)]


@pytest.fixture
def test_metadata():
    """Create test metadata for testing."""
    return [{"id": i, "text": f"test_{i}"} for i in range(10)]


@pytest.fixture
def test_documents():
    """Create test documents for testing."""
    return generate_test_documents(10)
