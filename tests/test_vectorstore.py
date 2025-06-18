"""Unit tests for the VectorStore class."""

import pytest
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_memvid.vectorstore import VectorStore
from langchain_memvid.config import VectorStoreConfig


class MockEmbeddings(Embeddings):
    """Mock embeddings for testing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Mock embed_documents method."""
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Mock embed_query method."""
        return [0.1, 0.2, 0.3]


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
def vector_store(temp_dir, mock_embeddings):
    """Create a VectorStore instance for testing."""
    video_file = temp_dir / "test.mp4"
    index_dir = temp_dir / "index.d"
    return VectorStore(
        embedding=mock_embeddings,
        video_file=video_file,
        index_dir=index_dir
    )


def test_init(vector_store, temp_dir):
    """Test VectorStore initialization."""
    assert vector_store.video_file == (temp_dir / "test.mp4").absolute()
    assert vector_store.index_dir == (temp_dir / "index.d").absolute()
    assert vector_store.config == VectorStoreConfig()
    assert vector_store._retriever is None


def test_add_texts(vector_store):
    """Test adding texts to VectorStore."""
    texts = ["Hello world", "Test document"]
    metadatas = [{"source": "test1"}, {"source": "test2"}]

    with patch("langchain_memvid.vectorstore.Encoder.build_video") as mock_build:
        mock_build.return_value = Mock(total_chunks=2, video_size_mb=1.0)
        ids = vector_store.add_texts(texts, metadatas=metadatas)

    assert ids == ["0", "1"]
    assert vector_store._retriever is None


def test_add_texts_empty(vector_store):
    """Test adding empty texts list raises ValueError."""
    with pytest.raises(ValueError, match="No texts provided to add"):
        vector_store.add_texts([])


def test_add_documents(vector_store):
    """Test adding documents to VectorStore."""
    docs = [
        Document(page_content="Hello world", metadata={"source": "test1"}),
        Document(page_content="Test document", metadata={"source": "test2"})
    ]

    with patch("langchain_memvid.vectorstore.Encoder.build_video") as mock_build:
        mock_build.return_value = Mock(total_chunks=2, video_size_mb=1.0)
        ids = vector_store.add_documents(docs)

    assert ids == ["0", "1"]


@pytest.mark.asyncio
async def test_aadd_texts(vector_store):
    """Test adding texts asynchronously."""
    texts = ["Hello world", "Test document"]
    metadatas = [{"source": "test1"}, {"source": "test2"}]

    with patch("langchain_memvid.vectorstore.Encoder.build_video") as mock_build:
        mock_build.return_value = Mock(total_chunks=2, video_size_mb=1.0)
        ids = await vector_store.aadd_texts(texts, metadatas=metadatas)

    assert ids == ["0", "1"]


@pytest.mark.asyncio
async def test_aadd_documents(vector_store):
    """Test adding documents asynchronously."""
    docs = [
        Document(page_content="Hello world", metadata={"source": "test1"}),
        Document(page_content="Test document", metadata={"source": "test2"})
    ]

    with patch("langchain_memvid.vectorstore.Encoder.build_video") as mock_build:
        mock_build.return_value = Mock(total_chunks=2, video_size_mb=1.0)
        ids = await vector_store.aadd_documents(docs)

    assert ids == ["0", "1"]


def test_similarity_search(vector_store):
    """Test similarity search."""
    # Mock retriever's _get_relevant_documents method
    mock_docs = [
        Document(page_content="Hello world", metadata={"distance": 0.1}),
        Document(page_content="Test document", metadata={"distance": 0.2})
    ]

    with patch("langchain_memvid.vectorstore.Retriever") as MockRetriever:
        mock_retriever = MockRetriever.return_value
        mock_retriever._get_relevant_documents.return_value = mock_docs
        mock_retriever.k = 4  # Set initial k value

        # Create video file to avoid RuntimeError
        vector_store.video_file.parent.mkdir(parents=True, exist_ok=True)
        vector_store.video_file.touch()

        # Force retriever creation before search
        vector_store._retriever = mock_retriever

        results = vector_store.similarity_search("query", k=2)

        # Verify k was temporarily changed during search
        mock_retriever._get_relevant_documents.assert_called_once()
        assert len(results) == 2
        assert results == mock_docs
        assert mock_retriever.k == 4  # Check if k was restored to default


def test_similarity_search_with_score(vector_store):
    """Test similarity search with scores."""
    mock_docs = [
        Document(page_content="Hello world", metadata={"distance": 0.1}),
        Document(page_content="Test document", metadata={"distance": 0.2})
    ]

    with patch("langchain_memvid.vectorstore.Retriever") as MockRetriever:
        mock_retriever = MockRetriever.return_value
        mock_retriever._get_relevant_documents.return_value = mock_docs

        # Create video file to avoid RuntimeError
        vector_store.video_file.parent.mkdir(parents=True, exist_ok=True)
        vector_store.video_file.touch()

        results = vector_store.similarity_search_with_score("query", k=2)

    assert len(results) == 2
    assert results == [(mock_docs[0], 0.1), (mock_docs[1], 0.2)]


@pytest.mark.asyncio
async def test_asimilarity_search(vector_store):
    """Test asynchronous similarity search."""
    mock_docs = [
        Document(page_content="Hello world", metadata={"distance": 0.1}),
        Document(page_content="Test document", metadata={"distance": 0.2})
    ]

    with patch("langchain_memvid.vectorstore.Retriever") as MockRetriever:
        mock_retriever = MockRetriever.return_value
        mock_retriever._get_relevant_documents.return_value = mock_docs

        # Create video file to avoid RuntimeError
        vector_store.video_file.parent.mkdir(parents=True, exist_ok=True)
        vector_store.video_file.touch()

        results = await vector_store.asimilarity_search("query", k=2)

    assert len(results) == 2
    assert results == mock_docs


@pytest.mark.asyncio
async def test_asimilarity_search_with_score(vector_store):
    """Test asynchronous similarity search with scores."""
    mock_docs = [
        Document(page_content="Hello world", metadata={"distance": 0.1}),
        Document(page_content="Test document", metadata={"distance": 0.2})
    ]

    with patch("langchain_memvid.vectorstore.Retriever") as MockRetriever:
        mock_retriever = MockRetriever.return_value
        mock_retriever._get_relevant_documents.return_value = mock_docs

        # Create video file to avoid RuntimeError
        vector_store.video_file.parent.mkdir(parents=True, exist_ok=True)
        vector_store.video_file.touch()

        results = await vector_store.asimilarity_search_with_score("query", k=2)

    assert len(results) == 2
    assert results == [(mock_docs[0], 0.1), (mock_docs[1], 0.2)]


def test_from_texts(mock_embeddings, temp_dir):
    """Test creating VectorStore from texts."""
    texts = ["Hello world", "Test document"]
    metadatas = [{"source": "test1"}, {"source": "test2"}]

    with patch("langchain_memvid.vectorstore.Encoder.build_video") as mock_build:
        mock_build.return_value = Mock(total_chunks=2, video_size_mb=1.0)
        vs = VectorStore.from_texts(
            texts=texts,
            embedding=mock_embeddings,
            video_file=temp_dir / "test.mp4",
            index_dir=temp_dir / "index.d",
            metadatas=metadatas
        )

    assert isinstance(vs, VectorStore)
    assert vs.video_file == (temp_dir / "test.mp4").absolute()
    assert vs.index_dir == (temp_dir / "index.d").absolute()


def test_from_documents(mock_embeddings, temp_dir):
    """Test creating VectorStore from documents."""
    docs = [
        Document(page_content="Hello world", metadata={"source": "test1"}),
        Document(page_content="Test document", metadata={"source": "test2"})
    ]

    with patch("langchain_memvid.vectorstore.Encoder.build_video") as mock_build:
        mock_build.return_value = Mock(total_chunks=2, video_size_mb=1.0)
        vs = VectorStore.from_documents(
            documents=docs,
            embedding=mock_embeddings,
            video_file=temp_dir / "test.mp4",
            index_dir=temp_dir / "index.d"
        )

    assert isinstance(vs, VectorStore)
    assert vs.video_file == (temp_dir / "test.mp4").absolute()
    assert vs.index_dir == (temp_dir / "index.d").absolute()


@pytest.mark.asyncio
async def test_afrom_texts(mock_embeddings, temp_dir):
    """Test creating VectorStore from texts asynchronously."""
    texts = ["Hello world", "Test document"]
    metadatas = [{"source": "test1"}, {"source": "test2"}]

    with patch("langchain_memvid.vectorstore.Encoder.build_video") as mock_build:
        mock_build.return_value = Mock(total_chunks=2, video_size_mb=1.0)
        vs = await VectorStore.afrom_texts(
            texts=texts,
            embedding=mock_embeddings,
            video_file=temp_dir / "test.mp4",
            index_dir=temp_dir / "index.d",
            metadatas=metadatas
        )

    assert isinstance(vs, VectorStore)
    assert vs.video_file == (temp_dir / "test.mp4").absolute()
    assert vs.index_dir == (temp_dir / "index.d").absolute()


@pytest.mark.asyncio
async def test_afrom_documents(mock_embeddings, temp_dir):
    """Test creating VectorStore from documents asynchronously."""
    docs = [
        Document(page_content="Hello world", metadata={"source": "test1"}),
        Document(page_content="Test document", metadata={"source": "test2"})
    ]

    with patch("langchain_memvid.vectorstore.Encoder.build_video") as mock_build:
        mock_build.return_value = Mock(total_chunks=2, video_size_mb=1.0)
        vs = await VectorStore.afrom_documents(
            documents=docs,
            embedding=mock_embeddings,
            video_file=temp_dir / "test.mp4",
            index_dir=temp_dir / "index.d"
        )

    assert isinstance(vs, VectorStore)
    assert vs.video_file == (temp_dir / "test.mp4").absolute()
    assert vs.index_dir == (temp_dir / "index.d").absolute()


def test_similarity_search_thread_safety(vector_store):
    """Test that concurrent similarity searches handle k values correctly."""
    import threading
    import queue
    from unittest.mock import MagicMock

    results_queue = queue.Queue()
    mock_docs = [
        Document(page_content=f"Doc {i}", metadata={"distance": 0.1 * i})
        for i in range(5)
    ]

    # Create a single mock retriever to be shared across threads
    mock_retriever = MagicMock()
    mock_retriever.k = 4  # Set initial k value

    def mock_get_relevant_documents(query):
        """Mock implementation that returns k documents based on current k value."""
        return mock_docs[:mock_retriever.k]

    mock_retriever._get_relevant_documents = mock_get_relevant_documents

    def search_thread(k_value):
        results = vector_store.similarity_search("query", k=k_value)
        results_queue.put((k_value, len(results)))

    # Set up the mock retriever
    with patch("langchain_memvid.vectorstore.Retriever") as MockRetriever:
        MockRetriever.return_value = mock_retriever

        # Create video file to avoid RuntimeError
        vector_store.video_file.parent.mkdir(parents=True, exist_ok=True)
        vector_store.video_file.touch()

        # Force retriever creation before threading
        vector_store._retriever = mock_retriever

        # Create threads that search with different k values
        threads = [
            threading.Thread(target=search_thread, args=(k,))
            for k in [2, 3, 4]
        ]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

    # Check results
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    # Verify each search returned the correct number of results
    for k_value, result_count in results:
        assert result_count == k_value, f"Search with k={k_value} returned {result_count} results"

    # Verify k was restored to original value
    assert mock_retriever.k == 4
