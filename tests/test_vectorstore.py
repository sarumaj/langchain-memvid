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


class TestVectorStoreInitialization:
    """Test cases for VectorStore initialization."""

    def test_init(self, vector_store, temp_dir):
        """Test VectorStore initialization."""
        assert vector_store.video_file == (temp_dir / "test.mp4").absolute()
        assert vector_store.index_dir == (temp_dir / "index.d").absolute()
        assert vector_store.config == VectorStoreConfig()
        assert vector_store._retriever is None


class TestVectorStoreTextOperations:
    """Test cases for text addition operations."""

    def test_add_texts(self, vector_store):
        """Test adding texts to VectorStore."""
        texts = ["Hello world", "Test document"]
        metadatas = [{"source": "test1"}, {"source": "test2"}]

        with patch("langchain_memvid.vectorstore.Encoder.build_video") as mock_build:
            mock_build.return_value = Mock(total_chunks=2, video_size_mb=1.0)
            ids = vector_store.add_texts(texts, metadatas=metadatas)

        assert ids == ["0", "1"]
        assert vector_store._retriever is None

    def test_add_texts_empty(self, vector_store):
        """Test adding empty texts list raises ValueError."""
        with pytest.raises(ValueError, match="No texts provided to add"):
            vector_store.add_texts([])

    def test_add_documents(self, vector_store):
        """Test adding documents to VectorStore."""
        docs = [
            Document(page_content="Hello world", metadata={"source": "test1"}),
            Document(page_content="Test document", metadata={"source": "test2"})
        ]

        with patch("langchain_memvid.vectorstore.Encoder.build_video") as mock_build:
            mock_build.return_value = Mock(total_chunks=2, video_size_mb=1.0)
            ids = vector_store.add_documents(docs)

        assert ids == ["0", "1"]


class TestVectorStoreAsyncOperations:
    """Test cases for asynchronous operations."""

    @pytest.mark.asyncio
    async def test_aadd_texts(self, vector_store):
        """Test adding texts asynchronously."""
        texts = ["Hello world", "Test document"]
        metadatas = [{"source": "test1"}, {"source": "test2"}]

        with patch("langchain_memvid.vectorstore.Encoder.build_video") as mock_build:
            mock_build.return_value = Mock(total_chunks=2, video_size_mb=1.0)
            ids = await vector_store.aadd_texts(texts, metadatas=metadatas)

        assert ids == ["0", "1"]

    @pytest.mark.asyncio
    async def test_aadd_documents(self, vector_store):
        """Test adding documents asynchronously."""
        docs = [
            Document(page_content="Hello world", metadata={"source": "test1"}),
            Document(page_content="Test document", metadata={"source": "test2"})
        ]

        with patch("langchain_memvid.vectorstore.Encoder.build_video") as mock_build:
            mock_build.return_value = Mock(total_chunks=2, video_size_mb=1.0)
            ids = await vector_store.aadd_documents(docs)

        assert ids == ["0", "1"]


class TestVectorStoreSearchOperations:
    """Test cases for search operations."""

    def test_similarity_search(self, vector_store):
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

    def test_similarity_search_with_score(self, vector_store):
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
    async def test_asimilarity_search(self, vector_store):
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
    async def test_asimilarity_search_with_score(self, vector_store):
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


class TestVectorStoreFactoryMethods:
    """Test cases for factory methods."""

    def test_from_texts(self, mock_embeddings, temp_dir):
        """Test from_texts factory method."""
        texts = ["Hello world", "Test document"]

        with patch("langchain_memvid.vectorstore.Encoder.build_video") as mock_build:
            mock_build.return_value = Mock(total_chunks=2, video_size_mb=1.0)
            vector_store = VectorStore.from_texts(
                texts, mock_embeddings, temp_dir / "test.mp4", temp_dir / "index.d"
            )

        assert isinstance(vector_store, VectorStore)
        assert vector_store.video_file == (temp_dir / "test.mp4").absolute()

    def test_from_documents(self, mock_embeddings, temp_dir):
        """Test from_documents factory method."""
        docs = [
            Document(page_content="Hello world", metadata={"source": "test1"}),
            Document(page_content="Test document", metadata={"source": "test2"})
        ]

        with patch("langchain_memvid.vectorstore.Encoder.build_video") as mock_build:
            mock_build.return_value = Mock(total_chunks=2, video_size_mb=1.0)
            vector_store = VectorStore.from_documents(
                docs, mock_embeddings, temp_dir / "test.mp4", temp_dir / "index.d"
            )

        assert isinstance(vector_store, VectorStore)
        assert vector_store.video_file == (temp_dir / "test.mp4").absolute()


class TestVectorStoreThreadSafety:
    """Test cases for thread safety."""

    def test_similarity_search_thread_safety(self, vector_store):
        """Test that similarity search is thread-safe."""
        import threading
        import time

        # Mock retriever
        mock_docs = [Document(page_content="test", metadata={"distance": 0.1})]

        def mock_get_relevant_documents(query):
            time.sleep(0.01)  # Simulate some processing time
            return mock_docs

        with patch("langchain_memvid.vectorstore.Retriever") as MockRetriever:
            mock_retriever = MockRetriever.return_value
            mock_retriever._get_relevant_documents.side_effect = mock_get_relevant_documents

            # Create video file to avoid RuntimeError
            vector_store.video_file.parent.mkdir(parents=True, exist_ok=True)
            vector_store.video_file.touch()
            vector_store._retriever = mock_retriever

            def search_thread(k_value):
                return vector_store.similarity_search("query", k=k_value)

            # Run multiple threads simultaneously
            threads = []
            for i in range(5):
                thread = threading.Thread(target=search_thread, args=(i + 1,))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Verify all searches completed successfully
            assert mock_retriever._get_relevant_documents.call_count == 5
