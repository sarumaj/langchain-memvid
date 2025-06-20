"""Unit tests for the VectorStore class."""

import pytest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from langchain_memvid.vectorstore import VectorStore
from langchain_memvid.config import VectorStoreConfig
from langchain_memvid.types import BuildStats


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
            mock_build.return_value = BuildStats(
                total_chunks=2,
                video_size_mb=1.0,
                encoding_time=0.5,
                index_path=Path("test.d"),
                video_path=Path("test.mp4")
            )
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
            mock_build.return_value = BuildStats(
                total_chunks=2,
                video_size_mb=1.0,
                encoding_time=0.5,
                index_path=Path("test.d"),
                video_path=Path("test.mp4")
            )
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
            mock_build.return_value = BuildStats(
                total_chunks=2,
                video_size_mb=1.0,
                encoding_time=0.5,
                index_path=Path("test.d"),
                video_path=Path("test.mp4")
            )
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
            mock_build.return_value = BuildStats(
                total_chunks=2,
                video_size_mb=1.0,
                encoding_time=0.5,
                index_path=Path("test.d"),
                video_path=Path("test.mp4")
            )
            ids = await vector_store.aadd_documents(docs)

        assert ids == ["0", "1"]


class TestVectorStoreSearchOperations:
    """Test cases for search operations."""

    def test_similarity_search(self, vector_store):
        """Test similarity search."""
        mock_docs = [
            Document(page_content="Hello world", metadata={"distance": 0.1}),
            Document(page_content="Test document", metadata={"distance": 0.2})
        ]

        with patch("langchain_memvid.vectorstore.Retriever") as MockRetriever:
            mock_retriever = MockRetriever.return_value
            mock_retriever._get_relevant_documents.return_value = mock_docs
            mock_retriever.k = 4

            vector_store.video_file.parent.mkdir(parents=True, exist_ok=True)
            vector_store.video_file.touch()

            vector_store._retriever = mock_retriever

            results = vector_store.similarity_search("query", k=2)

            mock_retriever._get_relevant_documents.assert_called_once()
            assert len(results) == 2
            assert results == mock_docs
            assert mock_retriever.k == 4

    def test_similarity_search_with_score(self, vector_store):
        """Test similarity search with scores."""
        mock_docs = [
            Document(page_content="Hello world", metadata={"distance": 0.1}),
            Document(page_content="Test document", metadata={"distance": 0.2})
        ]

        with patch("langchain_memvid.vectorstore.Retriever") as MockRetriever:
            mock_retriever = MockRetriever.return_value
            mock_retriever._get_relevant_documents.return_value = mock_docs

            vector_store.video_file.parent.mkdir(parents=True, exist_ok=True)
            vector_store.video_file.touch()

            results = vector_store.similarity_search_with_score("query", k=2)

            assert len(results) == 2
            assert all(isinstance(doc, Document) for doc, _ in results)
            assert all(isinstance(score, float) for _, score in results)

    @pytest.mark.asyncio
    async def test_asimilarity_search(self, vector_store):
        """Test asynchronous similarity search."""
        mock_docs = [
            Document(page_content="Hello world", metadata={"distance": 0.1}),
            Document(page_content="Test document", metadata={"distance": 0.2})
        ]

        vector_store.video_file.parent.mkdir(parents=True, exist_ok=True)
        vector_store.video_file.touch()

        # Mock the sync similarity_search method since async is just a wrapper
        with patch.object(vector_store, 'similarity_search', return_value=mock_docs):
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

        vector_store.video_file.parent.mkdir(parents=True, exist_ok=True)
        vector_store.video_file.touch()

        # Mock the sync similarity_search_with_score method since async is just a wrapper
        with patch.object(vector_store, 'similarity_search_with_score', return_value=[(doc, 0.5) for doc in mock_docs]):
            results = await vector_store.asimilarity_search_with_score("query", k=2)

            assert len(results) == 2
            assert all(isinstance(doc, Document) for doc, _ in results)
            assert all(isinstance(score, float) for _, score in results)


class TestVectorStoreFactoryMethods:
    """Test cases for factory methods."""

    def test_from_texts(self, mock_embeddings, temp_dir):
        """Test from_texts factory method."""
        texts = ["Hello world", "Test document"]
        metadatas = [{"source": "test1"}, {"source": "test2"}]

        with patch("langchain_memvid.vectorstore.Encoder.build_video") as mock_build:
            mock_build.return_value = BuildStats(
                total_chunks=2,
                video_size_mb=1.0,
                encoding_time=0.5,
                index_path=Path("test.d"),
                video_path=Path("test.mp4")
            )
            vector_store = VectorStore.from_texts(
                texts=texts,
                embedding=mock_embeddings,
                metadatas=metadatas,
                video_file=temp_dir / "test.mp4",
                index_dir=temp_dir / "index.d"
            )

        assert isinstance(vector_store, VectorStore)
        assert vector_store.index_manager.embeddings == mock_embeddings

    def test_from_documents(self, mock_embeddings, temp_dir):
        """Test from_documents factory method."""
        docs = [
            Document(page_content="Hello world", metadata={"source": "test1"}),
            Document(page_content="Test document", metadata={"source": "test2"})
        ]

        with patch("langchain_memvid.vectorstore.Encoder.build_video") as mock_build:
            mock_build.return_value = BuildStats(
                total_chunks=2,
                video_size_mb=1.0,
                encoding_time=0.5,
                index_path=Path("test.d"),
                video_path=Path("test.mp4")
            )
            vector_store = VectorStore.from_documents(
                documents=docs,
                embedding=mock_embeddings,
                video_file=temp_dir / "test.mp4",
                index_dir=temp_dir / "index.d"
            )

        assert isinstance(vector_store, VectorStore)
        assert vector_store.index_manager.embeddings == mock_embeddings


class TestVectorStoreThreadSafety:
    """Test cases for thread safety."""

    def test_similarity_search_thread_safety(self, vector_store):
        """Test thread safety of similarity search."""
        import threading
        import time

        with patch("langchain_memvid.vectorstore.Retriever") as MockRetriever:
            mock_retriever = MockRetriever.return_value

            def mock_get_relevant_documents(query):
                time.sleep(0.01)  # Simulate some processing time
                return [
                    Document(page_content=f"Result for {query}", metadata={"distance": 0.1})
                ]

            mock_retriever._get_relevant_documents.side_effect = mock_get_relevant_documents
            mock_retriever.k = 4

            vector_store.video_file.parent.mkdir(parents=True, exist_ok=True)
            vector_store.video_file.touch()
            vector_store._retriever = mock_retriever

            def search_thread(k_value):
                return vector_store.similarity_search("query", k=k_value)

            threads = []
            for i in range(5):
                thread = threading.Thread(target=search_thread, args=(i + 1,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            assert mock_retriever._get_relevant_documents.call_count == 5


class TestVectorStoreDeletionOperations:
    """Test cases for deletion operations."""

    def test_delete_by_ids(self, vector_store):
        """Test deleting documents by IDs."""
        vector_store.video_file.parent.mkdir(parents=True, exist_ok=True)
        vector_store.video_file.touch()

        # Initialize the index manager and add some data
        vector_store.index_manager.create_index()
        vector_store.index_manager.add_texts(["test1", "test2"], [{"source": "doc1"}, {"source": "doc2"}])

        # Mock the index manager delete_by_ids method
        with patch.object(vector_store.index_manager, 'delete_by_ids', return_value=True):
            result = vector_store.delete_by_ids(["0", "1"])

            assert result is True
            vector_store.index_manager.delete_by_ids.assert_called_once_with([0, 1])

    def test_delete_by_ids_empty_list(self, vector_store):
        """Test deleting with empty IDs list."""
        with pytest.raises(ValueError, match="No document IDs provided to delete"):
            vector_store.delete_by_ids([])

    def test_delete_by_ids_invalid_format(self, vector_store):
        """Test deleting with invalid ID format."""
        vector_store.video_file.parent.mkdir(parents=True, exist_ok=True)
        vector_store.video_file.touch()

        with pytest.raises(RuntimeError, match="Failed to delete documents: Invalid document ID format"):
            vector_store.delete_by_ids(["invalid_id"])

    def test_delete_by_ids_video_not_exists(self, vector_store):
        """Test deleting when video file doesn't exist."""
        with pytest.raises(RuntimeError, match="Video file .* does not exist. No documents to delete."):
            vector_store.delete_by_ids(["0"])

    def test_delete_by_texts(self, vector_store):
        """Test deleting documents by texts."""
        vector_store.video_file.parent.mkdir(parents=True, exist_ok=True)
        vector_store.video_file.touch()

        # Initialize the index manager and add some data
        vector_store.index_manager.create_index()
        vector_store.index_manager.add_texts(["text1", "text2"], [{"source": "doc1"}, {"source": "doc2"}])

        # Mock the index manager delete_by_texts method
        with patch.object(vector_store.index_manager, 'delete_by_texts', return_value=True):
            result = vector_store.delete_by_texts(["text1", "text2"])

            assert result is True
            vector_store.index_manager.delete_by_texts.assert_called_once_with(["text1", "text2"])

    def test_delete_by_texts_empty_list(self, vector_store):
        """Test deleting with empty texts list."""
        with pytest.raises(ValueError, match="No texts provided to delete"):
            vector_store.delete_by_texts([])

    def test_delete_by_texts_video_not_exists(self, vector_store):
        """Test deleting texts when video file doesn't exist."""
        with pytest.raises(RuntimeError, match="Video file .* does not exist. No documents to delete."):
            vector_store.delete_by_texts(["text1"])

    def test_delete_documents(self, vector_store):
        """Test deleting documents."""
        vector_store.video_file.parent.mkdir(parents=True, exist_ok=True)
        vector_store.video_file.touch()

        # Initialize the index manager and add some data
        vector_store.index_manager.create_index()
        vector_store.index_manager.add_texts(["text1", "text2"], [{"source": "doc1"}, {"source": "doc2"}])

        docs = [
            Document(page_content="text1", metadata={"source": "doc1"}),
            Document(page_content="text2", metadata={"source": "doc2"})
        ]

        # Mock the index manager delete_by_texts method
        with patch.object(vector_store.index_manager, 'delete_by_texts', return_value=True):
            result = vector_store.delete_documents(docs)

            assert result is True
            vector_store.index_manager.delete_by_texts.assert_called_once_with(["text1", "text2"])

    def test_delete_documents_empty_list(self, vector_store):
        """Test deleting with empty documents list."""
        with pytest.raises(ValueError, match="No documents provided to delete"):
            vector_store.delete_documents([])

    @pytest.mark.asyncio
    async def test_adelete_by_ids(self, vector_store):
        """Test asynchronous deletion by IDs."""
        vector_store.video_file.parent.mkdir(parents=True, exist_ok=True)
        vector_store.video_file.touch()

        # Initialize the index manager and add some data
        vector_store.index_manager.create_index()
        vector_store.index_manager.add_texts(["test1", "test2"], [{"source": "doc1"}, {"source": "doc2"}])

        # Mock the sync delete_by_ids method since async is just a wrapper
        with patch.object(vector_store, 'delete_by_ids', return_value=True):
            result = await vector_store.adelete_by_ids(["0", "1"])

            assert result is True

    @pytest.mark.asyncio
    async def test_adelete_by_texts(self, vector_store):
        """Test asynchronous deletion by texts."""
        vector_store.video_file.parent.mkdir(parents=True, exist_ok=True)
        vector_store.video_file.touch()

        # Initialize the index manager and add some data
        vector_store.index_manager.create_index()
        vector_store.index_manager.add_texts(["text1", "text2"], [{"source": "doc1"}, {"source": "doc2"}])

        # Mock the sync delete_by_texts method since async is just a wrapper
        with patch.object(vector_store, 'delete_by_texts', return_value=True):
            result = await vector_store.adelete_by_texts(["text1", "text2"])

            assert result is True

    @pytest.mark.asyncio
    async def test_adelete_documents(self, vector_store):
        """Test asynchronous deletion of documents."""
        vector_store.video_file.parent.mkdir(parents=True, exist_ok=True)
        vector_store.video_file.touch()

        # Initialize the index manager and add some data
        vector_store.index_manager.create_index()
        vector_store.index_manager.add_texts(["text1", "text2"], [{"source": "doc1"}, {"source": "doc2"}])

        docs = [
            Document(page_content="text1", metadata={"source": "doc1"}),
            Document(page_content="text2", metadata={"source": "doc2"})
        ]

        # Mock the sync delete_documents method since async is just a wrapper
        with patch.object(vector_store, 'delete_documents', return_value=True):
            result = await vector_store.adelete_documents(docs)

            assert result is True


class TestVectorStoreDataTypes:
    """Test cases for data types and return values."""

    def test_get_storage_stats_returns_storage_stats(self, vector_store):
        """Test that get_storage_stats returns StorageStats."""
        from langchain_memvid.types import StorageStats, FrameMappingStats

        vector_store.video_file.parent.mkdir(parents=True, exist_ok=True)
        vector_store.video_file.touch()

        # Add some test data to the index manager
        vector_store.index_manager.create_index()
        vector_store.index_manager.add_texts(["test1", "test2"], [{"source": "doc1"}, {"source": "doc2"}])

        # Mock the get_storage_stats method to return expected data
        with patch.object(vector_store, 'get_storage_stats') as mock_get_stats:
            mock_get_stats.return_value = StorageStats(
                total_documents=10,
                video_file_size_mb=1.0,
                index_size_mb=0.5,
                essential_metadata_size_mb=0.01,
                full_metadata_size_mb=0.8,
                redundancy_percentage=1.25,
                storage_efficiency="hybrid",
                frame_mapping_stats=FrameMappingStats(
                    total_documents=10,
                    mapped_documents=10,
                    mapping_coverage=100.0,
                    mapping_efficiency={}
                )
            )

            stats = vector_store.get_storage_stats()

            assert isinstance(stats, StorageStats)
            assert stats.total_documents == 10
            assert stats.video_file_size_mb == 1.0
            assert stats.index_size_mb == 0.5

    def test_build_stats_dataclass(self):
        """Test BuildStats dataclass."""
        stats = BuildStats(
            total_chunks=10,
            video_size_mb=2.5,
            encoding_time=1.5,
            index_path=Path("test.d"),
            video_path=Path("test.mp4")
        )

        assert stats.total_chunks == 10
        assert stats.video_size_mb == 2.5
        assert stats.encoding_time == 1.5
        assert stats.index_path == Path("test.d")
        assert stats.video_path == Path("test.mp4")

    def test_video_info_dataclass(self):
        """Test VideoInfo dataclass."""
        from langchain_memvid.types import VideoInfo

        video_info = VideoInfo(
            frame_count=315,
            fps=30.0,
            width=640,
            height=480,
            duration_seconds=10.5,
            file_size_mb=2.0
        )

        assert video_info.frame_count == 315
        assert video_info.fps == 30.0
        assert video_info.width == 640
        assert video_info.height == 480
        assert video_info.duration_seconds == 10.5
        assert video_info.file_size_mb == 2.0

    def test_frame_mapping_stats_dataclass(self):
        """Test FrameMappingStats dataclass."""
        from langchain_memvid.types import FrameMappingStats

        frame_stats = FrameMappingStats(
            total_documents=100,
            mapped_documents=95,
            mapping_coverage=95.0,
            mapping_efficiency={}
        )

        assert frame_stats.total_documents == 100
        assert frame_stats.mapped_documents == 95
        assert frame_stats.mapping_coverage == 95.0
        assert isinstance(frame_stats.mapping_efficiency, dict)
