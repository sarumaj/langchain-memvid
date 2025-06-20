"""Benchmark tests for VectorStore performance measurements."""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List
import random
import string

from langchain_core.documents import Document

from langchain_memvid.vectorstore import VectorStore
from langchain_memvid.config import VectorStoreConfig

# Import shared test utilities from conftest
from conftest import generate_test_texts, BenchmarkEmbeddings


def generate_test_queries(count: int, min_length: int = 10, max_length: int = 50) -> List[str]:
    """Generate test queries for search benchmarking."""
    return generate_test_texts(count, min_length, max_length)


@pytest.fixture
def benchmark_temp_dir():
    """Create a temporary directory for benchmark tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def small_vector_store(benchmark_temp_dir, benchmark_embeddings):
    """Create a VectorStore instance for small-scale benchmarks."""
    video_file = benchmark_temp_dir / "small_benchmark.mp4"
    index_dir = benchmark_temp_dir / "small_index"
    return VectorStore(
        embedding=benchmark_embeddings,
        video_file=video_file,
        index_dir=index_dir
    )


@pytest.fixture
def medium_vector_store(benchmark_temp_dir, benchmark_embeddings):
    """Create a VectorStore instance for medium-scale benchmarks."""
    video_file = benchmark_temp_dir / "medium_benchmark.mp4"
    index_dir = benchmark_temp_dir / "medium_index"
    return VectorStore(
        embedding=benchmark_embeddings,
        video_file=video_file,
        index_dir=index_dir
    )


@pytest.fixture
def large_vector_store(benchmark_temp_dir, benchmark_embeddings):
    """Create a VectorStore instance for large-scale benchmarks."""
    video_file = benchmark_temp_dir / "large_benchmark.mp4"
    index_dir = benchmark_temp_dir / "large_index"
    return VectorStore(
        embedding=benchmark_embeddings,
        video_file=video_file,
        index_dir=index_dir
    )


@pytest.mark.benchmark
class TestVectorStoreInitializationBenchmark:
    """Benchmark tests for VectorStore initialization."""

    def test_initialization_performance(self, benchmark, benchmark_temp_dir, benchmark_embeddings):
        """Benchmark VectorStore initialization time."""
        video_file = benchmark_temp_dir / "init_benchmark.mp4"
        index_dir = benchmark_temp_dir / "init_index"

        def init_vector_store():
            return VectorStore(
                embedding=benchmark_embeddings,
                video_file=video_file,
                index_dir=index_dir
            )

        result = benchmark(init_vector_store)
        assert result is not None


@pytest.mark.benchmark
class TestVectorStoreTextAdditionBenchmark:
    """Benchmark tests for text addition operations."""

    def test_add_texts_small_batch(self, benchmark, small_vector_store):
        """Benchmark adding a small batch of texts."""
        texts = generate_test_texts(10)
        metadatas = [{"source": f"test_{i}", "batch": "small"} for i in range(len(texts))]

        def add_texts():
            return small_vector_store.add_texts(texts, metadatas=metadatas)

        result = benchmark(add_texts)
        assert len(result) == 10

    def test_add_texts_medium_batch(self, benchmark, medium_vector_store):
        """Benchmark adding a medium batch of texts."""
        texts = generate_test_texts(100)
        metadatas = [{"source": f"test_{i}", "batch": "medium"} for i in range(len(texts))]

        def add_texts():
            return medium_vector_store.add_texts(texts, metadatas=metadatas)

        result = benchmark(add_texts)
        assert len(result) == 100

    def test_add_texts_large_batch(self, benchmark, large_vector_store):
        """Benchmark adding a large batch of texts."""
        texts = generate_test_texts(1000)
        metadatas = [{"source": f"test_{i}", "batch": "large"} for i in range(len(texts))]

        def add_texts():
            return large_vector_store.add_texts(texts, metadatas=metadatas)

        result = benchmark(add_texts)
        assert len(result) == 1000

    def test_add_documents_performance(self, benchmark, small_vector_store):
        """Benchmark adding documents vs texts."""
        texts = generate_test_texts(50)
        documents = [
            Document(page_content=text, metadata={"source": f"doc_{i}"})
            for i, text in enumerate(texts)
        ]

        def add_documents():
            return small_vector_store.add_documents(documents)

        result = benchmark(add_documents)
        assert len(result) == 50


@pytest.mark.benchmark
class TestVectorStoreSearchBenchmark:
    """Benchmark tests for search operations."""

    @pytest.fixture
    def populated_vector_store(self, small_vector_store):
        """Create a VectorStore with pre-populated data for search benchmarks."""
        texts = generate_test_texts(100)
        metadatas = [{"source": f"test_{i}"} for i in range(len(texts))]
        small_vector_store.add_texts(texts, metadatas=metadatas)
        return small_vector_store

    def test_similarity_search_performance(self, benchmark, populated_vector_store):
        """Benchmark similarity search performance."""
        queries = generate_test_queries(10)

        def similarity_search():
            results = []
            for query in queries:
                docs = populated_vector_store.similarity_search(query, k=5)
                results.append(len(docs))
            return results

        result = benchmark(similarity_search)
        assert len(result) == 10
        assert all(count == 5 for count in result)

    def test_similarity_search_with_score_performance(self, benchmark, populated_vector_store):
        """Benchmark similarity search with scores performance."""
        queries = generate_test_queries(10)

        def similarity_search_with_score():
            results = []
            for query in queries:
                docs_with_scores = populated_vector_store.similarity_search_with_score(query, k=5)
                results.append(len(docs_with_scores))
            return results

        result = benchmark(similarity_search_with_score)
        assert len(result) == 10
        assert all(count == 5 for count in result)

    def test_search_with_different_k_values(self, benchmark, populated_vector_store):
        """Benchmark search performance with different k values."""
        query = "test query for benchmarking"

        def search_k_1():
            return populated_vector_store.similarity_search(query, k=1)

        def search_k_5():
            return populated_vector_store.similarity_search(query, k=5)

        def search_k_10():
            return populated_vector_store.similarity_search(query, k=10)

        def search_k_20():
            return populated_vector_store.similarity_search(query, k=20)

        # Benchmark different k values
        result_k1 = benchmark(search_k_1)
        result_k5 = benchmark(search_k_5)
        result_k10 = benchmark(search_k_10)
        result_k20 = benchmark(search_k_20)

        assert len(result_k1) == 1
        assert len(result_k5) == 5
        assert len(result_k10) == 10
        assert len(result_k20) == 20


@pytest.mark.benchmark
class TestVectorStoreScalingBenchmark:
    """Benchmark tests for scaling characteristics."""

    def test_scaling_with_document_count(self, benchmark, benchmark_temp_dir, benchmark_embeddings):
        """Benchmark how performance scales with document count."""
        document_counts = [10, 50, 100, 200]
        results = {}

        for count in document_counts:
            video_file = benchmark_temp_dir / f"scale_{count}.mp4"
            index_dir = benchmark_temp_dir / f"scale_{count}_index"

            vector_store = VectorStore(
                embedding=benchmark_embeddings,
                video_file=video_file,
                index_dir=index_dir
            )

            texts = generate_test_texts(count)
            metadatas = [{"source": f"scale_{i}"} for i in range(count)]

            def add_texts():
                return vector_store.add_texts(texts, metadatas=metadatas)

            result = benchmark(add_texts)
            results[count] = result

        # Verify results
        for count, result in results.items():
            assert len(result) == count

    def test_search_scaling_with_index_size(self, benchmark, benchmark_temp_dir, benchmark_embeddings):
        """Benchmark search performance as index size increases."""
        index_sizes = [50, 100, 200, 500]
        search_results = {}

        for size in index_sizes:
            video_file = benchmark_temp_dir / f"search_scale_{size}.mp4"
            index_dir = benchmark_temp_dir / f"search_scale_{size}_index"

            vector_store = VectorStore(
                embedding=benchmark_embeddings,
                video_file=video_file,
                index_dir=index_dir
            )

            # Populate with documents
            texts = generate_test_texts(size)
            metadatas = [{"source": f"search_scale_{i}"} for i in range(size)]
            vector_store.add_texts(texts, metadatas=metadatas)

            # Generate test queries
            queries = generate_test_queries(5)

            def search_operation():
                results = []
                for query in queries:
                    docs = vector_store.similarity_search(query, k=10)
                    results.append(len(docs))
                return results

            result = benchmark(search_operation)
            search_results[size] = result

        # Verify results
        for size, result in search_results.items():
            assert len(result) == 5
            assert all(count == 10 for count in result)


@pytest.mark.benchmark
class TestVectorStoreMemoryBenchmark:
    """Benchmark tests for memory usage."""

    def test_memory_usage_with_large_documents(self, benchmark, benchmark_temp_dir, benchmark_embeddings):
        """Benchmark memory usage with large documents."""
        # Generate large documents
        large_texts = []
        for i in range(20):
            # Generate text with ~1000 characters
            words = []
            for _ in range(200):
                word_length = random.randint(3, 15)
                word = ''.join(random.choices(string.ascii_lowercase, k=word_length))
                words.append(word)
            large_texts.append(' '.join(words))

        video_file = benchmark_temp_dir / "memory_benchmark.mp4"
        index_dir = benchmark_temp_dir / "memory_index"

        vector_store = VectorStore(
            embedding=benchmark_embeddings,
            video_file=video_file,
            index_dir=index_dir
        )

        metadatas = [{"source": f"large_doc_{i}", "size": "large"} for i in range(len(large_texts))]

        def add_large_texts():
            return vector_store.add_texts(large_texts, metadatas=metadatas)

        result = benchmark(add_large_texts)
        assert len(result) == 20


@pytest.mark.benchmark
class TestVectorStoreAsyncBenchmark:
    """Benchmark tests for asynchronous operations."""

    @pytest.mark.asyncio
    async def test_async_add_texts_performance(self, benchmark, small_vector_store):
        """Benchmark asynchronous text addition."""
        texts = generate_test_texts(50)
        metadatas = [{"source": f"async_{i}"} for i in range(len(texts))]

        async def async_add_texts():
            return await small_vector_store.aadd_texts(texts, metadatas=metadatas)

        result = benchmark(async_add_texts)
        assert len(result) == 50

    @pytest.mark.asyncio
    async def test_async_search_performance(self, benchmark, small_vector_store):
        """Benchmark asynchronous search operations."""
        # First populate the vector store
        texts = generate_test_texts(100)
        metadatas = [{"source": f"async_search_{i}"} for i in range(len(texts))]
        await small_vector_store.aadd_texts(texts, metadatas=metadatas)

        queries = generate_test_queries(10)

        async def async_search():
            results = []
            for query in queries:
                docs = await small_vector_store.asimilarity_search(query, k=5)
                results.append(len(docs))
            return results

        result = benchmark(async_search)
        assert len(result) == 10
        assert all(count == 5 for count in result)


@pytest.mark.benchmark
class TestVectorStoreFactoryMethodBenchmark:
    """Benchmark tests for factory methods."""

    def test_from_texts_performance(self, benchmark, benchmark_temp_dir, benchmark_embeddings):
        """Benchmark from_texts factory method."""
        texts = generate_test_texts(100)
        metadatas = [{"source": f"factory_{i}"} for i in range(len(texts))]

        video_file = benchmark_temp_dir / "factory_benchmark.mp4"
        index_dir = benchmark_temp_dir / "factory_index"

        def from_texts():
            return VectorStore.from_texts(
                texts=texts,
                embedding=benchmark_embeddings,
                video_file=video_file,
                index_dir=index_dir,
                metadatas=metadatas
            )

        result = benchmark(from_texts)
        assert result is not None

    def test_from_documents_performance(self, benchmark, benchmark_temp_dir, benchmark_embeddings):
        """Benchmark from_documents factory method."""
        texts = generate_test_texts(100)
        documents = [
            Document(page_content=text, metadata={"source": f"factory_doc_{i}"})
            for i, text in enumerate(texts)
        ]

        video_file = benchmark_temp_dir / "factory_docs_benchmark.mp4"
        index_dir = benchmark_temp_dir / "factory_docs_index"

        def from_documents():
            return VectorStore.from_documents(
                documents=documents,
                embedding=benchmark_embeddings,
                video_file=video_file,
                index_dir=index_dir
            )

        result = benchmark(from_documents)
        assert result is not None


@pytest.mark.benchmark
class TestVectorStoreConfigurationBenchmark:
    """Benchmark tests for different configuration options."""

    def test_different_embedding_dimensions(self, benchmark, benchmark_temp_dir):
        """Benchmark performance with different embedding dimensions."""
        dimensions = [128, 256, 384, 512]
        results = {}

        for dim in dimensions:
            embeddings = BenchmarkEmbeddings(dimension=dim)
            video_file = benchmark_temp_dir / f"dim_{dim}.mp4"
            index_dir = benchmark_temp_dir / f"dim_{dim}_index"

            vector_store = VectorStore(
                embedding=embeddings,
                video_file=video_file,
                index_dir=index_dir
            )

            texts = generate_test_texts(50)
            metadatas = [{"source": f"dim_{dim}_{i}"} for i in range(len(texts))]

            def add_texts():
                return vector_store.add_texts(texts, metadatas=metadatas)

            result = benchmark(add_texts)
            results[dim] = result

        # Verify results
        for dim, result in results.items():
            assert len(result) == 50

    def test_custom_config_performance(self, benchmark, benchmark_temp_dir, benchmark_embeddings):
        """Benchmark performance with custom configuration."""
        # Create custom config with different settings
        custom_config = VectorStoreConfig(
            encoder=VectorStoreConfig.EncoderConfig(
                qr_version=10,
                qr_error_correction="M",
                qr_box_size=10,
                qr_border=4
            ),
            index=VectorStoreConfig.IndexConfig(
                index_type="Flat",
                metric="L2"
            )
        )

        video_file = benchmark_temp_dir / "custom_config.mp4"
        index_dir = benchmark_temp_dir / "custom_config_index"

        vector_store = VectorStore(
            embedding=benchmark_embeddings,
            video_file=video_file,
            index_dir=index_dir,
            config=custom_config
        )

        texts = generate_test_texts(100)
        metadatas = [{"source": f"custom_{i}"} for i in range(len(texts))]

        def add_texts_with_custom_config():
            return vector_store.add_texts(texts, metadatas=metadatas)

        result = benchmark(add_texts_with_custom_config)
        assert len(result) == 100
