[![release](https://github.com/sarumaj/langchain-memvid/actions/workflows/release.yml/badge.svg)](https://github.com/sarumaj/langchain-memvid/actions/workflows/release.yml)
[![GitHub Release](https://img.shields.io/github/v/release/sarumaj/langchain-memvid?logo=github)](https://github.com/sarumaj/langchain-memvid/releases/latest)
[![Libraries.io dependency status for GitHub repo](https://img.shields.io/librariesio/github/sarumaj/langchain-memvid)](https://github.com/sarumaj/langchain-memvid/blob/main/pyproject.toml)

---

# Langchain Memvid

A Langchain implementation for Memvid, enabling efficient video-based document storage and retrieval using vector embeddings.

## Overview

Langchain Memvid is a powerful tool that combines the capabilities of Langchain with video-based document storage. It allows you to store text chunks in video frames using QR codes and retrieve them efficiently using vector similarity search. This approach provides a unique way to store and retrieve information while maintaining visual accessibility.

## Features

- **Vector Store Integration**: Seamless integration with Langchain's vector store interface
- **Video-based Storage**: Store text chunks in video frames using QR codes
- **Efficient Retrieval**: Fast similarity search using FAISS indexing
- **Flexible Configuration**: Customizable settings for encoding, indexing, and retrieval
- **Multiple Embedding Models**: Support for various embedding models through Langchain
- **Granular Control**: Access to low-level components for fine-tuned control
- **Comprehensive Testing**: Extensive test suite with performance benchmarks

## Installation

```bash
pip install langchain-memvid
```

For development and testing:

```bash
pip install -e ".[test]"
```

## Quick Start

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_memvid import VectorStore

# Initialize vector store
embeddings = HuggingFaceEmbeddings()
vs = VectorStore.from_texts(
    texts=["Important fact 1", "Important fact 2", "Historical event details"],
    embedding=embeddings,
    video_file="knowledge_base.mp4",
    index_dir="knowledge_base_index.d",
    metadatas=[
        {"id": 0, "source": "doc1.txt", "category": "facts"},
        {"id": 1, "source": "doc1.txt", "category": "facts"},
        {"id": 2, "source": "doc2.txt", "category": "history"}
    ]
)

# Search for similar content
results = vs.similarity_search("query", k=2)
```

## IPython Extension (Optional)

For enhanced Jupyter notebook and IPython experience, use the optional IPython extension:

```python
%pip install langchain-memvid
%load_ext ipykernel_memvid_extension
```

The extension provides magic commands for:
- Displaying data as bullet lists and tables (`%as_bullet_list`, `%as_table`)
- Automatic cleanup of temporary files (`%cleanup`)
- Package installation with visual feedback (`%pip_install`)
- Sound notifications for cell completion
- Enhanced progress bars

For detailed usage instructions, see [IPYTHON_EXTENSION.md](IPYTHON_EXTENSION.md).

## Example Notebooks

To help you get started, we provide two comprehensive Jupyter notebooks in the [examples](examples) directory:

1. [quickstart.ipynb](examples/quickstart.ipynb) - A basic tutorial demonstrating core functionality and common use cases
2. [advanced.ipynb](examples/advanced.ipynb) - An in-depth guide covering advanced features and customization options

These notebooks provide hands-on examples and detailed explanations of the library's capabilities.

## Testing and Benchmarking

The project includes a comprehensive test suite and performance benchmarks to ensure reliability and measure performance characteristics.

### Running Tests

#### Unit Tests

Run all unit tests:

```bash
pytest
```

Run specific test categories:

```bash
# Vector store tests
pytest tests/test_vectorstore.py

# Retriever tests
pytest tests/test_retriever.py

# Encoder tests
pytest tests/test_encoder.py

# Index tests
pytest tests/test_index.py

# Video processing tests
pytest tests/test_video.py

# IPython extension tests
pytest tests/test_ipykernel_memvid_extension.py
```

#### Test Coverage

The test suite covers:

- **VectorStore**: Core vector store functionality, initialization, text addition, and search operations
- **Retriever**: Document retrieval with similarity search and scoring
- **Encoder**: Text encoding, QR code generation, and video building
- **Index**: FAISS index management and operations
- **Video Processing**: Video encoding/decoding and QR code extraction
- **IPython Extension**: Magic commands and utility functions
- **Configuration**: Settings validation and management
- **Error Handling**: Exception scenarios and edge cases

### Performance Benchmarks

The project includes comprehensive benchmark tests to measure performance characteristics of the VectorStore implementation.

#### Running Benchmarks

**Note**: Benchmarks are disabled by default to avoid slowing down regular test runs. To enable benchmarks, override the `--benchmark-disable` and `--benchmark-skip` flags from `pyproject.toml` with `--benchmark-enable` and `--benchmark-only` command-line flags.

Run all benchmarks:

```bash
pytest tests/test_vectorstore_benchmark.py -v --benchmark-only --benchmark-enable
```

Run specific benchmark categories:

```bash
# Search performance only
pytest tests/test_vectorstore_benchmark.py -v --benchmark-only --benchmark-enable -k "search"

# Scaling tests only
pytest tests/test_vectorstore_benchmark.py -v --benchmark-only --benchmark-enable -k "scaling"

# Memory usage tests only
pytest tests/test_vectorstore_benchmark.py -v --benchmark-only --benchmark-enable -k "memory"

# Async operation tests only
pytest tests/test_vectorstore_benchmark.py -v --benchmark-only --benchmark-enable -k "async"
```

#### Benchmark Categories

The benchmark suite measures:

1. **Initialization Performance** (`TestVectorStoreInitializationBenchmark`)
   - VectorStore creation time with different configurations

2. **Text Addition Performance** (`TestVectorStoreTextAdditionBenchmark`)
   - Adding texts and documents at different scales (10, 100, 1000 documents)
   - Performance comparison between `add_texts()` and `add_documents()`

3. **Search Performance** (`TestVectorStoreSearchBenchmark`)
   - Similarity search performance with various parameters
   - Search with and without similarity scores
   - Performance with different k values (1, 5, 10, 20)

4. **Scaling Characteristics** (`TestVectorStoreScalingBenchmark`)
   - Performance scaling with document count (10, 50, 100, 200 documents)
   - Search performance scaling with index size (50, 100, 200, 500 documents)

5. **Memory Usage** (`TestVectorStoreMemoryBenchmark`)
   - Memory consumption with large documents (~1000 characters each)

6. **Async Operations** (`TestVectorStoreAsyncBenchmark`)
   - Performance of asynchronous text addition and search operations

7. **Factory Methods** (`TestVectorStoreFactoryMethodBenchmark`)
   - Performance of `from_texts()` and `from_documents()` factory methods

8. **Configuration Impact** (`TestVectorStoreConfigurationBenchmark`)
   - Performance with different embedding dimensions (128, 256, 384, 512)
   - Impact of custom QR code and index configurations

#### Interpreting Benchmark Results

Benchmark output shows:
- **Mean**: Average execution time
- **Std**: Standard deviation of execution times
- **Min/Max**: Minimum and maximum execution times
- **Rounds**: Number of benchmark rounds executed
- **Iterations**: Number of iterations per round

Example output:
```
test_add_texts_small_batch         0.1234 ± 0.0123 (mean ± std. dev. of 3 runs, 10 iterations each)
test_add_texts_medium_batch        1.2345 ± 0.1234 (mean ± std. dev. of 3 runs, 10 iterations each)
test_add_texts_large_batch        12.3456 ± 1.2345 (mean ± std. dev. of 3 runs, 10 iterations each)
```

#### Saving and Comparing Results

Save benchmark results:

```bash
pytest tests/test_vectorstore_benchmark.py --benchmark-only --benchmark-enable --benchmark-save=results.json --benchmark-save-data
```

Compare with previous results:

```bash
pytest tests/test_vectorstore_benchmark.py --benchmark-only --benchmark-enable --benchmark-compare=results.json
```

### Test Data Generation

The benchmarks use generated test data with realistic characteristics:

- **Text Generation**: Realistic text with varying lengths (50-200 characters)
- **Embeddings**: Deterministic embeddings based on text content hash
- **Metadata**: Simple metadata with source and batch information
- **Queries**: Generated queries for search testing

### Continuous Integration

The project includes GitHub Actions workflows that run:

- Unit tests on multiple Python versions
- Code coverage reporting
- Linting and code quality checks
- Release automation

## Advanced Usage

For more granular control, you can use the individual components:

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_memvid import (
    Encoder,
    IndexConfig,
    IndexManager,
    QRCodeConfig,
    VideoConfig,
    Retriever,
    VectorStoreConfig
)
from langchain_memvid.video import VideoProcessor
from pathlib import Path

# Initialize components
config = IndexConfig(
    index_type="faiss",
    metric="cosine",
    nlist=6  # Number of clusters for IVF index
)
embeddings = HuggingFaceEmbeddings()
index_manager = IndexManager(config=config, embeddings=embeddings)

# Add texts with metadata
texts = ["text chunk 1", "text chunk 2"]
metadata = [
    {"id": 0, "source": "doc1.txt", "category": "example"},
    {"id": 1, "source": "doc1.txt", "category": "example"}
]
index_manager.add_texts(texts, metadata)

# Configure video processing
video_config = VideoConfig(
    fps=30,
    resolution=(1920, 1080),
    codec="mp4v"
)
qrcode_config = QRCodeConfig(
    error_correction="H",
    box_size=10,
    border=4
)

# Create video processor and encoder
video_processor = VideoProcessor(
    video_config=video_config,
    qrcode_config=qrcode_config
)
encoder = Encoder(video_processor, index_manager)

# Build video with encoded data
video_file = Path("output.mp4")
index_dir = Path("index.d")
encoder.build_video(video_file, index_dir)

# Initialize retriever for searching
retriever = Retriever(
    video_file=video_file,
    index_dir=index_dir,
    config=VectorStoreConfig(
        video=video_config,
        qrcode=qrcode_config
    ),
    index_manager=index_manager,
    k=2
)

# Search for similar content
results = retriever.retrieve("query")
```

## Requirements

- Python >= 3.12
- OpenCV or/and ffmpeg
- FAISS
- Langchain
- Other dependencies as specified in pyproject.toml

## License

This project is licensed under the BSD-3-Clause License - see the LICENSE file for details.

## License Notices

This project uses several open-source libraries:

- FAISS is licensed under the Apache License 2.0
- OpenCV is licensed under the Apache License 2.0
- Langchain is licensed under the MIT License
- QRCode is licensed under the BSD License
- Pydantic is licensed under the MIT License
- Other dependencies are licensed under MIT, Apache 2.0, or AGPL-3.0

Full license texts for all dependencies are included in the [licenses](licenses) directory of this project. Each license file is named according to the project it belongs to (e.g., `licenses/faiss.txt`, `licenses/opencv.txt`, etc.).

For the original license texts and more information, please refer to the respective projects' repositories.

## Acknowledgments

- [Langchain](https://github.com/langchain-ai/langchain) for providing the framework and tools for building LLM applications
- [Memvid](https://github.com/Olow304/memvid) for the original video-based document storage concept
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [OpenCV](https://opencv.org/) for video processing capabilities
- [QRCode](https://github.com/lincolnloop/python-qrcode) for QR code generation
- [Pydantic](https://github.com/pydantic/pydantic) for data validation