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

## Installation

```bash
pip install langchain-memvid
```

Optional dependencies:
```bash
# For testing
pip install langchain-memvid[test]

# For PDF support
pip install langchain-memvid[pdf]

# For EPUB support
pip install langchain-memvid[epub]

# For all features
pip install langchain-memvid[all]
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

## Example Notebooks

To help you get started, we provide two comprehensive Jupyter notebooks in the `examples` directory:

1. `quickstart.ipynb` - A basic tutorial demonstrating core functionality and common use cases
2. `advanced.ipynb` - An in-depth guide covering advanced features and customization options

These notebooks provide hands-on examples and detailed explanations of the library's capabilities.

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
- OpenCV
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

Full license texts for all dependencies are included in the `licenses` directory of this project. Each license file is named according to the project it belongs to (e.g., `licenses/faiss.txt`, `licenses/opencv.txt`, etc.).

For the original license texts and more information, please refer to the respective projects' repositories.

## Acknowledgments

- [Langchain](https://github.com/langchain-ai/langchain) for providing the framework and tools for building LLM applications
- [Memvid](https://github.com/Olow304/memvid) for the original video-based document storage concept
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [OpenCV](https://opencv.org/) for video processing capabilities
- [QRCode](https://github.com/lincolnloop/python-qrcode) for QR code generation
- [Pydantic](https://github.com/pydantic/pydantic) for data validation