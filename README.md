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
from langchain_memvid import VectorStore, VectorStoreConfig

# Initialize vector store
vs_cfg = VectorStoreConfig()
embeddings = HuggingFaceEmbeddings()
vs = VectorStore(
    video_file="output.mp4",
    index_file="index.json",
    embedding=embeddings,
    config=vs_cfg
)

# Add text chunks
chunks = ["Important fact 1", "Important fact 2", "Historical event details"]
vs.add_texts(chunks)

# Search for similar content
results = vs.similarity_search("machine learning algorithms", top_k=3)
```

## Advanced Usage

For more granular control, you can use the individual components:

```python
from langchain_memvid.index import IndexManager, IndexConfig
from langchain_memvid.retriever import Retriever, RetrieverConfig
from langchain_memvid.encoder import Encoder, EncoderConfig

# Initialize components
config = IndexConfig(index_type="Flat")
embeddings = HuggingFaceEmbeddings()
index_manager = IndexManager(config=config, embeddings=embeddings)

# Add and search chunks
chunks = ["text chunk 1", "text chunk 2"]
frame_numbers = [1, 2]
chunk_ids = index_manager.add_chunks(chunks, frame_numbers)
results = index_manager.search("query", top_k=5)

# Encode to video
enc_cfg = EncoderConfig()
encoder = Encoder(enc_cfg, index_manager)
encoder.add_chunks(chunks)
encoder.build_video(output_file="output.mp4", index_file="index.json")

# Retrieve from video
ret_cfg = RetrieverConfig()
retriever = Retriever("output.mp4", "index.json", ret_cfg, index_manager)
results = retriever.search_with_metadata("query", top_k=3)
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
- [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) for HTML parsing
- [EbookLib](https://github.com/aerkalov/ebooklib) for EPUB handling