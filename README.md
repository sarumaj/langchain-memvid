[![release](https://github.com/sarumaj/langchain-memvid/actions/workflows/release.yml/badge.svg)](https://github.com/sarumaj/langchain-memvid/actions/workflows/release.yml)
[![GitHub Release](https://img.shields.io/github/v/release/sarumaj/langchain-memvid?logo=github)](https://github.com/sarumaj/langchain-memvid/releases/latest)
[![Libraries.io dependency status for GitHub repo](https://img.shields.io/librariesio/github/sarumaj/langchain-memvid)](https://github.com/sarumaj/langchain-memvid/blob/main/pyproject.toml)

---

# Langchain Memvid

A Langchain implementation for Memvid, enabling efficient video-based document storage and retrieval using vector embeddings.

> Memvid revolutionizes AI memory management by encoding text data into videos, enabling lightning-fast semantic search across millions of text chunks with sub-second retrieval times. Unlike traditional vector databases that consume massive amounts of RAM and storage, Memvid compresses your knowledge base into compact video files while maintaining instant access to any piece of information.
>
> <cite>[Saleban Olow](https://github.com/Olow304)</cite>, author of [memvid](https://github.com/Olow304/memvid), 5th of June, 2025

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

For a complete quick start guide, see our **[quickstart.py](examples/quickstart.py)** example.

For detailed explanations and interactive examples, check out **[quickstart.ipynb](examples/quickstart.ipynb)**.

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

## Examples

We provide comprehensive examples in multiple formats to help you get started:

### 📓 Jupyter Notebooks (Interactive)

The most detailed examples with explanations and visual outputs:

- **[quickstart.ipynb](examples/quickstart.ipynb)** - Basic tutorial demonstrating core functionality
- **[advanced.ipynb](examples/advanced.ipynb)** - Advanced features and customization options

### 📄 Python Scripts (Executable)

Auto-generated Python files from the notebooks:

- **[quickstart.py](examples/quickstart.py)** - Basic usage example
- **[advanced.py](examples/advanced.py)** - Advanced usage example

### Running the Examples

1. **Interactive (Recommended)**: Open the `.ipynb` files in Jupyter
2. **Script**: Run the `.py` files directly with Python

The Python files are automatically generated when notebooks are executed, ensuring they stay in sync.

## Testing

For comprehensive testing information, including unit tests, test coverage, and continuous integration details, see [TESTING.md](TESTING.md).

## Benchmarking

For detailed benchmarking information, including performance tests, benchmark categories, and result interpretation, see [BENCHMARKING.md](BENCHMARKING.md).

## Advanced Usage

For comprehensive advanced usage examples, see our **[advanced.py](examples/advanced.py)** example.

For detailed explanations and interactive examples, check out **[advanced.ipynb](examples/advanced.ipynb)**.

## Requirements

- Python >= 3.12
- OpenCV or/and ffmpeg
- FAISS
- Langchain
- Other dependencies as specified in pyproject.toml

## License

This project is licensed under the BSD-3-Clause License - see the LICENSE file for details.

For information about third-party licenses, see [LICENSES.md](LICENSES.md).

## Acknowledgments

- [Langchain](https://github.com/langchain-ai/langchain) for providing the framework and tools for building LLM applications
- [Memvid](https://github.com/Olow304/memvid) for the original video-based document storage concept
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [OpenCV](https://opencv.org/) for video processing capabilities
- [QRCode](https://github.com/lincolnloop/python-qrcode) for QR code generation
- [Pydantic](https://github.com/pydantic/pydantic) for data validation
