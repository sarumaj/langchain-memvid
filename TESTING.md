# Testing

The project includes a comprehensive test suite to ensure reliability and measure performance characteristics.

## Running Tests

### Unit Tests

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

### Test Coverage

The test suite covers:

- **VectorStore**: Core vector store functionality, initialization, text addition, and search operations
- **Retriever**: Document retrieval with similarity search and scoring
- **Encoder**: Text encoding, QR code generation, and video building
- **Index**: FAISS index management and operations
- **Video Processing**: Video encoding/decoding and QR code extraction
- **IPython Extension**: Magic commands and utility functions
- **Configuration**: Settings validation and management
- **Error Handling**: Exception scenarios and edge cases

## Continuous Integration

The project includes GitHub Actions workflows that run:

- Unit tests on multiple Python versions
- Code coverage reporting
- Linting and code quality checks
- Release automation

## Performance Benchmarks

For detailed information about running and interpreting benchmarks, see [BENCHMARKING.md](BENCHMARKING.md). 