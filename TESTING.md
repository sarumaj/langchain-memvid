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

Forgejo Actions runs `.forgejo/workflows/test.yml` on every pull request and
every push to `main` and `renovate/all-patch`: the tests on Python 3.15, with
ffmpeg. Their results go to a pull request as a comment (`dawid/ci-actions`'
test-report), with each failure's message. `.forgejo/workflows/release.yml`
publishes the releases; the documentation site is built on GitHub
(`.github/workflows/gh-pages.yml`).

## Performance Benchmarks

For detailed information about running and interpreting benchmarks, see [BENCHMARKING.md](BENCHMARKING.md). 