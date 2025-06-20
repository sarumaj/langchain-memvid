# Benchmarking

The project includes comprehensive benchmark tests to measure performance characteristics of the VectorStore implementation.

## Running Benchmarks

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

## Benchmark Categories

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

## Interpreting Benchmark Results

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

## Saving and Comparing Results

Save benchmark results:

```bash
pytest tests/test_vectorstore_benchmark.py --benchmark-only --benchmark-enable --benchmark-save=results.json --benchmark-save-data
```

Compare with previous results:

```bash
pytest tests/test_vectorstore_benchmark.py --benchmark-only --benchmark-enable --benchmark-compare=results.json
```

## Test Data Generation

The benchmarks use generated test data with realistic characteristics:

- **Text Generation**: Realistic text with varying lengths (50-200 characters)
- **Embeddings**: Deterministic embeddings based on text content hash
- **Metadata**: Simple metadata with source and batch information
- **Queries**: Generated queries for search testing 