# Contributing

Contributions are expected to be indispensable, well-tested, and cleanly written for reviewability and maintainability.
Poorly written contributions may be rejected without review.

## Setup

```bash
uv sync
```

## Linting

```bash
uv run bash scripts/lint.sh
```

## Testing

```bash
uv run pytest
```

### Doc snippet tests

`tests/test_docs.py` uses [pytest-examples](https://github.com/pydantic/pytest-examples) to discover and run every fenced Python code block in `docs/`, `README.md`, and `data_publishing/DATASET_CARD.md`.

All blocks within a single file share a chained namespace, so a variable defined in an earlier block is visible to later ones.
This matches the sequential reading order of the guide.
Data is loaded from HF Hub by the `from_hf()` calls in the blocks themselves (cached locally after first download).

#### Code fence annotations

Annotate the opening fence to control test behavior. Annotations are invisible in rendered docs.

````markdown
```python {test="skip"}
# Always skipped (e.g. placeholder paths, heavy downloads).
```

```python {test="skip-bulk"}
# Skipped by default.  Pass --run-bulk to include.
```
````

To run bulk-download blocks:

```bash
uv run pytest tests/test_docs.py --run-bulk
```

Note that the above (with `--run-bulk`) will run doc snippets that basically download the *whole* ~100 GB dataset into your `HF_HOME`.
