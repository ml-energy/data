# mlenergy-data

Python toolkit for [ML.ENERGY](https://ml.energy) Benchmark datasets: loading raw results, filtering and analyzing runs, fitting models, and building data packages.

## Installation

```bash
pip install -e .
```

## Quick example

```python
from mlenergy_data.records import LLMRuns

runs = LLMRuns.from_directory("/path/to/compiled/data")

# Find the most energy-efficient model on GPQA
best = min(runs.task("gpqa"), key=lambda r: r.energy_per_token_joules)
print(f"{best.nickname}: {best.energy_per_token_joules:.3f} J/tok")

# Column access via .data
energies = runs.data.energy_per_token_joules  # list[float]
```

## Documentation

See the full [documentation site](https://ml-energy.github.io/mlenergy-data/) for:

- [Usage guide](https://ml-energy.github.io/mlenergy-data/guide/) — progressive walkthrough from loading data to fitting models.
- [API reference](https://ml-energy.github.io/mlenergy-data/api/records/) — auto-generated from docstrings.
