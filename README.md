UrbanAccessAnalyzer

## Development checks

Run these before opening a PR:

```bash
pip install -e ".[dev]"
black --check .
ruff check .
pytest -q
```

If the repository has no tests yet, `pytest` may not run any tests. Add tests with new features when possible.

## Tsunami batch runner (single region)

Run the notebook pipeline non-interactively for one region:

```bash
python examples/tsunami_batch.py --city-name "Onagawa, Miyagi, Japan" --overwrite
```

This writes city outputs under `tsunami_study/<sanitized_city_name>/` and generates:
- `metrics.json`
- `metrics.csv`
