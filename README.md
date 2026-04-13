UrbanAccessAnalyzer

## Development checks

Run these before opening a PR:

```bash
pip install -e .[dev]
black --check .
ruff check .
pytest -q
```

If the repository has no tests yet, `pytest` may not run any tests. Add tests with new features when possible.
