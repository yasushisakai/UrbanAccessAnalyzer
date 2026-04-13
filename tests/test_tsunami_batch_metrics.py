from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "tsunami_batch.py"
SPEC = importlib.util.spec_from_file_location("tsunami_batch", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def write_population_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["accessibility", "population"])
        writer.writeheader()
        writer.writerows(rows)


def test_compute_population_metrics_basic(tmp_path: Path) -> None:
    csv_path = tmp_path / "population.csv"
    write_population_csv(
        csv_path,
        [
            {"accessibility": "1.0", "population": "50"},
            {"accessibility": "0.5", "population": "30"},
            {"accessibility": "0.0", "population": "20"},
        ],
    )

    metrics = MODULE.compute_population_metrics(csv_path, affected_threshold=0.4)

    assert metrics["total_population"] == 100.0
    assert metrics["affected_population"] == 20.0
    assert metrics["affected_ratio"] == 0.2
    # (50*(1-1.0) + 30*(1-0.5) + 20*(1-0.0)) / 100 = 0.35
    assert metrics["severity_index"] == 0.35


def test_compute_population_metrics_skips_invalid_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "population.csv"
    write_population_csv(
        csv_path,
        [
            {"accessibility": "0.2", "population": "10"},
            {"accessibility": "", "population": "5"},
            {"accessibility": "0.8", "population": "-2"},
            {"accessibility": "bad", "population": "7"},
        ],
    )

    metrics = MODULE.compute_population_metrics(csv_path, affected_threshold=0.4)

    assert metrics["valid_rows"] == 1
    assert metrics["total_population"] == 10.0
    assert metrics["affected_population"] == 10.0
    assert metrics["affected_ratio"] == 1.0
