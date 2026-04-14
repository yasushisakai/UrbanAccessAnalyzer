from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "tsunami_multi_batch.py"
SPEC = importlib.util.spec_from_file_location("tsunami_multi_batch", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def write_population_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["accessibility", "population"])
        writer.writeheader()
        writer.writerows(rows)


def test_assess_run_quality_missing_population_csv(tmp_path: Path) -> None:
    result = MODULE.assess_run_quality(tmp_path)
    assert result["status"] == "invalid"
    assert "population.csv" in result["reason"]


def test_assess_run_quality_zero_population_invalid(tmp_path: Path) -> None:
    write_population_csv(
        tmp_path / "population.csv",
        [
            {"accessibility": "0.5", "population": "0"},
            {"accessibility": "0.2", "population": "-1"},
        ],
    )

    result = MODULE.assess_run_quality(tmp_path)
    assert result["status"] == "invalid"
    assert "total_population" in result["reason"]


def test_assess_run_quality_valid_population(tmp_path: Path) -> None:
    write_population_csv(
        tmp_path / "population.csv",
        [
            {"accessibility": "1.0", "population": "10"},
            {"accessibility": "0.5", "population": "20"},
            {"accessibility": "", "population": "5"},
        ],
    )

    result = MODULE.assess_run_quality(tmp_path)
    assert result["status"] == "success"
    assert result["diagnostics"]["valid_rows"] == 2
    assert result["diagnostics"]["accessibility_non_null_ratio"] == 2 / 3


def test_prune_invalid_rows_from_summary(tmp_path: Path) -> None:
    summary = tmp_path / "comparison_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["center_lat", "center_lng", "square_km", "city_name"],
        )
        writer.writeheader()
        writer.writerow({"center_lat": "42.1", "center_lng": "-71.1", "square_km": "1.0", "city_name": "A"})
        writer.writerow({"center_lat": "42.2", "center_lng": "-71.2", "square_km": "1.0", "city_name": "B"})

    removed = MODULE.prune_invalid_rows_from_summary(
        summary,
        [
            {"center_lat": "42.2", "center_lng": "-71.2", "square_km": "1"},
        ],
    )

    assert removed == 1
    with summary.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["city_name"] == "A"
