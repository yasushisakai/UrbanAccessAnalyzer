from __future__ import annotations

import csv
import importlib.util
import json
from argparse import Namespace
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "tsunami_multi_batch.py"
)
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
        writer.writerow(
            {
                "center_lat": "42.1",
                "center_lng": "-71.1",
                "square_km": "1.0",
                "city_name": "A",
            }
        )
        writer.writerow(
            {
                "center_lat": "42.2",
                "center_lng": "-71.2",
                "square_km": "1.0",
                "city_name": "B",
            }
        )

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


def _write_batch_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "center_lat",
                "center_lng",
                "nickname",
                "city_name",
                "square_km",
                "enabled",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "center_lat": "42.10000",
                "center_lng": "-71.10000",
                "nickname": "Cached",
                "city_name": "Cached City",
                "square_km": "1.0",
                "enabled": "true",
            }
        )


def test_main_resume_skips_cached_row(monkeypatch, tmp_path: Path) -> None:
    batch_csv = tmp_path / "batch.csv"
    _write_batch_csv(batch_csv)

    results_path = tmp_path / "results"
    city_dir = results_path / "Cached_City"
    city_dir.mkdir(parents=True, exist_ok=True)
    write_population_csv(
        city_dir / "population.csv",
        [{"accessibility": "0.8", "population": "100"}],
    )
    (city_dir / "metrics.json").write_text("{}", encoding="utf-8")

    summary = results_path / "comparison_summary.csv"
    results_path.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "city_filename",
                "center_lat",
                "center_lng",
                "square_km",
                "Final_tsunami_safety_pop_weighted",
                "Final_tsunami_safety_mean",
                "total_population",
                "normalization_elevation_danger_m",
                "normalization_elevation_safe_m",
                "normalization_population_cell_cap",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "city_filename": "Cached_City",
                "center_lat": "42.10000",
                "center_lng": "-71.10000",
                "square_km": "1.0",
                "Final_tsunami_safety_pop_weighted": "0.5",
                "Final_tsunami_safety_mean": "0.4",
                "total_population": "100",
                "normalization_elevation_danger_m": "2",
                "normalization_elevation_safe_m": "30",
                "normalization_population_cell_cap": "120",
            }
        )

    monkeypatch.setattr(
        MODULE,
        "parse_args",
        lambda: Namespace(
            batch_csv=str(batch_csv),
            results_path=str(results_path),
            timeout=1800,
            square_km=1.0,
            overwrite=False,
            resume=True,
            limit=None,
            fail_fast=False,
            dry_run=False,
            python="python",
            script_path="examples/tsunami_batch.py",
        ),
    )

    def _should_not_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called for resume cache hit")

    monkeypatch.setattr(MODULE.subprocess, "run", _should_not_run)
    monkeypatch.setattr(MODULE, "_git_commit_hash", lambda: "test-sha")

    rc = MODULE.main()
    assert rc == 0

    report = results_path / "multi_batch_report.csv"
    with report.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["status"] == "cached"

    metadata = json.loads((results_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["defaults"]["resume"] is True
    assert metadata["run_counts"]["cached"] == 1


def test_main_no_resume_runs_even_with_cached_files(
    monkeypatch, tmp_path: Path
) -> None:
    batch_csv = tmp_path / "batch.csv"
    _write_batch_csv(batch_csv)

    results_path = tmp_path / "results"
    city_dir = results_path / "Cached_City"
    city_dir.mkdir(parents=True, exist_ok=True)
    write_population_csv(
        city_dir / "population.csv",
        [{"accessibility": "0.8", "population": "100"}],
    )
    (city_dir / "metrics.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        MODULE,
        "parse_args",
        lambda: Namespace(
            batch_csv=str(batch_csv),
            results_path=str(results_path),
            timeout=1800,
            square_km=1.0,
            overwrite=False,
            resume=False,
            limit=None,
            fail_fast=False,
            dry_run=False,
            python="python",
            script_path="examples/tsunami_batch.py",
        ),
    )

    calls = {"count": 0}

    def _run(*args, **kwargs):
        calls["count"] += 1

        class _Proc:
            returncode = 0
            stderr = ""

        return _Proc()

    monkeypatch.setattr(MODULE.subprocess, "run", _run)
    monkeypatch.setattr(MODULE, "_git_commit_hash", lambda: "test-sha")

    rc = MODULE.main()
    assert rc == 0
    assert calls["count"] == 1
