#!/usr/bin/env python3
"""Run the tsunami notebook non-interactively for one region.

This script is a first milestone runner for Issue #1:
- Reuses `examples/tsunami.ipynb` by patching key cells with CLI inputs.
- Executes the notebook headlessly via `jupyter nbconvert --execute`.
- Writes normalized metrics (JSON + CSV) from `population.csv`.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional
import re


LOGGER = logging.getLogger("tsunami_batch")


def sanitize_filename(name: str) -> str:
    """Fallback sanitizer for deterministic output folder names."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    cleaned = cleaned.strip("_")
    return cleaned or "region"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tsunami study for one region")
    parser.add_argument("--city-name", default=None, help="City name for geocoding")
    parser.add_argument(
        "--aoi-path",
        default=None,
        help="Optional AOI file path (.gpkg/.geojson/.shp). If provided, overrides map drawing.",
    )
    parser.add_argument(
        "--results-path", default="tsunami_study", help="Output root folder"
    )
    parser.add_argument(
        "--notebook-path", default="examples/tsunami.ipynb", help="Source notebook"
    )
    parser.add_argument(
        "--executed-notebook-path",
        default=None,
        help="Optional path to save executed notebook copy. Default: <results_path>/_executed_notebooks/<city>.ipynb",
    )

    parser.add_argument("--min-height", type=float, default=2)
    parser.add_argument("--max-height", type=float, default=10)
    parser.add_argument("--dem-type", default="COP30")
    parser.add_argument("--min-distance", type=float, default=100)
    parser.add_argument("--max-distance", type=float, default=1000)
    parser.add_argument("--n-accessibility-scores", type=int, default=5)
    parser.add_argument("--min-edge-length", type=float, default=50)
    parser.add_argument("--h3-resolution", type=int, default=11)
    parser.add_argument(
        "--show-maps", action="store_true", help="Keep notebook map outputs enabled"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Delete city output folder before run"
    )

    parser.add_argument("--affected-threshold", type=float, default=0.4)
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Notebook execution timeout in seconds",
    )

    return parser.parse_args()


def resolve_city_name(city_name: Optional[str], aoi_path: Optional[str]) -> str:
    if city_name:
        return city_name
    if aoi_path:
        return Path(aoi_path).stem.replace("_", " ").replace("-", " ")
    raise ValueError("Provide either --city-name or --aoi-path")


def setup_logging(results_path: Path, city_filename: str) -> Path:
    log_dir = results_path / "_batch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{city_filename}.log"

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    return log_path


def patch_notebook(
    notebook: dict, args: argparse.Namespace, city_name: str, aoi_path: Optional[str]
) -> None:
    """Patch specific cells in tsunami.ipynb for headless deterministic execution."""

    # Cell 4: parameter block
    notebook["cells"][4]["source"] = [
        f"city_name = {city_name!r}\n",
        "\n",
        f"min_height = {args.min_height}\n",
        f"max_height = {args.max_height}\n",
        f"dem_type = {args.dem_type!r}\n",
        "\n",
        f"min_distance = {args.min_distance}\n",
        f"max_distance = {args.max_distance}\n",
        "\n",
        f"n_accessibility_scores = {args.n_accessibility_scores}\n",
        "\n",
        "download_buffer = max_distance\n",
        f"min_edge_length = {args.min_edge_length}\n",
        f"h3_resolution = {args.h3_resolution}\n",
        "\n",
        f"show_maps = {bool(args.show_maps)}\n",
        f"overwrite = {bool(args.overwrite)}\n",
    ]

    # Cell 5: output path block
    notebook["cells"][5]["source"] = [
        f"results_path = {str(args.results_path)!r}\n",
        "city_filename = utils.sanitize_filename(city_name)\n",
        'city_results_path = results_path + "/" + city_filename\n',
    ]

    # Cell 11: remove interactive drawing map creation
    notebook["cells"][11]["source"] = [
        "# Interactive map drawing is disabled for batch mode.\n",
        "aoi_4326 = aoi.to_crs(4326)\n",
        "centroid = aoi_4326.union_all().centroid\n",
        "center = [centroid.y, centroid.x]\n",
        "m = None\n",
    ]

    # Cell 12: deterministic AOI choice (custom file OR full city boundary)
    if aoi_path:
        source = [
            f"aoi = gpd.read_file({str(aoi_path)!r})\n",
            "aoi\n",
        ]
    else:
        source = [
            "aoi = utils.get_city_geometry(city_name)\n",
            "aoi\n",
        ]
    notebook["cells"][12]["source"] = source


def run_notebook(notebook_path: Path, timeout: int) -> None:
    if shutil.which("jupyter") is None:
        raise RuntimeError(
            "`jupyter` command not found. Install Jupyter to run notebook execution, "
            "for example: pip install jupyterlab"
        )

    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
        str(notebook_path),
    ]
    LOGGER.info("Executing notebook: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr_tail = "\n".join(proc.stderr.splitlines()[-60:])
        stdout_tail = "\n".join(proc.stdout.splitlines()[-40:])
        raise RuntimeError(
            "Notebook execution failed with non-zero exit code.\n"
            f"Command: {' '.join(cmd)}\n\n"
            f"STDOUT (tail):\n{stdout_tail}\n\nSTDERR (tail):\n{stderr_tail}"
        )


def compute_population_metrics(
    population_csv_path: Path, affected_threshold: float
) -> dict:
    if not population_csv_path.exists():
        raise FileNotFoundError(f"population.csv not found: {population_csv_path}")

    total_population = 0.0
    affected_population = 0.0
    weighted_risk_sum = 0.0
    valid_rows = 0

    with population_csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"accessibility", "population"}
        if not required_cols.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"population.csv must contain columns {sorted(required_cols)}; "
                f"found {reader.fieldnames}"
            )

        for row in reader:
            try:
                accessibility = float(row["accessibility"])
                population = float(row["population"])
            except (TypeError, ValueError):
                continue

            if population <= 0:
                continue

            valid_rows += 1
            total_population += population
            weighted_risk_sum += population * (1 - accessibility)
            if accessibility < affected_threshold:
                affected_population += population

    if total_population <= 0:
        raise ValueError("Total population is zero after filtering valid rows.")

    severity_index = weighted_risk_sum / total_population
    affected_ratio = affected_population / total_population

    return {
        "severity_index": round(severity_index, 6),
        "affected_population": round(affected_population, 2),
        "total_population": round(total_population, 2),
        "affected_ratio": round(affected_ratio, 6),
        "affected_threshold": affected_threshold,
        "valid_rows": valid_rows,
    }


def write_metrics(
    city_results_path: Path, city_name: str, metrics: dict
) -> tuple[Path, Path]:
    metrics_json_path = city_results_path / "metrics.json"
    metrics_csv_path = city_results_path / "metrics.csv"

    payload = {"city_name": city_name, **metrics}

    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with metrics_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(payload.keys()))
        writer.writeheader()
        writer.writerow(payload)

    return metrics_json_path, metrics_csv_path


def main() -> int:
    args = parse_args()

    city_name = resolve_city_name(args.city_name, args.aoi_path)
    city_filename = sanitize_filename(city_name)

    results_path = Path(args.results_path)
    city_results_path = results_path / city_filename
    log_path = setup_logging(results_path, city_filename)

    LOGGER.info("Starting tsunami batch run for city=%s", city_name)
    LOGGER.info("Log file: %s", log_path)

    notebook_src = Path(args.notebook_path)
    if not notebook_src.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_src}")

    with notebook_src.open("r", encoding="utf-8") as f:
        notebook = json.load(f)

    patch_notebook(notebook, args=args, city_name=city_name, aoi_path=args.aoi_path)

    with tempfile.TemporaryDirectory(prefix="tsunami_batch_") as tmp:
        tmp_path = Path(tmp)
        run_notebook_path = tmp_path / "tsunami_run.ipynb"
        with run_notebook_path.open("w", encoding="utf-8") as f:
            json.dump(notebook, f)

        run_notebook(run_notebook_path, timeout=args.timeout)

        if args.executed_notebook_path:
            executed_notebook_path = Path(args.executed_notebook_path)
        else:
            executed_notebook_path = (
                results_path / "_executed_notebooks" / f"{city_filename}.executed.ipynb"
            )
        executed_notebook_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_notebook_path, executed_notebook_path)
        LOGGER.info("Saved executed notebook: %s", executed_notebook_path)

    population_csv_path = city_results_path / "population.csv"
    metrics = compute_population_metrics(
        population_csv_path, affected_threshold=args.affected_threshold
    )
    metrics_json_path, metrics_csv_path = write_metrics(
        city_results_path, city_name, metrics
    )

    LOGGER.info("Run complete for %s", city_name)
    LOGGER.info("City output directory: %s", city_results_path)
    LOGGER.info("Metrics JSON: %s", metrics_json_path)
    LOGGER.info("Metrics CSV: %s", metrics_csv_path)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
