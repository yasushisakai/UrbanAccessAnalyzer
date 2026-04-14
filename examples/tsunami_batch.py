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
import math


LOGGER = logging.getLogger("tsunami_batch")

# Fixed normalization settings for cross-location comparability.
# These are intentionally global (not per-AOI min/max) so scores can be ranked across lat/lng runs.
ELEVATION_DANGER_M = 2.0
ELEVATION_SAFE_M = 30.0
POPULATION_CELL_CAP = 120.0


def sanitize_filename(name: str) -> str:
    """Fallback sanitizer aligned with UrbanAccessAnalyzer.utils.sanitize_filename."""
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "_", name.strip().lower())
    cleaned = cleaned.strip("_")
    return cleaned or "region"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tsunami study for one region")
    parser.add_argument("--city-name", default=None, help="City name for geocoding")
    parser.add_argument(
        "--nickname",
        default=None,
        help="Optional friendly label for this run (e.g., 'Cambridge', 'Boston North End')",
    )
    parser.add_argument(
        "--aoi-path",
        default=None,
        help="Optional AOI file path (.gpkg/.geojson/.shp). If provided, overrides map drawing.",
    )
    parser.add_argument(
        "--center-lat",
        type=float,
        default=None,
        help="Center latitude for auto-generated square AOI (used with --center-lng)",
    )
    parser.add_argument(
        "--center-lng",
        type=float,
        default=None,
        help="Center longitude for auto-generated square AOI (used with --center-lat)",
    )
    parser.add_argument(
        "--square-km",
        type=float,
        default=1.0,
        help="Square AOI side length in kilometers when using center coordinates (default: 1.0)",
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


def resolve_city_name(
    city_name: Optional[str],
    aoi_path: Optional[str],
    center_lat: Optional[float],
    center_lng: Optional[float],
) -> str:
    if city_name:
        return city_name
    if aoi_path:
        return Path(aoi_path).stem.replace("_", " ").replace("-", " ")
    if center_lat is not None and center_lng is not None:
        return f"center_{center_lat:.5f}_{center_lng:.5f}"
    raise ValueError(
        "Provide either --city-name, --aoi-path, or both --center-lat and --center-lng"
    )


def build_square_geojson(center_lat: float, center_lng: float, square_km: float) -> dict:
    if square_km <= 0:
        raise ValueError("--square-km must be > 0")

    lat_rad = center_lat * 3.141592653589793 / 180.0
    half_km = square_km / 2.0

    # Approximate km-to-degree conversion for local square generation.
    delta_lat = half_km / 110.574
    cos_lat = max(abs(math.cos(lat_rad)), 1e-6)
    delta_lng = half_km / (111.320 * cos_lat)

    min_lng = center_lng - delta_lng
    max_lng = center_lng + delta_lng
    min_lat = center_lat - delta_lat
    max_lat = center_lat + delta_lat

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": f"square_{square_km}km",
                    "center_lat": center_lat,
                    "center_lng": center_lng,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [min_lng, min_lat],
                            [max_lng, min_lat],
                            [max_lng, max_lat],
                            [min_lng, max_lat],
                            [min_lng, min_lat],
                        ]
                    ],
                },
            }
        ],
    }


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
        f"results_path = {str(Path(args.results_path).resolve())!r}\n",
        "city_filename = utils.sanitize_filename(city_name)\n",
        'city_results_path = results_path + "/" + city_filename\n',
    ]

    # Cell 11: remove interactive drawing map creation
    notebook["cells"][11]["source"] = [
        "# Interactive map drawing is disabled for batch mode.\n",
        "center = None\n",
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


def _weighted_mean(values, weights) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return float("nan")
    v = values[valid].astype(float)
    w = weights[valid].astype(float)
    return float((v * w).sum() / w.sum())


def write_comparison_summary(
    results_path: Path,
    city_name: str,
    city_results_path: Path,
    args: argparse.Namespace,
    metrics: dict,
) -> Path:
    """Append one run row to a cross-location comparison CSV."""
    gdf = _prepare_abc_layers(city_results_path)

    population = gdf["population"].fillna(0.0)
    row = {
        "city_name": city_name,
        "nickname": (args.nickname or "").strip(),
        "city_filename": sanitize_filename(city_name),
        "center_lat": args.center_lat,
        "center_lng": args.center_lng,
        "square_km": args.square_km,
        "aoi_path": args.aoi_path or "",
        "total_population": round(float(population.sum()), 2),
        "severity_index": metrics.get("severity_index"),
        "affected_ratio": metrics.get("affected_ratio"),
        "A_tsunami_risk_mean": round(float(gdf["tsunami_risk_altitude"].mean(skipna=True)), 6),
        "B_population_exposure_mean": round(float(gdf["population_exposure"].mean(skipna=True)), 6),
        "C_access_risk_mean": round(float(gdf["evacuation_access_risk"].mean(skipna=True)), 6),
        "ABC_blended_mean": round(float(gdf["blended_risk_abc"].mean(skipna=True)), 6),
        "Final_tsunami_safety_mean": round(float(gdf["tsunami_safety_score"].mean(skipna=True)), 6),
        "A_tsunami_risk_pop_weighted": round(_weighted_mean(gdf["tsunami_risk_altitude"], population), 6),
        "B_population_exposure_pop_weighted": round(_weighted_mean(gdf["population_exposure"], population), 6),
        "C_access_risk_pop_weighted": round(_weighted_mean(gdf["evacuation_access_risk"], population), 6),
        "ABC_blended_pop_weighted": round(_weighted_mean(gdf["blended_risk_abc"], population), 6),
        "Final_tsunami_safety_pop_weighted": round(_weighted_mean(gdf["tsunami_safety_score"], population), 6),
        "normalization_elevation_danger_m": ELEVATION_DANGER_M,
        "normalization_elevation_safe_m": ELEVATION_SAFE_M,
        "normalization_population_cell_cap": POPULATION_CELL_CAP,
    }

    summary_path = results_path / "comparison_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "city_name",
        "nickname",
        "city_filename",
        "center_lat",
        "center_lng",
        "square_km",
        "aoi_path",
        "total_population",
        "severity_index",
        "affected_ratio",
        "A_tsunami_risk_mean",
        "B_population_exposure_mean",
        "C_access_risk_mean",
        "ABC_blended_mean",
        "Final_tsunami_safety_mean",
        "A_tsunami_risk_pop_weighted",
        "B_population_exposure_pop_weighted",
        "C_access_risk_pop_weighted",
        "ABC_blended_pop_weighted",
        "Final_tsunami_safety_pop_weighted",
        "normalization_elevation_danger_m",
        "normalization_elevation_safe_m",
        "normalization_population_cell_cap",
    ]

    existing_rows = []
    write_header = True
    if summary_path.exists():
        with summary_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            existing_fields = reader.fieldnames or []
        if existing_fields == fieldnames:
            write_header = False

    if write_header and existing_rows:
        # Schema changed: rewrite file preserving old rows where possible.
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for old in existing_rows:
                writer.writerow({k: old.get(k, "") for k in fieldnames})
            writer.writerow(row)
    else:
        with summary_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    return summary_path


def _normalize_to_unit_interval(values):
    values = values.astype(float)
    finite_values = values[values.notna()]
    if len(finite_values) == 0:
        return values * 0.0
    min_v = float(finite_values.min())
    max_v = float(finite_values.max())
    if max_v <= min_v:
        return values * 0.0
    return (values - min_v) / (max_v - min_v)


def _normalize_population_exposure(population_values):
    """Stable population normalization for cross-location ranking.

    Uses a capped log transform so extremely dense cells do not dominate,
    and values remain comparable across independent AOIs.
    """
    import numpy as np

    pop = population_values.fillna(0.0).clip(lower=0.0)
    cap = max(float(POPULATION_CELL_CAP), 1.0)
    return np.log1p(pop) / np.log1p(cap)


def _elevation_to_tsunami_risk(elevation_values):
    """Convert elevation meters to tsunami-risk proxy in [0,1] with fixed thresholds.

    - <= ELEVATION_DANGER_M -> 1.0 risk
    - >= ELEVATION_SAFE_M   -> 0.0 risk
    """
    elev = elevation_values.astype(float)
    denom = max(ELEVATION_SAFE_M - ELEVATION_DANGER_M, 1e-6)
    safe_fraction = ((elev - ELEVATION_DANGER_M) / denom).clip(0.0, 1.0)
    return 1.0 - safe_fraction


def _prepare_abc_layers(city_results_path: Path):
    import geopandas as gpd
    import pandas as pd
    import UrbanAccessAnalyzer.h3_utils as h3_utils

    population_gpkg_path = city_results_path / "population.gpkg"
    dem_path = city_results_path / "dem.tif"

    if not population_gpkg_path.exists():
        raise FileNotFoundError(f"population.gpkg not found: {population_gpkg_path}")
    if not dem_path.exists():
        raise FileNotFoundError(f"dem.tif not found: {dem_path}")

    gdf = gpd.read_file(population_gpkg_path)
    if "accessibility" not in gdf.columns or "population" not in gdf.columns:
        raise ValueError("population.gpkg must include 'accessibility' and 'population' columns")

    gdf["population"] = gdf["population"].fillna(0.0)

    # A) Pure tsunami-risk proxy from elevation: lower elevation => higher risk.
    dem_h3 = h3_utils.from_raster(str(dem_path), resolution=11, method="mean")
    dem_h3 = dem_h3.reset_index().rename(columns={"value": "elevation_m"})
    gdf = gdf.merge(dem_h3[["h3_cell", "elevation_m"]], on="h3_cell", how="left")
    gdf["tsunami_risk_altitude"] = _elevation_to_tsunami_risk(gdf["elevation_m"])

    # B) Population exposure (population concentration only; fixed transform across runs).
    gdf["population_exposure"] = _normalize_population_exposure(gdf["population"])

    # C) Evacuation-access risk (lower access => higher risk).
    accessibility = pd.to_numeric(gdf["accessibility"], errors="coerce").clip(0.0, 1.0)
    gdf["evacuation_access_risk"] = 1.0 - accessibility

    # Blend: A + B + C (equal weights), ignoring no-data cells component-wise.
    gdf["blended_risk_abc"] = gdf[
        ["tsunami_risk_altitude", "population_exposure", "evacuation_access_risk"]
    ].mean(axis=1, skipna=True)

    # Final safety score (higher is safer).
    gdf["tsunami_safety_score"] = 1.0 - gdf["blended_risk_abc"]

    return gdf


def write_heatmaps(city_results_path: Path) -> tuple[Path, Path, Path, Path]:
    """Export interactive HTML heatmaps for A/B/C components and blended A+B+C risk."""
    gdf = _prepare_abc_layers(city_results_path)

    heatmaps_dir = city_results_path / "heatmaps"
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    a_tsunami_map_path = heatmaps_dir / "a_tsunami_risk_altitude.html"
    b_population_map_path = heatmaps_dir / "b_population_exposure.html"
    c_access_map_path = heatmaps_dir / "c_evacuation_access_risk.html"
    blended_map_path = heatmaps_dir / "blended_risk_abc.html"

    satellite_tiles = "Esri.WorldImagery"
    layer_style = {
        "fillOpacity": 0.62,
        "opacity": 0.85,
        "weight": 0.35,
        "color": "#FFFFFF",
    }

    m_a = gdf.explore(
        column="tsunami_risk_altitude",
        cmap="turbo",
        vmin=0.0,
        vmax=1.0,
        legend=True,
        tiles=satellite_tiles,
        tooltip=["tsunami_risk_altitude", "elevation_m"],
        style_kwds=layer_style,
    )
    m_a.save(str(a_tsunami_map_path))

    m_b = gdf.explore(
        column="population_exposure",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        legend=True,
        tiles=satellite_tiles,
        tooltip=["population_exposure", "population"],
        style_kwds=layer_style,
    )
    m_b.save(str(b_population_map_path))

    m_c = gdf.explore(
        column="evacuation_access_risk",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        legend=True,
        tiles=satellite_tiles,
        tooltip=["evacuation_access_risk", "accessibility"],
        style_kwds=layer_style,
    )
    m_c.save(str(c_access_map_path))

    m_blend = gdf.explore(
        column="blended_risk_abc",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
        legend=True,
        tiles=satellite_tiles,
        tooltip=[
            "blended_risk_abc",
            "tsunami_risk_altitude",
            "population_exposure",
            "evacuation_access_risk",
        ],
        style_kwds=layer_style,
    )
    m_blend.save(str(blended_map_path))

    return a_tsunami_map_path, b_population_map_path, c_access_map_path, blended_map_path


def write_heatmap_images(city_results_path: Path) -> tuple[Path, Path, Path, Path]:
    """Export PNG heatmaps with explanatory captions for A/B/C components and A+B+C blend."""
    import matplotlib.pyplot as plt

    gdf = _prepare_abc_layers(city_results_path)

    # Basemap tiles require projected coordinates for contextily.
    gdf_plot = gdf.to_crs(epsg=3857)

    try:
        import contextily as ctx

        basemap_available = True
    except Exception:
        ctx = None
        basemap_available = False
        LOGGER.warning(
            "contextily is not available. PNGs will be generated without satellite basemap. "
            "Install with: pip install contextily"
        )

    heatmaps_dir = city_results_path / "heatmaps"
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    a_tsunami_png_path = heatmaps_dir / "a_tsunami_risk_altitude_satellite.png"
    b_population_png_path = heatmaps_dir / "b_population_exposure_satellite.png"
    c_access_png_path = heatmaps_dir / "c_evacuation_access_risk_satellite.png"
    blended_png_path = heatmaps_dir / "blended_risk_abc_satellite.png"

    plot_specs = [
        {
            "column": "tsunami_risk_altitude",
            "title": "A) Tsunami Risk from Altitude",
            "explanation": "High = lower elevation (higher tsunami risk proxy).",
            "cmap": "turbo",
            "vmin": 0.0,
            "vmax": 1.0,
            "output_path": a_tsunami_png_path,
        },
        {
            "column": "population_exposure",
            "title": "B) Population Exposure",
            "explanation": "High = more residents (population concentration).",
            "cmap": "viridis",
            "vmin": 0.0,
            "vmax": 1.0,
            "output_path": b_population_png_path,
        },
        {
            "column": "evacuation_access_risk",
            "title": "C) Evacuation Access Risk",
            "explanation": "High = poorer evacuation access (1 - accessibility).",
            "cmap": "magma",
            "vmin": 0.0,
            "vmax": 1.0,
            "output_path": c_access_png_path,
        },
        {
            "column": "blended_risk_abc",
            "title": "Blend: A + B + C",
            "explanation": "High = combined risk from altitude, population, and access.",
            "cmap": "inferno",
            "vmin": 0.0,
            "vmax": 1.0,
            "output_path": blended_png_path,
        },
    ]

    for spec in plot_specs:
        fig, ax = plt.subplots(figsize=(7.2, 7.2))
        gdf_plot.plot(
            column=spec["column"],
            cmap=spec["cmap"],
            ax=ax,
            legend=True,
            alpha=0.82,
            linewidth=0.25,
            edgecolor="#0f0f0f",
            vmin=spec["vmin"],
            vmax=spec["vmax"],
            legend_kwds={"shrink": 0.75, "label": spec["column"].replace("_", " ").title()},
            missing_kwds={"color": "#7f7f7f", "label": "No data"},
        )

        if basemap_available:
            try:
                ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, attribution=False)
            except Exception as exc:
                LOGGER.warning("Failed to add satellite basemap for %s: %s", spec["column"], exc)
                ax.set_facecolor("#161616")
        else:
            ax.set_facecolor("#161616")

        ax.set_axis_off()
        ax.text(
            0.5,
            0.995,
            spec["title"],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
            color="white",
            bbox={"facecolor": "black", "alpha": 0.78, "pad": 5, "edgecolor": "none"},
        )
        ax.text(
            0.5,
            0.948,
            spec["explanation"],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.68, "pad": 4, "edgecolor": "none"},
        )

        fig.tight_layout()
        fig.savefig(spec["output_path"], dpi=220, bbox_inches="tight", pad_inches=0, facecolor="#101010")
        plt.close(fig)

    return a_tsunami_png_path, b_population_png_path, c_access_png_path, blended_png_path


def write_final_tsunami_safety_image(city_results_path: Path) -> Path:
    """Export final tsunami safety score as a smooth surface over satellite imagery.

    Computation remains hex-based; only the final visualization is interpolated.
    """
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import numpy as np
    from shapely.geometry import Point

    gdf = _prepare_abc_layers(city_results_path)
    gdf_plot = gdf.to_crs(epsg=3857)

    heatmaps_dir = city_results_path / "heatmaps"
    heatmaps_dir.mkdir(parents=True, exist_ok=True)
    output_path = heatmaps_dir / "tsunami_safety_score_region.png"

    try:
        import contextily as ctx

        basemap_available = True
    except Exception:
        ctx = None
        basemap_available = False
        LOGGER.warning(
            "contextily is not available. Final safety image will be generated without satellite background. "
            "Install with: pip install contextily"
        )

    fig, ax = plt.subplots(figsize=(7.2, 7.2))

    minx, miny, maxx, maxy = gdf_plot.total_bounds
    pad_x = (maxx - minx) * 0.005
    pad_y = (maxy - miny) * 0.005
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    if basemap_available:
        try:
            ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, attribution=False)
        except Exception as exc:
            LOGGER.warning("Failed to add satellite basemap for final safety image: %s", exc)
            ax.set_facecolor("#161616")
    else:
        ax.set_facecolor("#161616")

    valid = gdf_plot[gdf_plot["tsunami_safety_score"].notna()].copy()
    if len(valid) >= 3:
        centroids = valid.geometry.centroid
        x = centroids.x.to_numpy()
        y = centroids.y.to_numpy()
        z = valid["tsunami_safety_score"].to_numpy(dtype=float)

        tri = mtri.Triangulation(x, y)
        region_shape = gdf_plot.geometry.union_all()

        # Mask triangles whose centroids fall outside the AOI to keep the surface region-bounded.
        tris = tri.triangles
        tri_cx = x[tris].mean(axis=1)
        tri_cy = y[tris].mean(axis=1)
        mask = np.array([not region_shape.contains(Point(cx, cy)) for cx, cy in zip(tri_cx, tri_cy)])
        tri.set_mask(mask)

        contour = ax.tricontourf(
            tri,
            z,
            levels=np.linspace(0.0, 1.0, 17),
            cmap="RdYlGn",
            vmin=0.0,
            vmax=1.0,
            alpha=0.72,
        )
        cbar = fig.colorbar(contour, ax=ax, shrink=0.78)
        cbar.set_label("Tsunami Safety Score (higher = safer)")

        # Optional faint boundary for geographic context.
        valid.boundary.plot(ax=ax, color="#111111", linewidth=0.2, alpha=0.25)
    else:
        # Fallback to hex fill when there are too few cells for triangulation.
        gdf_plot.plot(
            column="tsunami_safety_score",
            cmap="RdYlGn",
            vmin=0.0,
            vmax=1.0,
            ax=ax,
            legend=True,
            alpha=0.72,
            linewidth=0.35,
            edgecolor="#1a1a1a",
            legend_kwds={"shrink": 0.78, "label": "Tsunami Safety Score (higher = safer)"},
            missing_kwds={"color": "#7f7f7f", "label": "No data"},
        )

    ax.set_aspect("equal")
    ax.set_axis_off()

    ax.text(
        0.5,
        0.995,
        "Final Tsunami Safety Score (Surface)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="white",
        bbox={"facecolor": "black", "alpha": 0.78, "pad": 5, "edgecolor": "none"},
    )
    ax.text(
        0.5,
        0.948,
        "Higher score = safer (computed on hexes; displayed as smooth surface).",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.68, "pad": 4, "edgecolor": "none"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()

    if (args.center_lat is None) ^ (args.center_lng is None):
        raise ValueError("Provide both --center-lat and --center-lng together")

    city_name = resolve_city_name(
        args.city_name,
        args.aoi_path,
        args.center_lat,
        args.center_lng,
    )
    city_filename = sanitize_filename(city_name)

    results_path = Path(args.results_path)
    city_results_path = results_path / city_filename
    log_path = setup_logging(results_path, city_filename)

    LOGGER.info("Starting tsunami batch run for city=%s", city_name)
    LOGGER.info("Log file: %s", log_path)

    notebook_src = Path(args.notebook_path)
    if not notebook_src.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_src}")

    effective_aoi_path = str(Path(args.aoi_path).resolve()) if args.aoi_path else None
    if effective_aoi_path is not None and args.center_lat is not None and args.center_lng is not None:
        LOGGER.info("--aoi-path provided; ignoring --center-lat/--center-lng")

    if effective_aoi_path is None and args.center_lat is not None and args.center_lng is not None:
        generated_dir = results_path / "_generated_aoi"
        generated_dir.mkdir(parents=True, exist_ok=True)
        generated_aoi_path = generated_dir / f"{city_filename}.square.geojson"
        square_geojson = build_square_geojson(
            center_lat=args.center_lat,
            center_lng=args.center_lng,
            square_km=args.square_km,
        )
        with generated_aoi_path.open("w", encoding="utf-8") as f:
            json.dump(square_geojson, f)
        effective_aoi_path = str(generated_aoi_path.resolve())
        LOGGER.info(
            "Generated square AOI: %s (center=%s,%s size=%skm)",
            generated_aoi_path,
            args.center_lat,
            args.center_lng,
            args.square_km,
        )

    with notebook_src.open("r", encoding="utf-8") as f:
        notebook = json.load(f)

    patch_notebook(notebook, args=args, city_name=city_name, aoi_path=effective_aoi_path)

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
    a_tsunami_map_path, b_population_map_path, c_access_map_path, blended_map_path = write_heatmaps(
        city_results_path
    )
    a_tsunami_png_path, b_population_png_path, c_access_png_path, blended_png_path = write_heatmap_images(
        city_results_path
    )
    final_safety_image_path = write_final_tsunami_safety_image(city_results_path)
    comparison_summary_path = write_comparison_summary(
        results_path=results_path,
        city_name=city_name,
        city_results_path=city_results_path,
        args=args,
        metrics=metrics,
    )

    LOGGER.info("Run complete for %s", city_name)
    LOGGER.info("City output directory: %s", city_results_path)
    LOGGER.info("Metrics JSON: %s", metrics_json_path)
    LOGGER.info("Metrics CSV: %s", metrics_csv_path)
    LOGGER.info("A map (tsunami altitude risk): %s", a_tsunami_map_path)
    LOGGER.info("B map (population exposure): %s", b_population_map_path)
    LOGGER.info("C map (evacuation access risk): %s", c_access_map_path)
    LOGGER.info("Blended map (A+B+C): %s", blended_map_path)
    LOGGER.info("A image (tsunami altitude risk): %s", a_tsunami_png_path)
    LOGGER.info("B image (population exposure): %s", b_population_png_path)
    LOGGER.info("C image (evacuation access risk): %s", c_access_png_path)
    LOGGER.info("Blended image (A+B+C): %s", blended_png_path)
    LOGGER.info("Final image (tsunami safety score): %s", final_safety_image_path)
    LOGGER.info("Comparison summary CSV: %s", comparison_summary_path)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
