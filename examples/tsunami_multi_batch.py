#!/usr/bin/env python3
"""Run tsunami_batch.py for multiple locations defined in a CSV file.

CSV columns:
- Required: center_lat, center_lng
- Optional: nickname, city_name, square_km, enabled, aoi_path, timeout

This wrapper preserves tsunami_batch.py as the single-location engine.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tsunami_batch.py for multiple rows from CSV"
    )
    parser.add_argument(
        "--batch-csv", required=True, help="CSV file with center_lat/center_lng rows"
    )
    parser.add_argument(
        "--results-path", default="tsunami_study", help="Shared output root"
    )
    parser.add_argument(
        "--timeout", type=int, default=1800, help="Default per-run notebook timeout"
    )
    parser.add_argument(
        "--square-km",
        type=float,
        default=1.0,
        help="Default square size when missing in CSV",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Pass --overwrite to each single run"
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Reuse completed region outputs when artifacts and summary row exist (default)",
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Force re-running all enabled regions unless --dry-run",
    )
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--limit", type=int, default=None, help="Run only first N enabled rows"
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="Stop immediately on first failed row"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without running"
    )
    parser.add_argument(
        "--python", default=sys.executable, help="Python executable for child runs"
    )
    parser.add_argument(
        "--script-path",
        default="examples/tsunami_batch.py",
        help="Path to single-run script",
    )
    return parser.parse_args()


def _is_enabled(value: str) -> bool:
    if value is None:
        return True
    v = str(value).strip().lower()
    if v == "":
        return True
    return v not in {"0", "false", "no", "off", "disabled"}


def _build_command(row: Dict[str, str], args: argparse.Namespace) -> List[str]:
    lat = (row.get("center_lat") or "").strip()
    lng = (row.get("center_lng") or "").strip()
    if not lat or not lng:
        raise ValueError("Missing required center_lat/center_lng")

    timeout = (row.get("timeout") or "").strip() or str(args.timeout)
    square_km = (row.get("square_km") or "").strip() or str(args.square_km)

    cmd = [
        args.python,
        args.script_path,
        "--center-lat",
        lat,
        "--center-lng",
        lng,
        "--square-km",
        square_km,
        "--results-path",
        args.results_path,
        "--timeout",
        timeout,
    ]

    nickname = (row.get("nickname") or "").strip()
    if nickname:
        cmd += ["--nickname", nickname]

    city_name = (row.get("city_name") or "").strip()
    if city_name:
        cmd += ["--city-name", city_name]

    aoi_path = (row.get("aoi_path") or "").strip()
    if aoi_path:
        cmd += ["--aoi-path", aoi_path]

    if args.overwrite:
        cmd.append("--overwrite")

    return cmd


def sanitize_filename(name: str) -> str:
    """Keep folder naming aligned with tsunami_batch.py output conventions."""
    safe = (name or "").strip()
    safe = "_".join(safe.split())
    allowed = "._-"
    return (
        "".join(ch for ch in safe if ch.isalnum() or ch in allowed).strip("._-")
        or "output"
    )


def city_filename_for_row(row: Dict[str, str]) -> str:
    city_name = (row.get("city_name") or "").strip()
    if city_name:
        return sanitize_filename(city_name)

    lat = (row.get("center_lat") or "").strip()
    lng = (row.get("center_lng") or "").strip()
    return f"center_{float(lat):.5f}_{float(lng):.5f}"


def _normalize_numeric_token(value: str) -> str:
    try:
        return f"{float(value):.8f}"
    except Exception:
        return str(value).strip()


def _summary_key(
    center_lat: str, center_lng: str, square_km: str
) -> tuple[str, str, str]:
    return (
        _normalize_numeric_token(center_lat),
        _normalize_numeric_token(center_lng),
        _normalize_numeric_token(square_km),
    )


def prune_invalid_rows_from_summary(
    summary_path: Path, invalid_runs: List[Dict[str, str]]
) -> int:
    """Remove comparison_summary rows for runs marked invalid by QA gates."""
    if not summary_path.exists() or not invalid_runs:
        return 0

    with summary_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    invalid_keys = {
        _summary_key(
            run.get("center_lat", ""),
            run.get("center_lng", ""),
            run.get("square_km", ""),
        )
        for run in invalid_runs
    }

    kept_rows = []
    removed = 0
    for row in rows:
        key = _summary_key(
            row.get("center_lat", ""),
            row.get("center_lng", ""),
            row.get("square_km", ""),
        )
        if key in invalid_keys:
            removed += 1
            continue
        kept_rows.append(row)

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    return removed


def assess_run_quality(city_results_path: Path) -> Dict[str, Any]:
    """Apply lightweight QA gates and return structured status/reason/diagnostics."""
    diagnostics: Dict[str, Any] = {
        "valid_rows": 0,
        "total_rows": 0,
        "total_population": 0.0,
        "positive_population_ratio": 0.0,
        "accessibility_non_null_ratio": 0.0,
        "geometry_non_empty_ratio": "",
    }

    population_csv_path = city_results_path / "population.csv"
    if not population_csv_path.exists():
        return {
            "status": "invalid",
            "reason": "population.csv missing",
            "diagnostics": diagnostics,
        }

    with population_csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"accessibility", "population"}
        if not required_cols.issubset(set(reader.fieldnames or [])):
            return {
                "status": "invalid",
                "reason": f"population.csv missing required columns: {sorted(required_cols)}",
                "diagnostics": diagnostics,
            }

        accessibility_non_null = 0
        positive_population_rows = 0
        total_population = 0.0
        valid_rows = 0

        for row in reader:
            diagnostics["total_rows"] += 1

            accessibility_raw = (row.get("accessibility") or "").strip()
            if accessibility_raw != "":
                accessibility_non_null += 1

            try:
                population = float(row.get("population", ""))
            except Exception:
                continue

            if population > 0:
                positive_population_rows += 1
                total_population += population

                try:
                    float(accessibility_raw)
                    valid_rows += 1
                except Exception:
                    continue

    total_rows = diagnostics["total_rows"]
    if total_rows <= 0:
        return {
            "status": "invalid",
            "reason": "population.csv has no data rows",
            "diagnostics": diagnostics,
        }

    diagnostics["valid_rows"] = valid_rows
    diagnostics["total_population"] = round(total_population, 2)
    diagnostics["positive_population_ratio"] = positive_population_rows / total_rows
    diagnostics["accessibility_non_null_ratio"] = accessibility_non_null / total_rows

    if diagnostics["total_population"] <= 0:
        return {
            "status": "invalid",
            "reason": "total_population <= 0 after filtering",
            "diagnostics": diagnostics,
        }

    if diagnostics["accessibility_non_null_ratio"] <= 0:
        return {
            "status": "invalid",
            "reason": "accessibility coverage is zero",
            "diagnostics": diagnostics,
        }

    population_gpkg_path = city_results_path / "population.gpkg"
    if population_gpkg_path.exists():
        try:
            import geopandas as gpd

            gdf = gpd.read_file(population_gpkg_path)
            if len(gdf) <= 0:
                return {
                    "status": "invalid",
                    "reason": "population.gpkg has no geometries",
                    "diagnostics": diagnostics,
                }

            non_empty_ratio = float((~gdf.geometry.is_empty).mean())
            diagnostics["geometry_non_empty_ratio"] = non_empty_ratio
            if non_empty_ratio <= 0:
                return {
                    "status": "invalid",
                    "reason": "AOI geometry coverage is empty",
                    "diagnostics": diagnostics,
                }

            if "accessibility" not in gdf.columns:
                return {
                    "status": "invalid",
                    "reason": "population.gpkg missing accessibility column",
                    "diagnostics": diagnostics,
                }
        except Exception as exc:
            return {
                "status": "invalid",
                "reason": f"population.gpkg QA read failed: {exc}",
                "diagnostics": diagnostics,
            }

    return {
        "status": "success",
        "reason": "",
        "diagnostics": diagnostics,
    }


def _summary_contains_row(
    summary_path: Path, center_lat: str, center_lng: str, square_km: str
) -> bool:
    if not summary_path.exists():
        return False

    wanted = _summary_key(center_lat, center_lng, square_km)
    with summary_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = _summary_key(
                row.get("center_lat", ""),
                row.get("center_lng", ""),
                row.get("square_km", ""),
            )
            if key == wanted:
                return True
    return False


def _resume_cache_hit(
    row: Dict[str, str], args: argparse.Namespace, results_path: Path
) -> Dict[str, Any]:
    if args.overwrite or not getattr(args, "resume", True):
        return {"hit": False, "reason": "", "diagnostics": {}}

    city_dir = results_path / city_filename_for_row(row)
    if not city_dir.exists():
        return {"hit": False, "reason": "", "diagnostics": {}}

    if (
        not (city_dir / "population.csv").exists()
        or not (city_dir / "metrics.json").exists()
    ):
        return {"hit": False, "reason": "", "diagnostics": {}}

    square_km = (row.get("square_km") or "").strip() or str(args.square_km)
    if not _summary_contains_row(
        results_path / "comparison_summary.csv",
        (row.get("center_lat") or "").strip(),
        (row.get("center_lng") or "").strip(),
        square_km,
    ):
        return {"hit": False, "reason": "", "diagnostics": {}}

    quality = assess_run_quality(city_dir)
    if quality.get("status") != "success":
        return {"hit": False, "reason": "", "diagnostics": {}}

    return {
        "hit": True,
        "reason": "resume cache hit (existing artifacts reused)",
        "diagnostics": quality.get("diagnostics", {}),
    }


def _git_commit_hash() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if getattr(proc, "returncode", 1) != 0:
        return ""
    return str(getattr(proc, "stdout", "") or "").strip()


def _write_metadata(
    results_path: Path,
    args: argparse.Namespace,
    runs: List[Dict[str, Any]],
    started_unix: float,
    finished_unix: float,
) -> Path:
    status_counts: Dict[str, int] = {}
    for run in runs:
        status = str(run.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    metadata = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": datetime.fromtimestamp(
            started_unix, timezone.utc
        ).isoformat(),
        "completed_at_utc": datetime.fromtimestamp(
            finished_unix, timezone.utc
        ).isoformat(),
        "elapsed_sec": round(finished_unix - started_unix, 2),
        "git_commit": _git_commit_hash(),
        "batch_csv": str(Path(args.batch_csv).resolve()),
        "results_path": str(Path(args.results_path).resolve()),
        "script_path": str(args.script_path),
        "python": str(args.python),
        "defaults": {
            "timeout": int(args.timeout),
            "square_km": float(args.square_km),
            "overwrite": bool(args.overwrite),
            "resume": bool(getattr(args, "resume", True)),
            "dry_run": bool(args.dry_run),
            "fail_fast": bool(args.fail_fast),
            "limit": args.limit,
        },
        "run_counts": status_counts,
    }

    metadata_path = results_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata_path


def _regen_overall_ranking(results_path: Path) -> Path | None:
    summary_path = results_path / "comparison_summary.csv"
    if not summary_path.exists():
        return None

    with summary_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Keep latest row per location key.
    by_key = {}
    for row in rows:
        key = (
            row.get("city_filename", ""),
            row.get("center_lat", ""),
            row.get("center_lng", ""),
            row.get("square_km", ""),
        )
        by_key[key] = row
    clean_rows = list(by_key.values())

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(clean_rows)

    ranking_rows = []
    for row in clean_rows:
        score_raw = row.get("Final_tsunami_safety_pop_weighted", "")
        if not score_raw:
            continue
        try:
            score = float(score_raw)
        except Exception:
            continue

        ranking_rows.append(
            {
                "rank_score": score,
                "city_name": row.get("city_name", ""),
                "nickname": row.get("nickname", ""),
                "center_lat": row.get("center_lat", ""),
                "center_lng": row.get("center_lng", ""),
                "square_km": row.get("square_km", ""),
                "final_tsunami_safety_pop_weighted": f"{score:.6f}",
                "final_tsunami_safety_mean": row.get("Final_tsunami_safety_mean", ""),
                "total_population": row.get("total_population", ""),
                "normalization_elevation_danger_m": row.get(
                    "normalization_elevation_danger_m", ""
                ),
                "normalization_elevation_safe_m": row.get(
                    "normalization_elevation_safe_m", ""
                ),
                "normalization_population_cell_cap": row.get(
                    "normalization_population_cell_cap", ""
                ),
            }
        )

    ranking_rows.sort(key=lambda x: x["rank_score"], reverse=True)
    for i, row in enumerate(ranking_rows, start=1):
        row["rank"] = i

    fieldnames = [
        "rank",
        "city_name",
        "nickname",
        "center_lat",
        "center_lng",
        "square_km",
        "final_tsunami_safety_pop_weighted",
        "final_tsunami_safety_mean",
        "total_population",
        "normalization_elevation_danger_m",
        "normalization_elevation_safe_m",
        "normalization_population_cell_cap",
    ]

    ranking_path = results_path / "overall_safety_ranking.csv"
    with ranking_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranking_rows:
            row.pop("rank_score", None)
            writer.writerow(row)

    return ranking_path


def main() -> int:
    args = parse_args()
    batch_csv = Path(args.batch_csv)
    if not batch_csv.exists():
        raise FileNotFoundError(f"Batch CSV not found: {batch_csv}")

    results_path = Path(args.results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    runs = []
    invalid_runs: List[Dict[str, str]] = []
    started = time.time()

    with batch_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row_count = 0
        for idx, row in enumerate(reader, start=1):
            if not _is_enabled(row.get("enabled")):
                continue
            row_count += 1
            if args.limit is not None and row_count > args.limit:
                break

            nickname = (row.get("nickname") or "").strip()
            city_name = (row.get("city_name") or "").strip()
            lat = (row.get("center_lat") or "").strip()
            lng = (row.get("center_lng") or "").strip()
            square_km = (row.get("square_km") or "").strip() or str(args.square_km)

            resume_hit = _resume_cache_hit(row, args, results_path)
            if resume_hit.get("hit"):
                diagnostics = resume_hit.get("diagnostics", {})
                print(
                    f"[{idx}] SKIP (resume): {(resume_hit.get('reason') or '').strip()}"
                )
                runs.append(
                    {
                        "index": idx,
                        "nickname": nickname,
                        "city_name": city_name,
                        "center_lat": lat,
                        "center_lng": lng,
                        "square_km": square_km,
                        "status": "cached",
                        "reason": str(resume_hit.get("reason", "")),
                        "exit_code": "",
                        "duration_sec": "0",
                        "error_tail": "",
                        "valid_rows": diagnostics.get("valid_rows", ""),
                        "total_rows": diagnostics.get("total_rows", ""),
                        "total_population": diagnostics.get("total_population", ""),
                        "positive_population_ratio": diagnostics.get(
                            "positive_population_ratio", ""
                        ),
                        "accessibility_non_null_ratio": diagnostics.get(
                            "accessibility_non_null_ratio", ""
                        ),
                        "geometry_non_empty_ratio": diagnostics.get(
                            "geometry_non_empty_ratio", ""
                        ),
                    }
                )
                continue

            try:
                cmd = _build_command(row, args)
            except Exception as exc:
                runs.append(
                    {
                        "index": idx,
                        "nickname": nickname,
                        "city_name": city_name,
                        "center_lat": lat,
                        "center_lng": lng,
                        "square_km": square_km,
                        "status": "invalid",
                        "reason": f"row validation failed: {exc}",
                        "exit_code": "",
                        "duration_sec": "0",
                        "error_tail": str(exc),
                        "valid_rows": "",
                        "total_rows": "",
                        "total_population": "",
                        "positive_population_ratio": "",
                        "accessibility_non_null_ratio": "",
                        "geometry_non_empty_ratio": "",
                    }
                )
                invalid_runs.append(
                    {"center_lat": lat, "center_lng": lng, "square_km": square_km}
                )
                if args.fail_fast:
                    break
                continue

            print(f"\n[{idx}] Running: {' '.join(cmd)}")
            if args.dry_run:
                runs.append(
                    {
                        "index": idx,
                        "nickname": nickname,
                        "city_name": city_name,
                        "center_lat": lat,
                        "center_lng": lng,
                        "square_km": square_km,
                        "status": "dry_run",
                        "reason": "",
                        "exit_code": "",
                        "duration_sec": "0",
                        "error_tail": "",
                        "valid_rows": "",
                        "total_rows": "",
                        "total_population": "",
                        "positive_population_ratio": "",
                        "accessibility_non_null_ratio": "",
                        "geometry_non_empty_ratio": "",
                    }
                )
                continue

            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            dt = round(time.time() - t0, 2)

            tail = "\n".join((proc.stderr or "").splitlines()[-20:])
            quality = {"status": "", "reason": "", "diagnostics": {}}
            if proc.returncode == 0:
                city_dir = results_path / city_filename_for_row(row)
                quality = assess_run_quality(city_dir)

            status = "failed"
            reason = ""
            if proc.returncode == 0:
                status = quality.get("status", "success")
                reason = str(quality.get("reason", ""))
            diagnostics = (
                quality.get("diagnostics", {}) if isinstance(quality, dict) else {}
            )

            if status == "invalid":
                invalid_runs.append(
                    {"center_lat": lat, "center_lng": lng, "square_km": square_km}
                )

            runs.append(
                {
                    "index": idx,
                    "nickname": nickname,
                    "city_name": city_name,
                    "center_lat": lat,
                    "center_lng": lng,
                    "square_km": square_km,
                    "status": status,
                    "reason": reason,
                    "exit_code": str(proc.returncode),
                    "duration_sec": str(dt),
                    "error_tail": tail,
                    "valid_rows": diagnostics.get("valid_rows", ""),
                    "total_rows": diagnostics.get("total_rows", ""),
                    "total_population": diagnostics.get("total_population", ""),
                    "positive_population_ratio": diagnostics.get(
                        "positive_population_ratio", ""
                    ),
                    "accessibility_non_null_ratio": diagnostics.get(
                        "accessibility_non_null_ratio", ""
                    ),
                    "geometry_non_empty_ratio": diagnostics.get(
                        "geometry_non_empty_ratio", ""
                    ),
                }
            )

            if proc.returncode != 0:
                print(f"[{idx}] FAILED (exit={proc.returncode})")
                if tail:
                    print(tail)
                if args.fail_fast:
                    break
            elif status == "invalid":
                print(f"[{idx}] INVALID ({dt}s): {reason}")
                if args.fail_fast:
                    break
            else:
                print(f"[{idx}] OK ({dt}s)")

    report_path = results_path / "multi_batch_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "index",
            "nickname",
            "city_name",
            "center_lat",
            "center_lng",
            "square_km",
            "status",
            "reason",
            "exit_code",
            "duration_sec",
            "error_tail",
            "valid_rows",
            "total_rows",
            "total_population",
            "positive_population_ratio",
            "accessibility_non_null_ratio",
            "geometry_non_empty_ratio",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(runs)

    ranking_path = None
    pruned_rows = 0
    if not args.dry_run:
        summary_path = results_path / "comparison_summary.csv"
        pruned_rows = prune_invalid_rows_from_summary(summary_path, invalid_runs)
        ranking_path = _regen_overall_ranking(results_path)

    metadata_path = _write_metadata(results_path, args, runs, started, time.time())
    total_time = round(time.time() - started, 2)
    ok = sum(1 for r in runs if r["status"] == "success")
    cached = sum(1 for r in runs if r["status"] == "cached")
    invalid = sum(1 for r in runs if r["status"] == "invalid")
    failed = sum(1 for r in runs if r["status"] == "failed")

    print("\n=== Multi-run summary ===")
    print(f"Total rows processed: {len(runs)}")
    print(f"Succeeded: {ok}")
    print(f"Cached (resumed): {cached}")
    print(f"Invalid: {invalid}")
    print(f"Failed: {failed}")
    if pruned_rows:
        print(f"Pruned invalid rows from comparison summary: {pruned_rows}")
    print(f"Report: {report_path}")
    print(f"Metadata: {metadata_path}")
    if ranking_path is not None:
        print(f"Ranking: {ranking_path}")
    print(f"Elapsed: {total_time}s")

    return 1 if (failed > 0 or invalid > 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
