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
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tsunami_batch.py for multiple rows from CSV")
    parser.add_argument("--batch-csv", required=True, help="CSV file with center_lat/center_lng rows")
    parser.add_argument("--results-path", default="tsunami_study", help="Shared output root")
    parser.add_argument("--timeout", type=int, default=1800, help="Default per-run notebook timeout")
    parser.add_argument("--square-km", type=float, default=1.0, help="Default square size when missing in CSV")
    parser.add_argument("--overwrite", action="store_true", help="Pass --overwrite to each single run")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N enabled rows")
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately on first failed row")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--python", default=sys.executable, help="Python executable for child runs")
    parser.add_argument("--script-path", default="examples/tsunami_batch.py", help="Path to single-run script")
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
                "normalization_elevation_danger_m": row.get("normalization_elevation_danger_m", ""),
                "normalization_elevation_safe_m": row.get("normalization_elevation_safe_m", ""),
                "normalization_population_cell_cap": row.get("normalization_population_cell_cap", ""),
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
            lat = (row.get("center_lat") or "").strip()
            lng = (row.get("center_lng") or "").strip()

            try:
                cmd = _build_command(row, args)
            except Exception as exc:
                runs.append(
                    {
                        "index": idx,
                        "nickname": nickname,
                        "center_lat": lat,
                        "center_lng": lng,
                        "status": "failed",
                        "exit_code": "",
                        "duration_sec": "0",
                        "error_tail": str(exc),
                    }
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
                        "center_lat": lat,
                        "center_lng": lng,
                        "status": "dry_run",
                        "exit_code": "",
                        "duration_sec": "0",
                        "error_tail": "",
                    }
                )
                continue

            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            dt = round(time.time() - t0, 2)

            status = "success" if proc.returncode == 0 else "failed"
            tail = "\n".join((proc.stderr or "").splitlines()[-20:])
            runs.append(
                {
                    "index": idx,
                    "nickname": nickname,
                    "center_lat": lat,
                    "center_lng": lng,
                    "status": status,
                    "exit_code": str(proc.returncode),
                    "duration_sec": str(dt),
                    "error_tail": tail,
                }
            )

            if proc.returncode != 0:
                print(f"[{idx}] FAILED (exit={proc.returncode})")
                if tail:
                    print(tail)
                if args.fail_fast:
                    break
            else:
                print(f"[{idx}] OK ({dt}s)")

    report_path = results_path / "multi_batch_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "index",
            "nickname",
            "center_lat",
            "center_lng",
            "status",
            "exit_code",
            "duration_sec",
            "error_tail",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(runs)

    ranking_path = None
    if not args.dry_run:
        ranking_path = _regen_overall_ranking(results_path)

    total_time = round(time.time() - started, 2)
    ok = sum(1 for r in runs if r["status"] == "success")
    failed = sum(1 for r in runs if r["status"] == "failed")

    print("\n=== Multi-run summary ===")
    print(f"Total rows processed: {len(runs)}")
    print(f"Succeeded: {ok}")
    print(f"Failed: {failed}")
    print(f"Report: {report_path}")
    if ranking_path is not None:
        print(f"Ranking: {ranking_path}")
    print(f"Elapsed: {total_time}s")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
