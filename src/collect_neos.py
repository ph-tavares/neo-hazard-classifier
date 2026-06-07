"""Collect NEO data from NASA NeoWs `browse` into a CSV snapshot.

Usage:
  NASA_API_KEY=... python -m src.collect_neos --census --max-pages 1600
  NASA_API_KEY=... python -m src.collect_neos --pages 1000 --out data/neos_raw.csv

The key is read from the NASA_API_KEY environment variable and is never logged.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd
import requests

from src.neo_features import extract_features

BASE_URL = "https://api.nasa.gov/neo/rest/v1/neo/browse"


def fetch_page(page: int, size: int, api_key: str, max_retries: int = 5) -> dict:
    """GET one browse page, retrying on 429/5xx with linear backoff."""
    params = {"page": page, "size": size, "api_key": api_key}
    for attempt in range(max_retries):
        resp = requests.get(BASE_URL, params=params, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504):
            wait = 5 * (attempt + 1)
            print(f"  page {page}: HTTP {resp.status_code}, retry in {wait}s", flush=True)
            time.sleep(wait)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"page {page}: exhausted retries")


def rows_from_page(payload: dict) -> list[dict]:
    """Extract usable feature rows from one page payload (skips None)."""
    rows = []
    for neo in payload.get("near_earth_objects", []):
        row = extract_features(neo)
        if row is not None:
            rows.append(row)
    return rows


def census(api_key: str, max_pages: int, size: int, sleep: float) -> None:
    """Print per-page PHA count and close_approach_data coverage; stop early
    after several consecutive pages with zero coverage."""
    zero_streak = 0
    for page in range(max_pages):
        payload = fetch_page(page, size, api_key)
        neos = payload.get("near_earth_objects", [])
        n_pha = sum(1 for n in neos if n.get("is_potentially_hazardous_asteroid"))
        n_cad = sum(1 for n in neos if n.get("close_approach_data"))
        print(f"page {page:>4}: PHA {n_pha}/{len(neos)} | CAD {n_cad}/{len(neos)}", flush=True)
        zero_streak = zero_streak + 1 if n_cad == 0 else 0
        if zero_streak >= 5:
            print(f"stopping: 5 consecutive pages with no close_approach_data (last page {page})")
            return
        time.sleep(sleep)


def collect(api_key: str, pages: int, size: int, sleep: float, out: str) -> None:
    all_rows: list[dict] = []
    for page in range(pages):
        payload = fetch_page(page, size, api_key)
        rows = rows_from_page(payload)
        all_rows.extend(rows)
        if page % 50 == 0:
            print(f"page {page}/{pages}: cumulative usable rows = {len(all_rows)}", flush=True)
        time.sleep(sleep)
    df = pd.DataFrame(all_rows)
    df.to_csv(out, index=False)
    n_pha = int(df["is_potentially_hazardous_asteroid"].sum())
    print(f"\nwrote {len(df)} rows to {out} | PHA = {n_pha} ({n_pha / len(df):.1%})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect NEO data from NeoWs browse.")
    parser.add_argument("--census", action="store_true", help="probe pages instead of collecting")
    parser.add_argument("--max-pages", type=int, default=1600, help="census page cap")
    parser.add_argument("--pages", type=int, default=1000, help="pages to collect")
    parser.add_argument("--size", type=int, default=20, help="objects per page (max 20)")
    parser.add_argument("--sleep", type=float, default=0.6, help="seconds between requests")
    parser.add_argument("--out", default="data/neos_raw.csv")
    args = parser.parse_args(argv)

    api_key = os.environ.get("NASA_API_KEY")
    if not api_key:
        print("ERROR: set NASA_API_KEY in the environment", file=sys.stderr)
        return 1

    if args.census:
        census(api_key, args.max_pages, args.size, args.sleep)
    else:
        collect(api_key, args.pages, args.size, args.sleep, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
