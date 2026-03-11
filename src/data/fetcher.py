"""
Ligue 1 Data Fetcher
Descarga fixtures, estadísticas y player stats de API-Football con checkpointing.
"""

import json
import os
import time
from pathlib import Path

import requests
from tqdm import tqdm

BASE_URL = "https://v3.football.api-sports.io"
LEAGUE_ID = 61
SEASONS = [2021, 2022, 2023, 2024, 2025]
SLEEP_SECONDS = 0.1

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FIXTURES_DIR = RAW_DIR / "fixtures"
STATS_DIR = RAW_DIR / "team_stats"
PLAYERS_DIR = RAW_DIR / "player_stats"


def _get_api_key() -> str:
    """Get API key: st.secrets (production) → os.environ → .env file."""
    # 1. Streamlit secrets (production)
    try:
        import streamlit as st
        return st.secrets["API_FOOTBALL_KEY"]
    except Exception:
        pass

    # 2. Environment variable
    key = os.environ.get("API_FOOTBALL_KEY")
    if key:
        return key

    # 3. .env file
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("API_FOOTBALL_KEY="):
                return line.split("=", 1)[1].strip()

    raise RuntimeError(
        "API key not found. Set API_FOOTBALL_KEY in "
        ".streamlit/secrets.toml, environment, or .env file."
    )


HEADERS = {"x-apisports-key": _get_api_key()}

# Global request counter
_request_count = 0


def api_get(endpoint: str, params: dict) -> dict:
    """Single API call with error handling."""
    global _request_count
    resp = requests.get(f"{BASE_URL}/{endpoint}", headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    _request_count += 1
    data = resp.json()
    if data.get("errors"):
        raise RuntimeError(f"API error: {data['errors']}")
    return data


def fetch_season_fixtures(season: int) -> list[dict]:
    """Fetch all finished fixtures for a season. Uses checkpoint."""
    out_path = FIXTURES_DIR / f"ligue1_{season}_fixtures.json"
    if out_path.exists():
        print(f"  [CACHE] {out_path.name} ya existe, skip API call")
        with open(out_path) as f:
            return json.load(f)

    data = api_get("fixtures", {"league": LEAGUE_ID, "season": season, "status": "FT"})
    fixtures = data["response"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(fixtures, f, indent=2)

    print(f"  [API] Descargados {len(fixtures)} fixtures → {out_path.name}")
    return fixtures


def fetch_fixture_stats(fixture_id: int, season: int) -> dict | None:
    """Fetch statistics for a single fixture. Uses checkpoint."""
    season_dir = STATS_DIR / str(season)
    out_path = season_dir / f"{fixture_id}.json"
    if out_path.exists():
        return None

    time.sleep(SLEEP_SECONDS)
    data = api_get("fixtures/statistics", {"fixture": fixture_id})
    stats = data["response"]

    season_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def fetch_fixture_players(fixture_id: int, season: int) -> dict | None:
    """Fetch player stats for a single fixture. Uses checkpoint."""
    season_dir = PLAYERS_DIR / str(season)
    out_path = season_dir / f"{fixture_id}.json"
    if out_path.exists():
        return None

    time.sleep(SLEEP_SECONDS)
    data = api_get("fixtures/players", {"fixture": fixture_id})
    players = data["response"]

    season_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(players, f, indent=2)

    return players


def fetch_season(season: int, include_players: bool = False) -> dict:
    """Fetch fixtures + stats (+ optionally players) for a season."""
    print(f"\n{'='*50}")
    print(f"  SEASON {season}")
    print(f"{'='*50}")

    # Step 1: Fixtures
    fixtures = fetch_season_fixtures(season)
    fixture_ids = [f["fixture"]["id"] for f in fixtures]

    # Step 2: Stats per fixture
    season_stats_dir = STATS_DIR / str(season)
    cached = sum(1 for fid in fixture_ids if (season_stats_dir / f"{fid}.json").exists())
    to_fetch = len(fixture_ids) - cached

    if to_fetch == 0:
        print(f"  [CACHE] {len(fixture_ids)} stats ya cacheadas, skip")
    else:
        print(f"  Stats: {cached} cached, {to_fetch} por descargar")
        for fid in tqdm(fixture_ids, desc=f"  Stats {season}", unit="match"):
            fetch_fixture_stats(fid, season)

    # Step 3: Player stats (optional)
    if include_players:
        season_players_dir = PLAYERS_DIR / str(season)
        cached_p = sum(1 for fid in fixture_ids if (season_players_dir / f"{fid}.json").exists())
        to_fetch_p = len(fixture_ids) - cached_p

        if to_fetch_p == 0:
            print(f"  [CACHE] {len(fixture_ids)} player stats ya cacheadas, skip")
        else:
            print(f"  Players: {cached_p} cached, {to_fetch_p} por descargar")
            for fid in tqdm(fixture_ids, desc=f"  Players {season}", unit="match"):
                fetch_fixture_players(fid, season)

    return {"season": season, "fixtures": len(fixtures), "stats": len(fixture_ids)}


def fetch_all():
    """Fetch all seasons."""
    global _request_count
    _request_count = 0

    print("Ligue 1 Fetcher - API-Football")
    print(f"Seasons: {SEASONS}")
    print(f"Sleep: {SLEEP_SECONDS}s entre requests")

    results = []
    for season in SEASONS:
        info = fetch_season(season)
        results.append(info)

    print(f"\n{'='*50}")
    print("  RESUMEN")
    print(f"{'='*50}")
    total = 0
    for r in results:
        print(f"  Season {r['season']}: {r['fixtures']} fixtures")
        total += r["fixtures"]
    print(f"  TOTAL: {total} fixtures")
    print(f"  API requests: {_request_count}")
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        for s in [int(x) for x in sys.argv[1:]]:
            fetch_season(s)
    else:
        fetch_all()
