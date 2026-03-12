"""
Ligue 1 Data Update Pipeline
Re-fetches season data, upcoming fixtures, odds, and rebuilds features.
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fetcher import (
    FIXTURES_DIR,
    LEAGUE_ID,
    RAW_DIR,
    STATS_DIR,
    api_get,
    fetch_fixture_stats,
)

UPCOMING_PATH = RAW_DIR / "fixtures" / "upcoming.json"
ODDS_DIR = RAW_DIR / "odds"
UTC_MINUS_6 = timezone(timedelta(hours=-6))


def refresh_season_fixtures(season: int) -> list[dict]:
    """Delete cached fixtures and re-fetch finished matches from API."""
    cache_path = FIXTURES_DIR / f"ligue1_{season}_fixtures.json"
    if cache_path.exists():
        print(f"  Deleting cached {cache_path.name}...")
        cache_path.unlink()

    print(f"  Fetching season {season} finished fixtures...")
    data = api_get("fixtures", {"league": LEAGUE_ID, "season": season, "status": "FT"})
    fixtures = data["response"]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(fixtures, f, indent=2)

    print(f"  Downloaded {len(fixtures)} finished fixtures → {cache_path.name}")
    return fixtures


def fetch_new_stats(season: int):
    """Fetch stats for any fixtures missing from cache."""
    fixtures_path = FIXTURES_DIR / f"ligue1_{season}_fixtures.json"
    if not fixtures_path.exists():
        print(f"  No fixtures file for season {season}")
        return

    with open(fixtures_path) as f:
        fixtures = json.load(f)

    fixture_ids = [fx["fixture"]["id"] for fx in fixtures]
    season_stats_dir = STATS_DIR / str(season)
    cached = sum(1 for fid in fixture_ids if (season_stats_dir / f"{fid}.json").exists())
    to_fetch = len(fixture_ids) - cached

    if to_fetch == 0:
        print(f"  All {len(fixture_ids)} stats already cached for season {season}")
        return

    print(f"  Fetching {to_fetch} new stats for season {season}...")
    fetched = 0
    for fid in fixture_ids:
        result = fetch_fixture_stats(fid, season)
        if result is not None:
            fetched += 1
            if fetched % 10 == 0:
                print(f"    {fetched}/{to_fetch} fetched...")

    print(f"  Done: {fetched} new stats fetched for season {season}")


def fetch_upcoming_fixtures() -> list[dict]:
    """Fetch next not-started fixtures for Ligue 1."""
    print("  Fetching upcoming fixtures...")
    data = api_get("fixtures", {
        "league": LEAGUE_ID,
        "season": 2025,
        "status": "NS",
        "next": 10,
    })
    fixtures = data["response"]

    if not fixtures:
        # Fallback: try without next parameter
        print("  No results with next=10, trying status=NS only...")
        data = api_get("fixtures", {
            "league": LEAGUE_ID,
            "season": 2025,
            "status": "NS",
        })
        fixtures = data["response"]
        # Take only next 10 by date
        fixtures.sort(key=lambda fx: fx["fixture"]["date"])
        fixtures = fixtures[:10]

    # Add UTC-6 display dates
    for fx in fixtures:
        raw_date = fx["fixture"]["date"]
        try:
            utc_date = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            mx_date = utc_date.astimezone(UTC_MINUS_6)
            fx["fixture"]["date_utc6"] = mx_date.isoformat()
            fx["fixture"]["date_display"] = mx_date.strftime("%a %d %b, %H:%M")
        except Exception:
            fx["fixture"]["date_utc6"] = raw_date
            fx["fixture"]["date_display"] = raw_date

    UPCOMING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(UPCOMING_PATH, "w") as f:
        json.dump(fixtures, f, indent=2)

    print(f"  Downloaded {len(fixtures)} upcoming fixtures → {UPCOMING_PATH.name}")
    return fixtures


def fetch_upcoming_odds(fixtures: list[dict]) -> dict:
    """Fetch 1X2 odds for each upcoming fixture."""
    print("  Fetching odds for upcoming fixtures...")
    ODDS_DIR.mkdir(parents=True, exist_ok=True)

    odds_data = {}
    for fx in fixtures:
        fid = fx["fixture"]["id"]
        home = fx["teams"]["home"]["name"]
        away = fx["teams"]["away"]["name"]
        print(f"    {home} vs {away} (#{fid})...", end=" ", flush=True)

        try:
            data = api_get("odds", {"fixture": fid})
            if data["response"]:
                found = False
                for resp_item in data["response"]:
                    for bookmaker in resp_item.get("bookmakers", []):
                        for bet in bookmaker.get("bets", []):
                            if bet["name"] == "Match Winner":
                                odds_1x2 = {}
                                for val in bet["values"]:
                                    if val["value"] == "Home":
                                        odds_1x2["home"] = float(val["odd"])
                                    elif val["value"] == "Draw":
                                        odds_1x2["draw"] = float(val["odd"])
                                    elif val["value"] == "Away":
                                        odds_1x2["away"] = float(val["odd"])
                                if len(odds_1x2) == 3:
                                    odds_data[str(fid)] = {
                                        "bookmaker": bookmaker["name"],
                                        **odds_1x2,
                                    }
                                    print(f"OK ({bookmaker['name']})")
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        break
                if not found:
                    print("no 1X2 odds")
            else:
                print("no odds available")
        except Exception as e:
            print(f"error: {e}")

        time.sleep(0.15)

    odds_path = ODDS_DIR / "upcoming_odds.json"
    with open(odds_path, "w") as f:
        json.dump(odds_data, f, indent=2)

    print(f"  Saved odds for {len(odds_data)}/{len(fixtures)} fixtures")
    return odds_data


def run_full_update():
    """Run the complete data update pipeline."""
    print("=" * 60)
    print("  LIGUE 1 PREDICTOR — FULL DATA UPDATE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Step 1: Re-fetch season 2025 finished fixtures
    print("\n[1/6] Re-fetching season 2025 fixtures (FT)...")
    refresh_season_fixtures(2025)

    # Step 2: Fetch stats for new fixtures
    print("\n[2/6] Fetching stats for new fixtures...")
    fetch_new_stats(2025)

    # Step 3: Rebuild matches.parquet
    print("\n[3/6] Rebuilding matches.parquet...")
    from src.data.preprocessor import build_matches_df
    build_matches_df()

    # Step 4: Rebuild features.parquet
    print("\n[4/6] Rebuilding features.parquet...")
    from src.features.engineer import build_features
    build_features()

    # Step 5: Fetch upcoming fixtures
    print("\n[5/6] Fetching upcoming fixtures...")
    upcoming = fetch_upcoming_fixtures()

    # Step 6: Fetch odds
    print("\n[6/6] Fetching odds...")
    fetch_upcoming_odds(upcoming)

    print("\n" + "=" * 60)
    print("  UPDATE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_full_update()
