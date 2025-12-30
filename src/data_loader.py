"""
Data loader for tennis match data.

Handles loading data from:
- Jeff Sackmann's GitHub repositories (ATP/WTA historical data)
- Local CSV files
"""

import pandas as pd
import requests
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import os


# Jeff Sackmann's GitHub repositories
SACKMANN_ATP_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
SACKMANN_WTA_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master"


def download_sackmann_data(
    start_year: int = 2010,
    end_year: int = 2024,
    tour: str = 'atp',
    data_dir: str = 'data'
) -> List[Path]:
    """
    Download historical match data from Jeff Sackmann's GitHub.

    Args:
        start_year: First year to download
        end_year: Last year to download
        tour: 'atp' or 'wta'
        data_dir: Directory to save files

    Returns:
        List of downloaded file paths
    """
    base_url = SACKMANN_ATP_BASE if tour.lower() == 'atp' else SACKMANN_WTA_BASE
    prefix = 'atp' if tour.lower() == 'atp' else 'wta'

    # Create data directory
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    downloaded = []

    for year in tqdm(range(start_year, end_year + 1), desc=f"Downloading {tour.upper()} data"):
        filename = f"{prefix}_matches_{year}.csv"
        url = f"{base_url}/{filename}"
        local_path = data_path / filename

        # Skip if already downloaded
        if local_path.exists():
            downloaded.append(local_path)
            continue

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            with open(local_path, 'wb') as f:
                f.write(response.content)

            downloaded.append(local_path)
        except requests.RequestException as e:
            print(f"Failed to download {filename}: {e}")

    return downloaded


def download_player_rankings(tour: str = 'atp', data_dir: str = 'data') -> Optional[Path]:
    """Download current ATP/WTA rankings."""
    base_url = SACKMANN_ATP_BASE if tour.lower() == 'atp' else SACKMANN_WTA_BASE
    prefix = 'atp' if tour.lower() == 'atp' else 'wta'

    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    filename = f"{prefix}_rankings_current.csv"
    url = f"{base_url}/{filename}"
    local_path = data_path / filename

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(local_path, 'wb') as f:
            f.write(response.content)

        return local_path
    except requests.RequestException as e:
        print(f"Failed to download rankings: {e}")
        return None


def load_matches(
    data_dir: str = 'data',
    start_year: int = 2010,
    end_year: int = 2024,
    tour: str = 'atp'
) -> pd.DataFrame:
    """
    Load match data from local CSV files.

    Returns combined DataFrame with all matches.
    """
    prefix = 'atp' if tour.lower() == 'atp' else 'wta'
    data_path = Path(data_dir)

    all_matches = []

    for year in range(start_year, end_year + 1):
        filepath = data_path / f"{prefix}_matches_{year}.csv"
        if filepath.exists():
            df = pd.read_csv(filepath, low_memory=False)
            all_matches.append(df)

    if not all_matches:
        return pd.DataFrame()

    combined = pd.concat(all_matches, ignore_index=True)

    # Clean and standardize columns
    combined = clean_match_data(combined)

    return combined


def clean_match_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize match data."""
    # Ensure required columns exist
    required_cols = ['winner_name', 'loser_name', 'surface', 'tourney_date']

    for col in required_cols:
        if col not in df.columns:
            if col == 'surface':
                df[col] = 'Hard'  # Default
            elif col == 'tourney_date':
                df[col] = 0
            else:
                df[col] = ''

    # Convert tourney_date to datetime if numeric
    if df['tourney_date'].dtype in ['int64', 'float64']:
        df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')

    # Fill missing surfaces
    df['surface'] = df['surface'].fillna('Hard')

    # Remove walkovers and retirements for cleaner data (optional)
    if 'score' in df.columns:
        df = df[~df['score'].str.contains('W/O|RET|DEF', na=False, case=False)]

    return df


def get_recent_matches(
    df: pd.DataFrame,
    player: str,
    n: int = 10,
    surface: Optional[str] = None
) -> pd.DataFrame:
    """Get a player's most recent matches."""
    player_matches = df[
        (df['winner_name'] == player) | (df['loser_name'] == player)
    ].copy()

    if surface:
        player_matches = player_matches[
            player_matches['surface'].str.lower() == surface.lower()
        ]

    if 'tourney_date' in player_matches.columns:
        player_matches = player_matches.sort_values('tourney_date', ascending=False)

    return player_matches.head(n)


def get_head_to_head(
    df: pd.DataFrame,
    player_a: str,
    player_b: str
) -> dict:
    """Get head-to-head record between two players."""
    h2h = df[
        ((df['winner_name'] == player_a) & (df['loser_name'] == player_b)) |
        ((df['winner_name'] == player_b) & (df['loser_name'] == player_a))
    ]

    wins_a = len(h2h[h2h['winner_name'] == player_a])
    wins_b = len(h2h[h2h['winner_name'] == player_b])

    return {
        player_a: wins_a,
        player_b: wins_b,
        'total_matches': wins_a + wins_b,
        'matches': h2h
    }


def download_odds_data(
    start_year: int = 2010,
    end_year: int = 2024,
    data_dir: str = 'data'
) -> List[Path]:
    """
    Download historical odds data from tennis-data.co.uk.

    This source includes odds from multiple bookmakers:
    - B365 (Bet365)
    - PS (Pinnacle)
    - Max (Best available)
    - Avg (Average)
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    downloaded = []

    for year in tqdm(range(start_year, end_year + 1), desc="Downloading odds data"):
        local_path = data_path / f"odds_{year}.xlsx"

        if local_path.exists():
            downloaded.append(local_path)
            continue

        # tennis-data.co.uk URL format
        url = f"http://www.tennis-data.co.uk/{year}/{year}.xlsx"

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            with open(local_path, 'wb') as f:
                f.write(response.content)

            downloaded.append(local_path)
        except requests.RequestException as e:
            # Try alternative XLS format for older years
            try:
                url_xls = f"http://www.tennis-data.co.uk/{year}/{year}.xls"
                response = requests.get(url_xls, timeout=60)
                response.raise_for_status()

                local_path_xls = data_path / f"odds_{year}.xls"
                with open(local_path_xls, 'wb') as f:
                    f.write(response.content)

                downloaded.append(local_path_xls)
            except:
                print(f"Failed to download odds for {year}")

    return downloaded


def load_odds_data(
    data_dir: str = 'data',
    start_year: int = 2010,
    end_year: int = 2024
) -> pd.DataFrame:
    """
    Load odds data from tennis-data.co.uk Excel files.

    Returns DataFrame with match odds including:
    - Winner/Loser names
    - Tournament info
    - Surface
    - Odds from multiple bookmakers (B365W/L, PSW/L, MaxW/L, AvgW/L)
    """
    data_path = Path(data_dir)
    all_data = []

    for year in range(start_year, end_year + 1):
        # Try xlsx first, then xls
        for ext in ['.xlsx', '.xls']:
            filepath = data_path / f"odds_{year}{ext}"
            if filepath.exists():
                try:
                    df = pd.read_excel(filepath, engine='openpyxl' if ext == '.xlsx' else 'xlrd')
                    df['year'] = year
                    all_data.append(df)
                    break
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)

    # Standardize column names
    combined = standardize_odds_columns(combined)

    return combined


def standardize_odds_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize odds data column names to match our format."""
    # Common column mappings
    column_map = {
        'Winner': 'winner_name',
        'Loser': 'loser_name',
        'Surface': 'surface',
        'Tournament': 'tourney_name',
        'Date': 'tourney_date',
        'Round': 'round',
        'Best of': 'best_of',
        'WRank': 'winner_rank',
        'LRank': 'loser_rank',
        # Odds columns - keep as is but ensure consistency
        'B365W': 'B365W',
        'B365L': 'B365L',
        'PSW': 'PSW',
        'PSL': 'PSL',
        'MaxW': 'MaxW',
        'MaxL': 'MaxL',
        'AvgW': 'AvgW',
        'AvgL': 'AvgL',
    }

    # Rename columns that exist
    for old, new in column_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})

    # Ensure date is datetime
    if 'tourney_date' in df.columns:
        df['tourney_date'] = pd.to_datetime(df['tourney_date'], errors='coerce')

    return df


def load_matches_with_odds(
    data_dir: str = 'data',
    start_year: int = 2010,
    end_year: int = 2024,
    download: bool = False
) -> pd.DataFrame:
    """
    Load matches with odds data for backtesting.

    This uses tennis-data.co.uk which has both match results and odds.
    """
    if download:
        download_odds_data(start_year, end_year, data_dir)

    odds_data = load_odds_data(data_dir, start_year, end_year)

    if len(odds_data) == 0:
        print("No odds data found. Run with download=True to fetch data.")
        return pd.DataFrame()

    # Filter out rows without essential data
    required = ['winner_name', 'loser_name', 'AvgW', 'AvgL']
    for col in required:
        if col in odds_data.columns:
            odds_data = odds_data.dropna(subset=[col])

    return odds_data


if __name__ == "__main__":
    # Example usage
    print("Downloading ATP match data...")
    files = download_sackmann_data(start_year=2020, end_year=2024, tour='atp')
    print(f"Downloaded {len(files)} files")

    print("\nLoading matches...")
    matches = load_matches(start_year=2020, end_year=2024)
    print(f"Loaded {len(matches)} matches")

    if len(matches) > 0:
        print(f"\nColumns: {matches.columns.tolist()}")
        print(f"\nSample match:")
        print(matches[['tourney_name', 'winner_name', 'loser_name', 'surface', 'score']].head(1))
