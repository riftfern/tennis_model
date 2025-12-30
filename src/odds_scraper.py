"""
Odds scraper for tennis betting.

Scrapes odds from various sources to compare with model predictions.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import re
import time


@dataclass
class Match:
    """Represents an upcoming tennis match."""
    player_a: str
    player_b: str
    tournament: str
    surface: str
    start_time: Optional[datetime]
    odds_a: Optional[float] = None
    odds_b: Optional[float] = None
    bookmaker: Optional[str] = None


class OddsScraper:
    """Base class for odds scrapers."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def get_upcoming_matches(self) -> List[Match]:
        """Override in subclasses."""
        raise NotImplementedError


class OddsAPIClient:
    """
    Client for The Odds API (https://the-odds-api.com/).

    Free tier: 500 requests/month
    Provides odds from multiple bookmakers.
    """

    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()

    def get_active_tennis_sports(self) -> List[Dict]:
        """Get list of currently active tennis tournaments."""
        url = f"{self.BASE_URL}/sports"
        params = {'apiKey': self.api_key}

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            sports = response.json()
            return [s for s in sports if 'tennis' in s['key'].lower() and s.get('active', False)]
        except requests.RequestException as e:
            print(f"Error fetching sports: {e}")
            return []

    def get_all_tennis_sports(self) -> List[Dict]:
        """Get list of all tennis tournaments (including inactive)."""
        url = f"{self.BASE_URL}/sports"
        params = {'apiKey': self.api_key, 'all': 'true'}

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            sports = response.json()
            return [s for s in sports if 'tennis' in s['key'].lower()]
        except requests.RequestException as e:
            print(f"Error fetching sports: {e}")
            return []

    def get_tennis_odds(
        self,
        regions: str = 'us,uk,eu',
        markets: str = 'h2h',
        odds_format: str = 'decimal'
    ) -> List[Dict]:
        """
        Get upcoming tennis match odds from all active tournaments.

        Args:
            regions: Comma-separated regions (us, uk, eu, au)
            markets: Market types (h2h for match winner)
            odds_format: 'decimal', 'american', or 'fractional'
        """
        # Get currently active tennis sports
        active_sports = self.get_active_tennis_sports()

        if not active_sports:
            print("No active tennis tournaments found.")
            print("Tennis is likely in off-season. Check back closer to tournament dates.")
            all_tennis = self.get_all_tennis_sports()
            if all_tennis:
                print(f"\nAvailable tournaments (currently inactive):")
                for sport in all_tennis[:10]:
                    print(f"  - {sport['title']}")
            return []

        print(f"Found {len(active_sports)} active tennis tournament(s):")
        for sport in active_sports:
            print(f"  - {sport['title']}")

        all_matches = []

        for sport in active_sports:
            sport_key = sport['key']
            url = f"{self.BASE_URL}/sports/{sport_key}/odds"
            params = {
                'apiKey': self.api_key,
                'regions': regions,
                'markets': markets,
                'oddsFormat': odds_format
            }

            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                matches = response.json()
                all_matches.extend(matches)
            except requests.RequestException as e:
                print(f"Error fetching {sport_key}: {e}")

        return all_matches

    def parse_odds_response(self, matches: List[Dict]) -> pd.DataFrame:
        """Parse API response into DataFrame."""
        rows = []

        for match in matches:
            match_id = match.get('id')
            sport = match.get('sport_key')
            commence_time = match.get('commence_time')
            home_team = match.get('home_team')
            away_team = match.get('away_team')

            # Get best odds from all bookmakers
            best_odds = {}

            for bookmaker in match.get('bookmakers', []):
                book_name = bookmaker.get('key')

                for market in bookmaker.get('markets', []):
                    if market.get('key') == 'h2h':
                        for outcome in market.get('outcomes', []):
                            player = outcome.get('name')
                            price = outcome.get('price')

                            if player not in best_odds or price > best_odds[player]['odds']:
                                best_odds[player] = {
                                    'odds': price,
                                    'bookmaker': book_name
                                }

            if len(best_odds) == 2:
                players = list(best_odds.keys())
                rows.append({
                    'match_id': match_id,
                    'sport': sport,
                    'commence_time': commence_time,
                    'player_a': players[0],
                    'player_b': players[1],
                    'odds_a': best_odds[players[0]]['odds'],
                    'odds_b': best_odds[players[1]]['odds'],
                    'book_a': best_odds[players[0]]['bookmaker'],
                    'book_b': best_odds[players[1]]['bookmaker']
                })

        return pd.DataFrame(rows)


def american_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal."""
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to American."""
    if decimal_odds >= 2.0:
        return int((decimal_odds - 1) * 100)
    else:
        return int(-100 / (decimal_odds - 1))


def implied_probability(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    return 1 / decimal_odds


def calculate_overround(odds_a: float, odds_b: float) -> float:
    """Calculate bookmaker's overround (vig/juice)."""
    return (implied_probability(odds_a) + implied_probability(odds_b) - 1) * 100


def remove_vig(odds_a: float, odds_b: float) -> tuple:
    """Remove vig to get fair odds."""
    prob_a = implied_probability(odds_a)
    prob_b = implied_probability(odds_b)
    total = prob_a + prob_b

    fair_prob_a = prob_a / total
    fair_prob_b = prob_b / total

    return (1 / fair_prob_a, 1 / fair_prob_b)


def normalize_player_name(name: str) -> str:
    """Normalize player name for matching."""
    # Remove common suffixes/prefixes
    name = name.strip()
    name = re.sub(r'\s+', ' ', name)

    # Handle "Last, First" format
    if ',' in name:
        parts = name.split(',')
        if len(parts) == 2:
            name = f"{parts[1].strip()} {parts[0].strip()}"

    return name.lower()


def match_player_names(name1: str, name2: str) -> bool:
    """Check if two player names refer to the same person."""
    n1 = normalize_player_name(name1)
    n2 = normalize_player_name(name2)

    # Exact match
    if n1 == n2:
        return True

    # Check if last names match (common case)
    parts1 = n1.split()
    parts2 = n2.split()

    if parts1[-1] == parts2[-1]:
        # Same last name, check first initial
        if parts1[0][0] == parts2[0][0]:
            return True

    return False


# Example usage and testing
if __name__ == "__main__":
    print("Odds Scraper Module")
    print("=" * 50)

    # Test odds conversions
    print("\nOdds Conversion Examples:")
    print(f"American +150 -> Decimal: {american_to_decimal(150):.2f}")
    print(f"American -150 -> Decimal: {american_to_decimal(-150):.2f}")
    print(f"Decimal 2.50 -> American: {decimal_to_american(2.50)}")
    print(f"Decimal 1.67 -> American: {decimal_to_american(1.67)}")

    # Test overround calculation
    print("\nOverround Example:")
    odds_a, odds_b = 1.90, 1.90
    print(f"Odds: {odds_a} vs {odds_b}")
    print(f"Overround: {calculate_overround(odds_a, odds_b):.1f}%")

    fair_a, fair_b = remove_vig(odds_a, odds_b)
    print(f"Fair odds: {fair_a:.2f} vs {fair_b:.2f}")
