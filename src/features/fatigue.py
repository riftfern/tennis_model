"""
Fatigue tracking for tennis predictions.

Tracks factors that affect player stamina:
- Recent match load (7/14/30 days)
- Match duration stress
- Tournament depth
- Travel across continents
"""

import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict


@dataclass
class MatchLoad:
    """Record of a recent match for fatigue calculation."""
    date: datetime
    tournament: str
    round_name: str
    minutes: int
    sets_played: int
    location_ioc: str


@dataclass
class PlayerFatigue:
    """Current fatigue state for a player."""
    matches_7_days: int = 0
    matches_14_days: int = 0
    matches_30_days: int = 0
    minutes_7_days: int = 0
    minutes_14_days: int = 0
    sets_7_days: int = 0
    current_tournament_matches: int = 0
    last_match_date: Optional[datetime] = None
    last_location: Optional[str] = None


class FatigueTracker:
    """Track and calculate fatigue factors for players."""

    # Round progression (for tournament depth)
    ROUND_ORDER = {
        'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4,
        'QF': 5, 'SF': 6, 'F': 7,
        'RR': 3, 'BR': 4,  # Round robin, bronze match
        '1R': 1, '2R': 2, '3R': 3, '4R': 4,  # Alternative naming
    }

    # Continent mapping for travel fatigue
    CONTINENT_MAP = {
        # North America
        'USA': 'NA', 'CAN': 'NA', 'MEX': 'NA',
        # South America
        'BRA': 'SA', 'ARG': 'SA', 'CHI': 'SA', 'COL': 'SA', 'PER': 'SA', 'ECU': 'SA',
        # Europe
        'GBR': 'EU', 'FRA': 'EU', 'GER': 'EU', 'ESP': 'EU', 'ITA': 'EU',
        'AUT': 'EU', 'SUI': 'EU', 'NED': 'EU', 'BEL': 'EU', 'CZE': 'EU',
        'RUS': 'EU', 'POL': 'EU', 'SRB': 'EU', 'CRO': 'EU', 'GRE': 'EU',
        'POR': 'EU', 'SWE': 'EU', 'NOR': 'EU', 'DEN': 'EU', 'FIN': 'EU',
        'UKR': 'EU', 'ROU': 'EU', 'HUN': 'EU', 'SVK': 'EU', 'BUL': 'EU',
        'MON': 'EU', 'LUX': 'EU',
        # Oceania
        'AUS': 'OC', 'NZL': 'OC',
        # Asia
        'CHN': 'AS', 'JPN': 'AS', 'KOR': 'AS', 'IND': 'AS', 'UAE': 'AS',
        'QAT': 'AS', 'KAZ': 'AS', 'THA': 'AS', 'SGP': 'AS', 'HKG': 'AS',
        'TPE': 'AS', 'MAS': 'AS', 'IDN': 'AS',
        # Africa
        'RSA': 'AF', 'EGY': 'AF', 'MAR': 'AF', 'TUN': 'AF',
    }

    def __init__(self):
        # player -> list of MatchLoad
        self.player_history: Dict[str, List[MatchLoad]] = defaultdict(list)

    def add_match(
        self,
        player: str,
        match_date: datetime,
        tournament: str,
        round_name: str,
        minutes: int,
        sets_played: int,
        location_ioc: str
    ):
        """Add a match to player's history."""
        match = MatchLoad(
            date=match_date,
            tournament=tournament,
            round_name=round_name,
            minutes=minutes or 0,
            sets_played=sets_played,
            location_ioc=location_ioc or 'UNK'
        )

        self.player_history[player].append(match)

        # Keep only last 60 days
        cutoff = match_date - timedelta(days=60)
        self.player_history[player] = [
            m for m in self.player_history[player]
            if m.date >= cutoff
        ]

    def process_match(self, match: Dict):
        """Process a match record and update both players."""
        winner = match.get('winner_name')
        loser = match.get('loser_name')

        # Parse date
        tourney_date = match.get('tourney_date')
        if isinstance(tourney_date, (int, float)):
            try:
                match_date = datetime.strptime(str(int(tourney_date)), '%Y%m%d')
            except:
                match_date = datetime.now()
        elif isinstance(tourney_date, datetime):
            match_date = tourney_date
        else:
            match_date = datetime.now()

        tournament = match.get('tourney_name', '')
        round_name = match.get('round', '')
        minutes = match.get('minutes', 0) or 0

        # Estimate sets from score
        score = match.get('score', '')
        sets_played = score.count('-') if isinstance(score, str) else 3

        # Location from winner's IOC (approximation)
        location = match.get('winner_ioc', 'UNK')

        if winner:
            self.add_match(winner, match_date, tournament, round_name,
                          minutes, sets_played, location)
        if loser:
            self.add_match(loser, match_date, tournament, round_name,
                          minutes, sets_played, location)

    def get_player_fatigue(
        self,
        player: str,
        current_date: datetime
    ) -> PlayerFatigue:
        """Calculate current fatigue state for a player."""
        history = self.player_history.get(player, [])

        fatigue = PlayerFatigue()

        if not history:
            return fatigue

        for match in history:
            days_ago = (current_date - match.date).days

            if days_ago <= 7:
                fatigue.matches_7_days += 1
                fatigue.minutes_7_days += match.minutes
                fatigue.sets_7_days += match.sets_played
            if days_ago <= 14:
                fatigue.matches_14_days += 1
                fatigue.minutes_14_days += match.minutes
            if days_ago <= 30:
                fatigue.matches_30_days += 1

        # Find last match
        sorted_history = sorted(history, key=lambda m: m.date, reverse=True)
        if sorted_history:
            fatigue.last_match_date = sorted_history[0].date
            fatigue.last_location = sorted_history[0].location_ioc

            # Count matches in current tournament
            last_tournament = sorted_history[0].tournament
            fatigue.current_tournament_matches = sum(
                1 for m in sorted_history
                if m.tournament == last_tournament and
                (current_date - m.date).days <= 14
            )

        return fatigue

    def calculate_fatigue_penalty(
        self,
        player: str,
        current_date: datetime,
        tournament_name: str = None,
        tournament_location: str = None
    ) -> float:
        """
        Calculate fatigue penalty (negative probability adjustment).

        Returns: Penalty in range [0, 0.15] where higher = more tired
        """
        fatigue = self.get_player_fatigue(player, current_date)
        penalty = 0.0

        # 1. Recent match load
        if fatigue.matches_7_days >= 4:
            penalty += 0.03 * (fatigue.matches_7_days - 3)  # 3%+ per extra match
        if fatigue.matches_14_days >= 7:
            penalty += 0.02 * (fatigue.matches_14_days - 6)

        # 2. Match duration stress (long matches drain energy)
        if fatigue.minutes_7_days > 300:  # 5+ hours in a week
            extra_hours = (fatigue.minutes_7_days - 300) / 60
            penalty += 0.02 * extra_hours

        # 3. Tournament depth
        if fatigue.current_tournament_matches >= 4:  # QF or later
            penalty += 0.01 * (fatigue.current_tournament_matches - 3)

        # 4. Travel fatigue (continent change in last 10 days)
        if fatigue.last_location and tournament_location:
            prev_continent = self.CONTINENT_MAP.get(fatigue.last_location, 'EU')
            curr_continent = self.CONTINENT_MAP.get(tournament_location, 'EU')

            if prev_continent != curr_continent:
                if fatigue.last_match_date:
                    days_since = (current_date - fatigue.last_match_date).days
                    if days_since <= 10:
                        penalty += 0.02  # Jet lag penalty
                        if days_since <= 5:
                            penalty += 0.01  # Severe jet lag

        # 5. Short rest (back-to-back days)
        if fatigue.last_match_date:
            days_since = (current_date - fatigue.last_match_date).days
            if days_since == 0:
                penalty += 0.03  # Same day (very rare, doubles maybe)
            elif days_since == 1:
                penalty += 0.01  # Day after

        # Cap penalty at 15%
        return min(0.15, penalty)

    def process_matches_df(self, matches_df: pd.DataFrame):
        """Process entire dataframe of matches."""
        if 'tourney_date' in matches_df.columns:
            matches_df = matches_df.sort_values('tourney_date')

        for _, match in matches_df.iterrows():
            self.process_match(match.to_dict())
