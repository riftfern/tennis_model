"""
Head-to-head analysis for tennis predictions.

Analyzes historical matchups between players:
- Overall H2H record
- Surface-specific H2H
- Recent results weighting
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class H2HRecord:
    """Head-to-head record between two players."""
    player_a: str
    player_b: str
    wins_a: int = 0
    wins_b: int = 0

    # Surface breakdown
    hard_a: int = 0
    hard_b: int = 0
    clay_a: int = 0
    clay_b: int = 0
    grass_a: int = 0
    grass_b: int = 0

    # Recent matches (most recent first)
    recent_matches: List[Dict] = field(default_factory=list)

    @property
    def total_matches(self) -> int:
        return self.wins_a + self.wins_b

    def get_surface_record(self, surface: str) -> Tuple[int, int]:
        """Get (player_a_wins, player_b_wins) for a surface."""
        surface = surface.lower()
        if 'hard' in surface:
            return (self.hard_a, self.hard_b)
        elif 'clay' in surface:
            return (self.clay_a, self.clay_b)
        elif 'grass' in surface:
            return (self.grass_a, self.grass_b)
        return (self.wins_a, self.wins_b)

    def win_rate_a(self, surface: str = None) -> float:
        """Get player A's win rate, optionally for a specific surface."""
        if surface:
            wins_a, wins_b = self.get_surface_record(surface)
            total = wins_a + wins_b
        else:
            wins_a = self.wins_a
            total = self.total_matches

        if total == 0:
            return 0.5
        return wins_a / total


class H2HAnalyzer:
    """Analyze and weight head-to-head records."""

    def __init__(self, matches_df: pd.DataFrame = None):
        """
        Args:
            matches_df: Historical matches dataframe
        """
        self.matches = matches_df
        self.h2h_cache: Dict[Tuple[str, str], H2HRecord] = {}

    def set_matches(self, matches_df: pd.DataFrame):
        """Set or update the matches dataframe."""
        self.matches = matches_df
        self.h2h_cache.clear()

    def get_h2h(self, player_a: str, player_b: str) -> H2HRecord:
        """Get H2H record between two players."""
        # Check cache with canonical key (sorted names)
        key = tuple(sorted([player_a, player_b]))

        if key in self.h2h_cache:
            cached = self.h2h_cache[key]
            # Return with correct orientation
            if cached.player_a == player_a:
                return cached
            else:
                # Swap orientation
                return H2HRecord(
                    player_a=player_a,
                    player_b=player_b,
                    wins_a=cached.wins_b,
                    wins_b=cached.wins_a,
                    hard_a=cached.hard_b,
                    hard_b=cached.hard_a,
                    clay_a=cached.clay_b,
                    clay_b=cached.clay_a,
                    grass_a=cached.grass_b,
                    grass_b=cached.grass_a,
                    recent_matches=cached.recent_matches
                )

        if self.matches is None or len(self.matches) == 0:
            return H2HRecord(player_a=player_a, player_b=player_b)

        # Query from matches
        h2h_matches = self.matches[
            ((self.matches['winner_name'] == player_a) & (self.matches['loser_name'] == player_b)) |
            ((self.matches['winner_name'] == player_b) & (self.matches['loser_name'] == player_a))
        ].copy()

        if 'tourney_date' in h2h_matches.columns:
            h2h_matches = h2h_matches.sort_values('tourney_date', ascending=False)

        record = H2HRecord(player_a=player_a, player_b=player_b)

        for _, match in h2h_matches.iterrows():
            winner = match['winner_name']
            surface = str(match.get('surface', 'Hard')).lower()

            if winner == player_a:
                record.wins_a += 1
                if 'hard' in surface:
                    record.hard_a += 1
                elif 'clay' in surface:
                    record.clay_a += 1
                elif 'grass' in surface:
                    record.grass_a += 1
            else:
                record.wins_b += 1
                if 'hard' in surface:
                    record.hard_b += 1
                elif 'clay' in surface:
                    record.clay_b += 1
                elif 'grass' in surface:
                    record.grass_b += 1

            # Store recent matches (up to 10)
            if len(record.recent_matches) < 10:
                record.recent_matches.append({
                    'date': match.get('tourney_date'),
                    'surface': surface,
                    'winner': winner,
                    'tournament': match.get('tourney_name', ''),
                    'score': match.get('score', '')
                })

        # Cache with canonical key
        self.h2h_cache[key] = record

        return record

    def calculate_h2h_adjustment(
        self,
        player_a: str,
        player_b: str,
        surface: str,
        base_prob_a: float
    ) -> float:
        """
        Calculate probability adjustment based on H2H record.

        Weighting strategy:
        - No H2H (0 matches): No adjustment
        - Limited H2H (1-2 matches): Small weight (5-10%)
        - Moderate H2H (3-5 matches): Medium weight (10-16%)
        - Significant H2H (6+ matches): Full weight (16-20% max)

        Args:
            player_a: First player name
            player_b: Second player name
            surface: Match surface
            base_prob_a: Base probability for player A from rating system

        Returns:
            Adjusted probability for player A
        """
        record = self.get_h2h(player_a, player_b)
        total = record.total_matches

        if total == 0:
            return base_prob_a

        # Surface-specific H2H
        surface_a, surface_b = record.get_surface_record(surface)
        surface_total = surface_a + surface_b

        # Calculate H2H win rate for player A
        overall_rate = record.win_rate_a()

        if surface_total >= 2:
            surface_rate = surface_a / surface_total
            # Weighted average favoring surface-specific when available
            h2h_rate = 0.6 * surface_rate + 0.4 * overall_rate
        else:
            h2h_rate = overall_rate

        # Weight based on sample size
        if total <= 2:
            weight = 0.05 * total  # 5-10%
        elif total <= 5:
            weight = 0.10 + 0.02 * (total - 2)  # 10-16%
        else:
            weight = 0.16 + 0.01 * min(total - 5, 4)  # 16-20% max

        # Recency boost: weight recent matches more
        if record.recent_matches:
            recent_3 = record.recent_matches[:3]
            recent_wins_a = sum(1 for m in recent_3 if m['winner'] == player_a)
            if len(recent_3) > 0:
                recent_rate = recent_wins_a / len(recent_3)
                # Blend recent into H2H rate (30% recent, 70% overall)
                h2h_rate = 0.7 * h2h_rate + 0.3 * recent_rate

        # Blend H2H with base probability
        adjusted = base_prob_a * (1 - weight) + h2h_rate * weight

        # Ensure result is bounded
        return max(0.05, min(0.95, adjusted))

    def get_h2h_summary(self, player_a: str, player_b: str) -> Dict:
        """Get a summary of H2H for display."""
        record = self.get_h2h(player_a, player_b)

        return {
            'total_matches': record.total_matches,
            f'{player_a}_wins': record.wins_a,
            f'{player_b}_wins': record.wins_b,
            'hard': f"{record.hard_a}-{record.hard_b}",
            'clay': f"{record.clay_a}-{record.clay_b}",
            'grass': f"{record.grass_a}-{record.grass_b}",
            'recent': [
                f"{m['winner']} ({m['surface']}, {m.get('tournament', '')})"
                for m in record.recent_matches[:5]
            ]
        }
