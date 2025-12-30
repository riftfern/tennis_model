"""
Surface analysis for tennis predictions.

Analyzes:
- Surface transition penalties
- Surface specialist detection
- Surface-specific performance profiles
"""

import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class SurfaceProfile:
    """Surface performance profile for a player."""
    matches: int = 0
    wins: int = 0
    win_rate: float = 0.5
    is_specialist: bool = False  # >50 matches and >65% win rate


class SurfaceAnalyzer:
    """Analyze surface-specific performance and transitions."""

    # Surface transition penalties (probability adjustment)
    TRANSITION_PENALTIES = {
        ('clay', 'grass'): 0.03,    # Hardest transition
        ('grass', 'clay'): 0.03,
        ('clay', 'hard'): 0.015,
        ('hard', 'clay'): 0.015,
        ('hard', 'grass'): 0.01,
        ('grass', 'hard'): 0.01,
    }

    def __init__(self, matches_df: pd.DataFrame = None):
        self.matches = matches_df
        self.surface_profiles: Dict[str, Dict[str, SurfaceProfile]] = defaultdict(dict)
        self.last_surface: Dict[str, str] = {}  # player -> last surface played
        self.last_match_date: Dict[str, object] = {}  # player -> last match date

    def set_matches(self, matches_df: pd.DataFrame):
        """Set or update matches dataframe."""
        self.matches = matches_df

    def build_profiles(self):
        """Build surface profiles from match data."""
        if self.matches is None:
            return

        # Reset
        self.surface_profiles.clear()

        for _, match in self.matches.iterrows():
            winner = match['winner_name']
            loser = match['loser_name']
            surface = str(match.get('surface', 'Hard')).lower()

            # Normalize surface names
            if 'hard' in surface:
                surface = 'hard'
            elif 'clay' in surface:
                surface = 'clay'
            elif 'grass' in surface:
                surface = 'grass'
            else:
                surface = 'hard'  # Default

            # Update winner
            if surface not in self.surface_profiles[winner]:
                self.surface_profiles[winner][surface] = SurfaceProfile()
            self.surface_profiles[winner][surface].matches += 1
            self.surface_profiles[winner][surface].wins += 1

            # Update loser
            if surface not in self.surface_profiles[loser]:
                self.surface_profiles[loser][surface] = SurfaceProfile()
            self.surface_profiles[loser][surface].matches += 1

        # Calculate win rates and specialist flags
        for player in self.surface_profiles:
            for surface in self.surface_profiles[player]:
                profile = self.surface_profiles[player][surface]
                if profile.matches > 0:
                    profile.win_rate = profile.wins / profile.matches
                    profile.is_specialist = (
                        profile.matches >= 50 and
                        profile.win_rate >= 0.65
                    )

    def update_player_surface(
        self,
        player: str,
        surface: str,
        match_date,
        won: bool
    ):
        """Update player's surface tracking after a match."""
        surface = surface.lower()
        if 'hard' in surface:
            surface = 'hard'
        elif 'clay' in surface:
            surface = 'clay'
        elif 'grass' in surface:
            surface = 'grass'

        # Track last surface and date
        self.last_surface[player] = surface
        self.last_match_date[player] = match_date

        # Update profile
        if surface not in self.surface_profiles[player]:
            self.surface_profiles[player][surface] = SurfaceProfile()

        self.surface_profiles[player][surface].matches += 1
        if won:
            self.surface_profiles[player][surface].wins += 1

        # Recalculate win rate
        profile = self.surface_profiles[player][surface]
        profile.win_rate = profile.wins / profile.matches if profile.matches > 0 else 0.5
        profile.is_specialist = profile.matches >= 50 and profile.win_rate >= 0.65

    def get_surface_profile(self, player: str, surface: str) -> SurfaceProfile:
        """Get surface profile for a player."""
        surface = surface.lower()
        if 'hard' in surface:
            surface = 'hard'
        elif 'clay' in surface:
            surface = 'clay'
        elif 'grass' in surface:
            surface = 'grass'

        if player in self.surface_profiles and surface in self.surface_profiles[player]:
            return self.surface_profiles[player][surface]

        return SurfaceProfile()  # Default

    def get_average_win_rate(self, player: str) -> float:
        """Get player's average win rate across surfaces."""
        if player not in self.surface_profiles:
            return 0.5

        profiles = self.surface_profiles[player]
        if not profiles:
            return 0.5

        total_matches = sum(p.matches for p in profiles.values())
        total_wins = sum(p.wins for p in profiles.values())

        if total_matches == 0:
            return 0.5

        return total_wins / total_matches

    def calculate_surface_adjustment(
        self,
        player: str,
        target_surface: str,
        last_surface: str = None,
        days_since_last: int = None
    ) -> float:
        """
        Calculate surface-based probability adjustment.

        Args:
            player: Player name
            target_surface: Surface of upcoming match
            last_surface: Surface of player's last match (optional)
            days_since_last: Days since last match (optional)

        Returns:
            Adjustment value (typically -0.05 to +0.05)
        """
        target_surface = target_surface.lower()
        adjustment = 0.0

        # 1. Surface specialist bonus/penalty
        target_profile = self.get_surface_profile(player, target_surface)
        avg_win_rate = self.get_average_win_rate(player)

        if target_profile.matches >= 20:
            surface_diff = target_profile.win_rate - avg_win_rate
            # Apply 30% of the differential
            adjustment += surface_diff * 0.3

        # 2. Specialist bonus
        if target_profile.is_specialist:
            adjustment += 0.02  # Extra 2% for specialists

        # 3. Transition penalty
        if last_surface is None:
            last_surface = self.last_surface.get(player)

        if last_surface and last_surface.lower() != target_surface:
            # Normalize surface names
            last_norm = last_surface.lower()
            if 'hard' in last_norm:
                last_norm = 'hard'
            elif 'clay' in last_norm:
                last_norm = 'clay'
            elif 'grass' in last_norm:
                last_norm = 'grass'

            target_norm = target_surface
            if 'hard' in target_norm:
                target_norm = 'hard'
            elif 'clay' in target_norm:
                target_norm = 'clay'
            elif 'grass' in target_norm:
                target_norm = 'grass'

            if last_norm != target_norm:
                key = (last_norm, target_norm)
                transition_penalty = self.TRANSITION_PENALTIES.get(key, 0)

                # Reduce penalty if more time to adjust
                if days_since_last is not None:
                    if days_since_last > 14:
                        transition_penalty *= 0.25  # Much reduced
                    elif days_since_last > 7:
                        transition_penalty *= 0.5  # Reduced

                adjustment -= transition_penalty

        return adjustment

    def process_matches_df(self, matches_df: pd.DataFrame):
        """Process matches to build profiles and track surfaces."""
        self.matches = matches_df
        self.build_profiles()

        # Also track last surface per player
        if 'tourney_date' in matches_df.columns:
            sorted_df = matches_df.sort_values('tourney_date')
        else:
            sorted_df = matches_df

        for _, match in sorted_df.iterrows():
            winner = match['winner_name']
            loser = match['loser_name']
            surface = str(match.get('surface', 'Hard')).lower()
            date = match.get('tourney_date')

            self.last_surface[winner] = surface
            self.last_surface[loser] = surface
            self.last_match_date[winner] = date
            self.last_match_date[loser] = date
