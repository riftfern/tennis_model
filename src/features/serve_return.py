"""
Serve and return statistics tracking for tennis predictions.

Tracks rolling averages of serve/return performance:
- First serve %, first serve points won %
- Second serve points won %
- Ace rate, double fault rate
- Break point save/conversion rates
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ServeReturnStats:
    """Rolling serve/return statistics for a player."""
    # Serve metrics
    first_serve_pct: float = 0.60       # % of first serves in
    first_serve_won_pct: float = 0.70   # % of first serve points won
    second_serve_won_pct: float = 0.50  # % of second serve points won
    ace_rate: float = 0.05              # Aces per service point
    df_rate: float = 0.03               # Double faults per service point
    service_hold_pct: float = 0.80      # % of service games won
    bp_save_pct: float = 0.60           # Break points saved %

    # Return metrics (derived from opponent's serve)
    first_return_won_pct: float = 0.30  # % of opponent's 1st serve points won
    second_return_won_pct: float = 0.50 # % of opponent's 2nd serve points won
    bp_conversion_pct: float = 0.40     # Break point conversion %

    # Sample sizes
    serve_points_total: int = 0
    return_points_total: int = 0
    matches_in_sample: int = 0

    @property
    def service_points_won_pct(self) -> float:
        """Overall service points won percentage."""
        return (self.first_serve_pct * self.first_serve_won_pct +
                (1 - self.first_serve_pct) * self.second_serve_won_pct)

    @property
    def return_points_won_pct(self) -> float:
        """Overall return points won percentage (estimate)."""
        # Assume opponent has similar first serve %
        return (0.60 * self.first_return_won_pct +
                0.40 * self.second_return_won_pct)


@dataclass
class MatchServeData:
    """Serve data from a single match."""
    svpt: int = 0           # Service points
    first_in: int = 0       # First serves in
    first_won: int = 0      # Points won on first serve
    second_won: int = 0     # Points won on second serve
    aces: int = 0
    dfs: int = 0
    sv_games: int = 0       # Service games
    sv_games_won: int = 0   # Service games won (estimated)
    bp_faced: int = 0
    bp_saved: int = 0

    # Return data (opponent's serve stats)
    ret_svpt: int = 0
    ret_first_in: int = 0
    ret_first_won: int = 0   # First return points won
    ret_second_won: int = 0  # Second return points won
    bp_chances: int = 0
    bp_converted: int = 0


class ServeReturnTracker:
    """Track rolling serve/return stats per player."""

    def __init__(
        self,
        window_matches: int = 20,
        surface_specific: bool = True,
        min_points: int = 50
    ):
        """
        Args:
            window_matches: Number of recent matches to use
            surface_specific: Track separate stats per surface
            min_points: Minimum service points for reliable stats
        """
        self.window = window_matches
        self.surface_specific = surface_specific
        self.min_points = min_points

        # player -> surface -> list of MatchServeData
        self.player_history: Dict[str, Dict[str, List[MatchServeData]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def _safe_int(self, value, default: int = 0) -> int:
        """Safely convert value to int, handling NaN."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def _extract_serve_data(self, match: Dict, prefix: str) -> Optional[MatchServeData]:
        """Extract serve stats from match row."""
        svpt = match.get(f'{prefix}svpt')
        if svpt is None or (isinstance(svpt, float) and pd.isna(svpt)) or svpt == 0:
            return None

        svpt = self._safe_int(svpt)
        if svpt == 0:
            return None

        first_in = self._safe_int(match.get(f'{prefix}1stIn', 0))
        first_won = self._safe_int(match.get(f'{prefix}1stWon', 0))
        second_won = self._safe_int(match.get(f'{prefix}2ndWon', 0))
        aces = self._safe_int(match.get(f'{prefix}ace', 0))
        dfs = self._safe_int(match.get(f'{prefix}df', 0))
        sv_games = self._safe_int(match.get(f'{prefix}SvGms', 0))
        bp_faced = self._safe_int(match.get(f'{prefix}bpFaced', 0))
        bp_saved = self._safe_int(match.get(f'{prefix}bpSaved', 0))

        # Estimate service games won (service games - breaks against)
        breaks_against = bp_faced - bp_saved
        sv_games_won = max(0, sv_games - breaks_against)

        return MatchServeData(
            svpt=svpt,
            first_in=first_in,
            first_won=first_won,
            second_won=second_won,
            aces=aces,
            dfs=dfs,
            sv_games=sv_games,
            sv_games_won=sv_games_won,
            bp_faced=bp_faced,
            bp_saved=bp_saved
        )

    def process_match(self, match: Dict):
        """
        Process a match and update rolling stats for both players.

        Args:
            match: Dictionary with match data including serve stats
        """
        winner = match.get('winner_name')
        loser = match.get('loser_name')
        surface = match.get('surface', 'Hard').lower()

        if not winner or not loser:
            return

        # Extract serve data
        winner_serve = self._extract_serve_data(match, 'w_')
        loser_serve = self._extract_serve_data(match, 'l_')

        if winner_serve is None or loser_serve is None:
            return

        # Add return data (opponent's serve becomes return stats)
        winner_serve.ret_svpt = loser_serve.svpt
        winner_serve.ret_first_in = loser_serve.first_in
        winner_serve.ret_first_won = loser_serve.svpt - loser_serve.first_won - (loser_serve.svpt - loser_serve.first_in - loser_serve.second_won)
        winner_serve.ret_second_won = (loser_serve.svpt - loser_serve.first_in) - loser_serve.second_won
        winner_serve.bp_chances = loser_serve.bp_faced
        winner_serve.bp_converted = loser_serve.bp_faced - loser_serve.bp_saved

        loser_serve.ret_svpt = winner_serve.svpt
        loser_serve.ret_first_in = winner_serve.first_in
        loser_serve.ret_first_won = winner_serve.svpt - winner_serve.first_won - (winner_serve.svpt - winner_serve.first_in - winner_serve.second_won)
        loser_serve.ret_second_won = (winner_serve.svpt - winner_serve.first_in) - winner_serve.second_won
        loser_serve.bp_chances = winner_serve.bp_faced
        loser_serve.bp_converted = winner_serve.bp_faced - winner_serve.bp_saved

        # Update histories
        for surface_key in ([surface, 'all'] if self.surface_specific else ['all']):
            self.player_history[winner][surface_key].append(winner_serve)
            self.player_history[loser][surface_key].append(loser_serve)

            # Trim to window size
            if len(self.player_history[winner][surface_key]) > self.window:
                self.player_history[winner][surface_key] = self.player_history[winner][surface_key][-self.window:]
            if len(self.player_history[loser][surface_key]) > self.window:
                self.player_history[loser][surface_key] = self.player_history[loser][surface_key][-self.window:]

    def get_player_stats(
        self,
        player: str,
        surface: str = None
    ) -> ServeReturnStats:
        """Get current rolling stats for a player."""
        if player not in self.player_history:
            return ServeReturnStats()

        # Determine which surface(s) to use
        if surface and self.surface_specific:
            surface_key = surface.lower()
            if surface_key in self.player_history[player] and len(self.player_history[player][surface_key]) >= 3:
                history = self.player_history[player][surface_key]
            else:
                # Fall back to all surfaces if not enough surface-specific data
                history = self.player_history[player].get('all', [])
        else:
            history = self.player_history[player].get('all', [])

        if not history:
            return ServeReturnStats()

        # Aggregate stats
        total_svpt = sum(m.svpt for m in history)
        total_first_in = sum(m.first_in for m in history)
        total_first_won = sum(m.first_won for m in history)
        total_second_won = sum(m.second_won for m in history)
        total_aces = sum(m.aces for m in history)
        total_dfs = sum(m.dfs for m in history)
        total_sv_games = sum(m.sv_games for m in history)
        total_sv_games_won = sum(m.sv_games_won for m in history)
        total_bp_faced = sum(m.bp_faced for m in history)
        total_bp_saved = sum(m.bp_saved for m in history)

        total_ret_svpt = sum(m.ret_svpt for m in history)
        total_ret_first_in = sum(m.ret_first_in for m in history)
        total_ret_first_won = sum(m.ret_first_won for m in history)
        total_ret_second_won = sum(m.ret_second_won for m in history)
        total_bp_chances = sum(m.bp_chances for m in history)
        total_bp_converted = sum(m.bp_converted for m in history)

        # Calculate percentages with safety checks
        second_serve_total = total_svpt - total_first_in

        stats = ServeReturnStats(
            first_serve_pct=total_first_in / total_svpt if total_svpt > 0 else 0.60,
            first_serve_won_pct=total_first_won / total_first_in if total_first_in > 0 else 0.70,
            second_serve_won_pct=total_second_won / second_serve_total if second_serve_total > 0 else 0.50,
            ace_rate=total_aces / total_svpt if total_svpt > 0 else 0.05,
            df_rate=total_dfs / total_svpt if total_svpt > 0 else 0.03,
            service_hold_pct=total_sv_games_won / total_sv_games if total_sv_games > 0 else 0.80,
            bp_save_pct=total_bp_saved / total_bp_faced if total_bp_faced > 0 else 0.60,

            first_return_won_pct=total_ret_first_won / total_ret_first_in if total_ret_first_in > 0 else 0.30,
            second_return_won_pct=total_ret_second_won / (total_ret_svpt - total_ret_first_in) if (total_ret_svpt - total_ret_first_in) > 0 else 0.50,
            bp_conversion_pct=total_bp_converted / total_bp_chances if total_bp_chances > 0 else 0.40,

            serve_points_total=total_svpt,
            return_points_total=total_ret_svpt,
            matches_in_sample=len(history)
        )

        return stats

    def calculate_serve_dominance(
        self,
        player_a: str,
        player_b: str,
        surface: str = None
    ) -> float:
        """
        Calculate serve dominance differential between two players.

        Positive value = player_a is more dominant
        Returns value typically in range [-0.15, 0.15]
        """
        stats_a = self.get_player_stats(player_a, surface)
        stats_b = self.get_player_stats(player_b, surface)

        # Service points won differential
        spw_diff = stats_a.service_points_won_pct - stats_b.service_points_won_pct

        # Return points won differential
        rpw_diff = stats_a.return_points_won_pct - stats_b.return_points_won_pct

        # Break point performance differential
        bp_diff = (stats_a.bp_save_pct - stats_b.bp_save_pct) * 0.5 + \
                  (stats_a.bp_conversion_pct - stats_b.bp_conversion_pct) * 0.5

        # Combined dominance (weighted)
        dominance = spw_diff * 0.4 + rpw_diff * 0.4 + bp_diff * 0.2

        return dominance

    def process_matches_df(self, matches_df: pd.DataFrame):
        """Process entire dataframe of matches."""
        # Sort by date
        if 'tourney_date' in matches_df.columns:
            matches_df = matches_df.sort_values('tourney_date')

        for _, match in matches_df.iterrows():
            self.process_match(match.to_dict())
