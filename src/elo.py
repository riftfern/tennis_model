"""
Elo rating system for tennis players.

Uses surface-specific Elo ratings and adjusts for:
- Match importance (Grand Slam, Masters, etc.)
- Surface (hard, clay, grass)
- Recent form
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class PlayerRating:
    """Stores a player's Elo ratings."""
    overall: float = 1500.0
    hard: float = 1500.0
    clay: float = 1500.0
    grass: float = 1500.0
    matches_played: int = 0

    def get_surface_rating(self, surface: str) -> float:
        """Get rating for a specific surface."""
        surface = surface.lower()
        if 'hard' in surface:
            return self.hard
        elif 'clay' in surface:
            return self.clay
        elif 'grass' in surface:
            return self.grass
        return self.overall

    def set_surface_rating(self, surface: str, rating: float):
        """Set rating for a specific surface."""
        surface = surface.lower()
        if 'hard' in surface:
            self.hard = rating
        elif 'clay' in surface:
            self.clay = rating
        elif 'grass' in surface:
            self.grass = rating
        self.overall = (self.hard + self.clay + self.grass) / 3


class TennisElo:
    """
    Elo rating system optimized for tennis.

    Key features:
    - Surface-specific ratings
    - K-factor adjustment based on tournament level
    - New player adjustment
    """

    # K-factor by tournament level
    K_FACTORS = {
        'G': 32,      # Grand Slam
        'M': 28,      # Masters 1000
        'A': 24,      # ATP 500
        'B': 20,      # ATP 250
        'F': 28,      # Tour Finals
        'D': 20,      # Davis Cup
        'C': 16,      # Challenger
        'default': 20
    }

    def __init__(self, k_factor: float = 24, initial_rating: float = 1500):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings: Dict[str, PlayerRating] = {}

    def get_player_rating(self, player: str) -> PlayerRating:
        """Get or create a player's rating."""
        if player not in self.ratings:
            self.ratings[player] = PlayerRating(
                overall=self.initial_rating,
                hard=self.initial_rating,
                clay=self.initial_rating,
                grass=self.initial_rating
            )
        return self.ratings[player]

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score (win probability) for player A."""
        exponent = (rating_b - rating_a) / 400
        exponent = max(-10, min(10, exponent))  # Prevent overflow
        return 1 / (1 + 10 ** exponent)

    def get_k_factor(self, tourney_level: str, matches_played: int) -> float:
        """Get K-factor based on tournament level and experience."""
        base_k = self.K_FACTORS.get(tourney_level, self.K_FACTORS['default'])

        # New player adjustment - higher K for players with fewer matches
        if matches_played < 30:
            base_k *= 1.5
        elif matches_played < 100:
            base_k *= 1.2

        return base_k

    def update_ratings(
        self,
        winner: str,
        loser: str,
        surface: str,
        tourney_level: str = 'default'
    ) -> Tuple[float, float]:
        """
        Update Elo ratings after a match.

        Returns: (winner_new_rating, loser_new_rating)
        """
        winner_rating = self.get_player_rating(winner)
        loser_rating = self.get_player_rating(loser)

        # Get surface-specific ratings
        winner_elo = winner_rating.get_surface_rating(surface)
        loser_elo = loser_rating.get_surface_rating(surface)

        # Calculate expected scores
        expected_winner = self.expected_score(winner_elo, loser_elo)
        expected_loser = 1 - expected_winner

        # Get K-factors
        k_winner = self.get_k_factor(tourney_level, winner_rating.matches_played)
        k_loser = self.get_k_factor(tourney_level, loser_rating.matches_played)

        # Update ratings (winner scored 1, loser scored 0)
        new_winner_elo = winner_elo + k_winner * (1 - expected_winner)
        new_loser_elo = loser_elo + k_loser * (0 - expected_loser)

        # Apply updates
        winner_rating.set_surface_rating(surface, new_winner_elo)
        loser_rating.set_surface_rating(surface, new_loser_elo)
        winner_rating.matches_played += 1
        loser_rating.matches_played += 1

        return new_winner_elo, new_loser_elo

    def predict_match(
        self,
        player_a: str,
        player_b: str,
        surface: str
    ) -> Dict[str, float]:
        """
        Predict match outcome probabilities.

        Returns dict with win probabilities for each player.
        """
        rating_a = self.get_player_rating(player_a)
        rating_b = self.get_player_rating(player_b)

        elo_a = rating_a.get_surface_rating(surface)
        elo_b = rating_b.get_surface_rating(surface)

        prob_a = self.expected_score(elo_a, elo_b)

        return {
            player_a: prob_a,
            player_b: 1 - prob_a,
            'elo_diff': elo_a - elo_b,
            f'{player_a}_elo': elo_a,
            f'{player_b}_elo': elo_b
        }

    def process_matches(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a dataframe of historical matches to build ratings.

        Expected columns: winner_name, loser_name, surface, tourney_level, tourney_date
        """
        # Sort by date
        if 'tourney_date' in matches_df.columns:
            matches_df = matches_df.sort_values('tourney_date')

        results = []

        for _, match in matches_df.iterrows():
            winner = match['winner_name']
            loser = match['loser_name']
            surface = match.get('surface', 'Hard')
            tourney_level = match.get('tourney_level', 'default')

            # Get pre-match ratings
            pre_winner_elo = self.get_player_rating(winner).get_surface_rating(surface)
            pre_loser_elo = self.get_player_rating(loser).get_surface_rating(surface)

            # Calculate pre-match prediction
            expected = self.expected_score(pre_winner_elo, pre_loser_elo)

            # Update ratings
            self.update_ratings(winner, loser, surface, tourney_level)

            results.append({
                'winner': winner,
                'loser': loser,
                'surface': surface,
                'winner_pre_elo': pre_winner_elo,
                'loser_pre_elo': pre_loser_elo,
                'predicted_winner_prob': expected,
                'prediction_correct': expected > 0.5
            })

        return pd.DataFrame(results)

    def get_top_players(self, n: int = 50, surface: Optional[str] = None) -> pd.DataFrame:
        """Get top N players by Elo rating."""
        data = []
        for player, rating in self.ratings.items():
            if surface:
                elo = rating.get_surface_rating(surface)
            else:
                elo = rating.overall
            data.append({
                'player': player,
                'elo': elo,
                'matches': rating.matches_played,
                'hard': rating.hard,
                'clay': rating.clay,
                'grass': rating.grass
            })

        df = pd.DataFrame(data)
        return df.nlargest(n, 'elo')


def implied_probability(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    return 1 / decimal_odds


def calculate_ev(model_prob: float, decimal_odds: float) -> float:
    """
    Calculate expected value of a bet.

    Args:
        model_prob: Our model's probability of winning (0-1)
        decimal_odds: Decimal odds offered by bookmaker

    Returns:
        Expected value as a percentage (e.g., 5.0 means +5% EV)
    """
    # EV = (prob_win * profit) - (prob_lose * stake)
    # For a $1 stake at decimal odds:
    profit_if_win = decimal_odds - 1
    prob_lose = 1 - model_prob

    ev = (model_prob * profit_if_win) - (prob_lose * 1)
    return ev * 100  # Return as percentage


def find_value_bets(
    predictions: Dict[str, Dict],
    odds: Dict[str, Dict],
    min_ev: float = 2.0,
    min_prob: float = 0.1
) -> pd.DataFrame:
    """
    Find +EV betting opportunities.

    Args:
        predictions: Dict of match_id -> {player: probability}
        odds: Dict of match_id -> {player: decimal_odds}
        min_ev: Minimum EV% to consider (default 2%)
        min_prob: Minimum model probability (avoid long shots)

    Returns:
        DataFrame of value bets sorted by EV
    """
    value_bets = []

    for match_id, pred in predictions.items():
        if match_id not in odds:
            continue

        match_odds = odds[match_id]

        for player, prob in pred.items():
            if player not in match_odds or prob < min_prob:
                continue

            decimal_odds = match_odds[player]
            ev = calculate_ev(prob, decimal_odds)
            implied_prob = implied_probability(decimal_odds)

            if ev >= min_ev:
                value_bets.append({
                    'match_id': match_id,
                    'player': player,
                    'model_prob': prob,
                    'implied_prob': implied_prob,
                    'decimal_odds': decimal_odds,
                    'ev_percent': ev,
                    'edge': prob - implied_prob
                })

    df = pd.DataFrame(value_bets)
    if len(df) > 0:
        df = df.sort_values('ev_percent', ascending=False)
    return df
