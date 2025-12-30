"""
Enhanced Elo rating system for tennis players.

Improvements over basic Elo:
- Recency decay for inactive players
- Form adjustment based on recent results vs expectation
- Opponent strength calibration for K-factor
- Surface-specific ratings with proper weighting
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class EnhancedPlayerRating:
    """Enhanced player rating with additional tracking."""
    overall: float = 1500.0
    hard: float = 1500.0
    clay: float = 1500.0
    grass: float = 1500.0
    matches_played: int = 0

    # Enhanced tracking
    last_match_date: Optional[datetime] = None
    recent_results: List[Tuple[datetime, bool, float, str]] = field(default_factory=list)  # (date, won, opp_rating, surface)
    form_adjustment: float = 0.0

    # Surface match counts
    hard_matches: int = 0
    clay_matches: int = 0
    grass_matches: int = 0

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

    def get_surface_matches(self, surface: str) -> int:
        """Get match count for a surface."""
        surface = surface.lower()
        if 'hard' in surface:
            return self.hard_matches
        elif 'clay' in surface:
            return self.clay_matches
        elif 'grass' in surface:
            return self.grass_matches
        return self.matches_played

    def add_result(self, date: datetime, won: bool, opp_rating: float, surface: str):
        """Add a match result to recent history."""
        self.recent_results.append((date, won, opp_rating, surface))
        # Keep only last 20 results
        if len(self.recent_results) > 20:
            self.recent_results = self.recent_results[-20:]

        # Update surface match count
        surface = surface.lower()
        if 'hard' in surface:
            self.hard_matches += 1
        elif 'clay' in surface:
            self.clay_matches += 1
        elif 'grass' in surface:
            self.grass_matches += 1


class EnhancedElo:
    """
    Enhanced Elo rating system with recency, form, and calibration.
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

    def __init__(
        self,
        initial_rating: float = 1500,
        inactivity_threshold_days: int = 180,
        decay_rate: float = 0.05,
        form_weight: float = 0.3
    ):
        self.initial_rating = initial_rating
        self.inactivity_threshold = inactivity_threshold_days
        self.decay_rate = decay_rate  # 5% per month after threshold
        self.form_weight = form_weight
        self.ratings: Dict[str, EnhancedPlayerRating] = {}

    def get_player_rating(self, player: str) -> EnhancedPlayerRating:
        """Get or create a player's rating."""
        if player not in self.ratings:
            self.ratings[player] = EnhancedPlayerRating(
                overall=self.initial_rating,
                hard=self.initial_rating,
                clay=self.initial_rating,
                grass=self.initial_rating
            )
        return self.ratings[player]

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score (win probability) for player A."""
        # Clip the exponent to avoid overflow
        exponent = (rating_b - rating_a) / 400
        exponent = max(-10, min(10, exponent))  # Bounds: ~0.00001 to ~0.99999
        return 1 / (1 + 10 ** exponent)

    def apply_inactivity_decay(
        self,
        player: str,
        current_date: datetime
    ) -> float:
        """
        Apply rating decay for inactive players.
        Returns the decay amount applied.
        """
        rating = self.ratings.get(player)
        if rating is None or rating.last_match_date is None:
            return 0.0

        days_inactive = (current_date - rating.last_match_date).days

        if days_inactive <= self.inactivity_threshold:
            return 0.0

        # Calculate months of inactivity beyond threshold
        months_over = (days_inactive - self.inactivity_threshold) / 30

        # Decay toward 1500 (regression to mean)
        for surface in ['hard', 'clay', 'grass']:
            current = getattr(rating, surface)
            target = self.initial_rating
            decay = (current - target) * self.decay_rate * months_over
            setattr(rating, surface, current - decay)

        # Update overall
        rating.overall = (rating.hard + rating.clay + rating.grass) / 3

        return decay

    def calculate_form_adjustment(
        self,
        player: str,
        surface: str = None,
        n_matches: int = 10
    ) -> float:
        """
        Calculate form bonus/penalty based on recent results.
        Returns adjustment in Elo points (-50 to +50).
        """
        rating = self.ratings.get(player)
        if rating is None or len(rating.recent_results) < 3:
            return 0.0

        # Filter by surface if specified
        if surface:
            surface = surface.lower()
            recent = [(d, w, r, s) for d, w, r, s in rating.recent_results
                     if surface in s.lower()][-n_matches:]
        else:
            recent = rating.recent_results[-n_matches:]

        if len(recent) < 3:
            return 0.0

        # Calculate actual vs expected performance
        expected_wins = 0.0
        actual_wins = 0.0

        for _, won, opp_rating, _ in recent:
            player_rating = rating.overall
            expected_wins += self.expected_score(player_rating, opp_rating)
            actual_wins += 1 if won else 0

        # Performance delta as fraction
        performance_delta = (actual_wins - expected_wins) / len(recent)

        # Scale to Elo points (max +/- 50)
        form_adjustment = performance_delta * 100
        form_adjustment = max(-50, min(50, form_adjustment))

        rating.form_adjustment = form_adjustment
        return form_adjustment

    def get_k_factor(
        self,
        tourney_level: str,
        matches_played: int
    ) -> float:
        """Get base K-factor based on tournament level and experience."""
        base_k = self.K_FACTORS.get(tourney_level, self.K_FACTORS['default'])

        # New player adjustment
        if matches_played < 30:
            base_k *= 1.5
        elif matches_played < 100:
            base_k *= 1.2

        return base_k

    def get_adjusted_k_factor(
        self,
        tourney_level: str,
        matches_played: int,
        winner_rating: float,
        loser_rating: float
    ) -> float:
        """
        K-factor adjusted for upset magnitude.
        Upsets (underdog wins) get higher K-factor.
        """
        base_k = self.get_k_factor(tourney_level, matches_played)

        # Expected score for the winner
        expected = self.expected_score(winner_rating, loser_rating)

        # If winner was underdog (expected < 0.5), boost K-factor
        if expected < 0.5:
            upset_bonus = (0.5 - expected) * 0.5  # Up to 25% bonus
            base_k *= (1 + upset_bonus)

        return base_k

    def update_ratings(
        self,
        winner: str,
        loser: str,
        surface: str,
        tourney_level: str = 'default',
        match_date: datetime = None
    ) -> Tuple[float, float]:
        """
        Update Elo ratings after a match.
        Returns: (winner_new_rating, loser_new_rating)
        """
        if match_date is None:
            match_date = datetime.now()

        winner_rating = self.get_player_rating(winner)
        loser_rating = self.get_player_rating(loser)

        # Apply inactivity decay before update
        self.apply_inactivity_decay(winner, match_date)
        self.apply_inactivity_decay(loser, match_date)

        # Get surface-specific ratings
        winner_elo = winner_rating.get_surface_rating(surface)
        loser_elo = loser_rating.get_surface_rating(surface)

        # Calculate expected scores
        expected_winner = self.expected_score(winner_elo, loser_elo)

        # Get adjusted K-factors
        k_winner = self.get_adjusted_k_factor(
            tourney_level,
            winner_rating.matches_played,
            winner_elo,
            loser_elo
        )
        k_loser = self.get_adjusted_k_factor(
            tourney_level,
            loser_rating.matches_played,
            loser_elo,
            winner_elo
        )

        # Update ratings
        new_winner_elo = winner_elo + k_winner * (1 - expected_winner)
        new_loser_elo = loser_elo + k_loser * (0 - (1 - expected_winner))

        # Apply updates
        winner_rating.set_surface_rating(surface, new_winner_elo)
        loser_rating.set_surface_rating(surface, new_loser_elo)

        # Update tracking
        winner_rating.matches_played += 1
        loser_rating.matches_played += 1
        winner_rating.last_match_date = match_date
        loser_rating.last_match_date = match_date

        # Add to recent results
        winner_rating.add_result(match_date, True, loser_elo, surface)
        loser_rating.add_result(match_date, False, winner_elo, surface)

        return new_winner_elo, new_loser_elo

    def predict_match(
        self,
        player_a: str,
        player_b: str,
        surface: str,
        include_form: bool = True,
        match_date: datetime = None
    ) -> Dict[str, float]:
        """
        Predict match outcome probabilities with form adjustment.
        """
        if match_date is None:
            match_date = datetime.now()

        rating_a = self.get_player_rating(player_a)
        rating_b = self.get_player_rating(player_b)

        # Apply inactivity decay
        self.apply_inactivity_decay(player_a, match_date)
        self.apply_inactivity_decay(player_b, match_date)

        # Base surface ratings
        elo_a = rating_a.get_surface_rating(surface)
        elo_b = rating_b.get_surface_rating(surface)

        # Apply form adjustment if requested
        if include_form:
            form_a = self.calculate_form_adjustment(player_a, surface)
            form_b = self.calculate_form_adjustment(player_b, surface)

            # Blend form into prediction (not into stored rating)
            effective_elo_a = elo_a + form_a * self.form_weight
            effective_elo_b = elo_b + form_b * self.form_weight
        else:
            effective_elo_a = elo_a
            effective_elo_b = elo_b
            form_a = form_b = 0

        prob_a = self.expected_score(effective_elo_a, effective_elo_b)

        return {
            player_a: prob_a,
            player_b: 1 - prob_a,
            'elo_diff': elo_a - elo_b,
            f'{player_a}_elo': elo_a,
            f'{player_b}_elo': elo_b,
            f'{player_a}_form': form_a,
            f'{player_b}_form': form_b,
            f'{player_a}_matches': rating_a.matches_played,
            f'{player_b}_matches': rating_b.matches_played
        }

    def process_matches(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a dataframe of historical matches to build ratings.
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

            # Get pre-match prediction
            pred = self.predict_match(winner, loser, surface, match_date=match_date)
            expected = pred[winner]

            # Update ratings
            self.update_ratings(winner, loser, surface, tourney_level, match_date)

            results.append({
                'winner': winner,
                'loser': loser,
                'surface': surface,
                'winner_pre_elo': pred[f'{winner}_elo'],
                'loser_pre_elo': pred[f'{loser}_elo'],
                'predicted_winner_prob': expected,
                'prediction_correct': expected > 0.5,
                'winner_form': pred.get(f'{winner}_form', 0),
                'loser_form': pred.get(f'{loser}_form', 0)
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
                'grass': rating.grass,
                'form': rating.form_adjustment,
                'last_match': rating.last_match_date
            })

        df = pd.DataFrame(data)
        if len(df) > 0:
            return df.nlargest(n, 'elo')
        return df
