"""
Ensemble prediction system combining all prediction signals.

Combines:
1. Base rating (Elo or Glicko-2)
2. Serve/Return differential
3. Head-to-head adjustment
4. Fatigue factors
5. Surface adjustment
"""

import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from ..ratings.elo_enhanced import EnhancedElo
from ..ratings.glicko2 import Glicko2System
from ..features.serve_return import ServeReturnTracker
from ..features.h2h import H2HAnalyzer
from ..features.fatigue import FatigueTracker
from ..features.surface import SurfaceAnalyzer


@dataclass
class PredictionBreakdown:
    """Detailed breakdown of a prediction."""
    base_prob: float
    serve_prob: float
    h2h_prob: float
    fatigue_adj_a: float
    fatigue_adj_b: float
    surface_adj_a: float
    surface_adj_b: float
    final_prob: float
    confidence: float


class EnsemblePredictor:
    """
    Combine all prediction signals into a final probability.
    """

    def __init__(
        self,
        elo_system: EnhancedElo = None,
        glicko_system: Glicko2System = None,
        serve_tracker: ServeReturnTracker = None,
        h2h_analyzer: H2HAnalyzer = None,
        fatigue_tracker: FatigueTracker = None,
        surface_analyzer: SurfaceAnalyzer = None,
        use_glicko: bool = False,
        weights: Dict[str, float] = None
    ):
        """
        Args:
            elo_system: Enhanced Elo rating system
            glicko_system: Glicko-2 rating system
            serve_tracker: Serve/return statistics tracker
            h2h_analyzer: Head-to-head analyzer
            fatigue_tracker: Fatigue tracker
            surface_analyzer: Surface analyzer
            use_glicko: If True, use Glicko-2 as base; else use Elo
            weights: Component weights (must sum to 1.0)
        """
        self.elo = elo_system or EnhancedElo()
        self.glicko = glicko_system or Glicko2System()
        self.serve = serve_tracker or ServeReturnTracker()
        self.h2h = h2h_analyzer or H2HAnalyzer()
        self.fatigue = fatigue_tracker or FatigueTracker()
        self.surface = surface_analyzer or SurfaceAnalyzer()
        self.use_glicko = use_glicko

        # Default weights
        self.weights = weights or {
            'base_rating': 0.50,
            'serve_return': 0.15,
            'h2h': 0.15,
            'fatigue': 0.10,
            'surface': 0.10
        }

    def set_weights(self, weights: Dict[str, float]):
        """Update component weights."""
        self.weights = weights

    def predict(
        self,
        player_a: str,
        player_b: str,
        surface: str,
        tournament: str = None,
        match_date: datetime = None,
        tournament_location: str = None
    ) -> Dict[str, Any]:
        """
        Generate prediction with full breakdown.

        Args:
            player_a: First player name
            player_b: Second player name
            surface: Match surface (Hard/Clay/Grass)
            tournament: Tournament name (optional)
            match_date: Date of match (optional, defaults to now)
            tournament_location: IOC code for location (optional)

        Returns:
            Dictionary with probabilities and breakdown
        """
        if match_date is None:
            match_date = datetime.now()

        # 1. Base rating probability
        if self.use_glicko:
            base_pred = self.glicko.predict_match(player_a, player_b, surface)
            base_prob_a = base_pred[player_a]
            confidence = base_pred.get('confidence', 0.5)
            rating_a = base_pred.get(f'{player_a}_rating', 1500)
            rating_b = base_pred.get(f'{player_b}_rating', 1500)
        else:
            base_pred = self.elo.predict_match(player_a, player_b, surface, match_date=match_date)
            base_prob_a = base_pred[player_a]
            confidence = 0.7  # Elo doesn't track uncertainty
            rating_a = base_pred.get(f'{player_a}_elo', 1500)
            rating_b = base_pred.get(f'{player_b}_elo', 1500)

        # 2. Serve/Return differential
        serve_diff = self.serve.calculate_serve_dominance(player_a, player_b, surface)
        # Convert differential to probability (sigmoid-like scaling)
        serve_prob_a = 0.5 + serve_diff * 2  # Scale factor
        serve_prob_a = max(0.2, min(0.8, serve_prob_a))  # Bound

        # 3. H2H adjustment
        h2h_prob_a = self.h2h.calculate_h2h_adjustment(
            player_a, player_b, surface, base_prob_a
        )

        # 4. Fatigue penalties
        fatigue_a = self.fatigue.calculate_fatigue_penalty(
            player_a, match_date, tournament, tournament_location
        )
        fatigue_b = self.fatigue.calculate_fatigue_penalty(
            player_b, match_date, tournament, tournament_location
        )
        # Net fatigue effect: positive if B is more tired
        fatigue_diff = fatigue_b - fatigue_a

        # 5. Surface adjustment
        surface_adj_a = self.surface.calculate_surface_adjustment(player_a, surface)
        surface_adj_b = self.surface.calculate_surface_adjustment(player_b, surface)
        surface_diff = surface_adj_a - surface_adj_b

        # Combine signals with weights
        w = self.weights

        # For fatigue and surface, we adjust the base probability
        fatigue_adjusted_prob = base_prob_a + fatigue_diff
        surface_adjusted_prob = base_prob_a + surface_diff

        combined_prob_a = (
            w['base_rating'] * base_prob_a +
            w['serve_return'] * serve_prob_a +
            w['h2h'] * h2h_prob_a +
            w['fatigue'] * fatigue_adjusted_prob +
            w['surface'] * surface_adjusted_prob
        )

        # Normalize and bound
        final_prob_a = max(0.05, min(0.95, combined_prob_a))

        return {
            player_a: final_prob_a,
            player_b: 1 - final_prob_a,
            'confidence': confidence,
            'breakdown': {
                'base_rating': base_prob_a,
                'serve_return': serve_prob_a,
                'h2h': h2h_prob_a,
                'fatigue_a': fatigue_a,
                'fatigue_b': fatigue_b,
                'surface_adj_a': surface_adj_a,
                'surface_adj_b': surface_adj_b
            },
            f'{player_a}_rating': rating_a,
            f'{player_b}_rating': rating_b,
            'rating_diff': rating_a - rating_b
        }

    def process_matches(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process historical matches to build all component models.

        Returns accuracy statistics.
        """
        # Sort by date
        if 'tourney_date' in matches_df.columns:
            matches_df = matches_df.sort_values('tourney_date')

        results = []

        for idx, match in matches_df.iterrows():
            winner = match['winner_name']
            loser = match['loser_name']
            surface = match.get('surface', 'Hard')
            tourney_level = match.get('tourney_level', 'default')
            tournament = match.get('tourney_name', '')

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

            # Get pre-match prediction (using current state before update)
            pred = self.predict(winner, loser, surface, tournament, match_date)
            expected = pred[winner]

            results.append({
                'winner': winner,
                'loser': loser,
                'surface': surface,
                'tournament': tournament,
                'date': match_date,
                'predicted_winner_prob': expected,
                'prediction_correct': expected > 0.5,
                'confidence': pred.get('confidence', 0),
                **{f'breakdown_{k}': v for k, v in pred.get('breakdown', {}).items()}
            })

            # Update all component models
            self.elo.update_ratings(winner, loser, surface, tourney_level, match_date)
            self.glicko.update_single_match(winner, loser, surface, match_date)
            self.serve.process_match(match.to_dict())
            self.fatigue.process_match(match.to_dict())
            self.surface.update_player_surface(winner, surface, match_date, True)
            self.surface.update_player_surface(loser, surface, match_date, False)

        return pd.DataFrame(results)

    def initialize_from_matches(self, matches_df: pd.DataFrame):
        """Initialize all components from historical data without tracking results."""
        # Build H2H cache
        self.h2h.set_matches(matches_df)

        # Build surface profiles
        self.surface.process_matches_df(matches_df)

        # Process serve stats
        self.serve.process_matches_df(matches_df)

        # Process fatigue (most recent data matters most)
        self.fatigue.process_matches_df(matches_df)

        # Build ratings
        if 'tourney_date' in matches_df.columns:
            matches_df = matches_df.sort_values('tourney_date')

        for _, match in matches_df.iterrows():
            winner = match['winner_name']
            loser = match['loser_name']
            surface = match.get('surface', 'Hard')
            tourney_level = match.get('tourney_level', 'default')

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

            self.elo.update_ratings(winner, loser, surface, tourney_level, match_date)
            self.glicko.update_single_match(winner, loser, surface, match_date)
