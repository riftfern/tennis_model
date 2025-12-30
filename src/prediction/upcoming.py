"""
Upcoming match predictor for live betting opportunities.

Fetches live odds and generates predictions with +EV detection.
"""

import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

from .ensemble import EnsemblePredictor
from ..odds_scraper import OddsAPIClient


def calculate_ev(model_prob: float, decimal_odds: float) -> float:
    """Calculate expected value percentage."""
    profit_if_win = decimal_odds - 1
    prob_lose = 1 - model_prob
    ev = (model_prob * profit_if_win) - (prob_lose * 1)
    return ev * 100


def implied_probability(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    return 1 / decimal_odds


class UpcomingMatchPredictor:
    """Generate predictions for upcoming tournament matches."""

    def __init__(
        self,
        ensemble: EnsemblePredictor,
        odds_client: OddsAPIClient = None
    ):
        """
        Args:
            ensemble: Trained ensemble predictor
            odds_client: Odds API client (optional, provide key at prediction time)
        """
        self.ensemble = ensemble
        self.odds_client = odds_client

    def set_odds_client(self, api_key: str):
        """Set up odds client with API key."""
        self.odds_client = OddsAPIClient(api_key)

    def get_predictions(
        self,
        tournament_name: str = None,
        surface: str = 'Hard',
        min_ev: float = 2.0
    ) -> pd.DataFrame:
        """
        Fetch upcoming matches and generate predictions.

        Args:
            tournament_name: Filter by tournament (optional)
            surface: Default surface if not detected
            min_ev: Minimum EV% to flag as value bet

        Returns:
            DataFrame with predictions and value bets
        """
        if self.odds_client is None:
            print("No odds client configured. Use set_odds_client(api_key) first.")
            return pd.DataFrame()

        # Get live odds
        odds_data = self.odds_client.get_tennis_odds()
        if not odds_data:
            print("No upcoming matches found or unable to fetch odds.")
            return pd.DataFrame()

        matches_df = self.odds_client.parse_odds_response(odds_data)

        if len(matches_df) == 0:
            print("No matches with odds available.")
            return pd.DataFrame()

        predictions = []

        for _, match in matches_df.iterrows():
            player_a = match['player_a']
            player_b = match['player_b']
            odds_a = match['odds_a']
            odds_b = match['odds_b']

            # Generate prediction
            pred = self.ensemble.predict(
                player_a, player_b, surface,
                tournament=tournament_name or 'Unknown',
                match_date=datetime.now()
            )

            prob_a = pred[player_a]
            prob_b = pred[player_b]

            # Calculate EV for both sides
            ev_a = calculate_ev(prob_a, odds_a)
            ev_b = calculate_ev(prob_b, odds_b)

            # Implied probabilities
            implied_a = implied_probability(odds_a)
            implied_b = implied_probability(odds_b)

            # Edge (model prob - implied prob)
            edge_a = prob_a - implied_a
            edge_b = prob_b - implied_b

            predictions.append({
                'player_a': player_a,
                'player_b': player_b,
                'prob_a': prob_a,
                'prob_b': prob_b,
                'odds_a': odds_a,
                'odds_b': odds_b,
                'implied_a': implied_a,
                'implied_b': implied_b,
                'ev_a': ev_a,
                'ev_b': ev_b,
                'edge_a': edge_a,
                'edge_b': edge_b,
                'confidence': pred.get('confidence', 0),
                'rating_a': pred.get(f'{player_a}_rating', 0),
                'rating_b': pred.get(f'{player_b}_rating', 0),
                'best_bet': player_a if ev_a > ev_b else player_b,
                'best_ev': max(ev_a, ev_b),
                'best_edge': max(edge_a, edge_b),
                'is_value': max(ev_a, ev_b) >= min_ev,
                'breakdown': pred.get('breakdown', {})
            })

        df = pd.DataFrame(predictions)

        # Sort by best EV
        if len(df) > 0:
            df = df.sort_values('best_ev', ascending=False)

        return df

    def get_value_bets(
        self,
        tournament_name: str = None,
        surface: str = 'Hard',
        min_ev: float = 2.0,
        min_edge: float = 0.03,
        min_confidence: float = 0.4
    ) -> pd.DataFrame:
        """
        Get only value betting opportunities.

        Args:
            tournament_name: Filter by tournament
            surface: Match surface
            min_ev: Minimum EV%
            min_edge: Minimum edge (probability difference)
            min_confidence: Minimum prediction confidence

        Returns:
            DataFrame with value bets only
        """
        all_preds = self.get_predictions(tournament_name, surface, min_ev)

        if len(all_preds) == 0:
            return pd.DataFrame()

        # Filter for value bets
        value_bets = all_preds[
            (all_preds['best_ev'] >= min_ev) &
            (all_preds['best_edge'] >= min_edge) &
            (all_preds['confidence'] >= min_confidence)
        ]

        return value_bets

    def print_predictions(
        self,
        predictions: pd.DataFrame,
        show_breakdown: bool = False
    ):
        """Pretty-print predictions."""
        print("\n" + "=" * 70)
        print("UPCOMING MATCH PREDICTIONS")
        print("=" * 70)

        if len(predictions) == 0:
            print("\nNo predictions to display.")
            return

        for _, row in predictions.iterrows():
            print(f"\n{row['player_a']} vs {row['player_b']}")
            print(f"  Model:   {row['player_a']}: {row['prob_a']*100:.1f}%  |  "
                  f"{row['player_b']}: {row['prob_b']*100:.1f}%")
            print(f"  Odds:    {row['odds_a']:.2f}  |  {row['odds_b']:.2f}")
            print(f"  Implied: {row['implied_a']*100:.1f}%  |  {row['implied_b']*100:.1f}%")

            # Show value bet if applicable
            if row['ev_a'] > 2:
                print(f"  >>> VALUE: {row['player_a']} @ {row['odds_a']:.2f} "
                      f"(EV: +{row['ev_a']:.1f}%, Edge: {row['edge_a']*100:.1f}%)")
            if row['ev_b'] > 2:
                print(f"  >>> VALUE: {row['player_b']} @ {row['odds_b']:.2f} "
                      f"(EV: +{row['ev_b']:.1f}%, Edge: {row['edge_b']*100:.1f}%)")

            if show_breakdown and 'breakdown' in row and row['breakdown']:
                print(f"  Breakdown:")
                bd = row['breakdown']
                print(f"    Base rating: {bd.get('base_rating', 0)*100:.1f}%")
                print(f"    Serve/Return: {bd.get('serve_return', 0)*100:.1f}%")
                print(f"    H2H: {bd.get('h2h', 0)*100:.1f}%")
                print(f"    Fatigue: A={bd.get('fatigue_a', 0)*100:.1f}%, B={bd.get('fatigue_b', 0)*100:.1f}%")

        print("\n" + "=" * 70)

    def predict_manual(
        self,
        player_a: str,
        player_b: str,
        surface: str,
        odds_a: float = None,
        odds_b: float = None,
        tournament: str = None
    ) -> Dict:
        """
        Generate prediction for a manually specified match.

        Args:
            player_a: First player name
            player_b: Second player name
            surface: Match surface
            odds_a: Decimal odds for player A (optional)
            odds_b: Decimal odds for player B (optional)
            tournament: Tournament name (optional)

        Returns:
            Prediction dictionary
        """
        pred = self.ensemble.predict(
            player_a, player_b, surface,
            tournament=tournament or 'Manual',
            match_date=datetime.now()
        )

        result = {
            'player_a': player_a,
            'player_b': player_b,
            'prob_a': pred[player_a],
            'prob_b': pred[player_b],
            'confidence': pred.get('confidence', 0),
            'rating_a': pred.get(f'{player_a}_rating', 0),
            'rating_b': pred.get(f'{player_b}_rating', 0),
            'breakdown': pred.get('breakdown', {})
        }

        if odds_a:
            result['odds_a'] = odds_a
            result['ev_a'] = calculate_ev(pred[player_a], odds_a)
            result['edge_a'] = pred[player_a] - implied_probability(odds_a)

        if odds_b:
            result['odds_b'] = odds_b
            result['ev_b'] = calculate_ev(pred[player_b], odds_b)
            result['edge_b'] = pred[player_b] - implied_probability(odds_b)

        return result

    def print_single_prediction(self, pred: Dict):
        """Print a single prediction nicely."""
        print(f"\n{pred['player_a']} vs {pred['player_b']}")
        print("-" * 50)
        print(f"Model probability:")
        print(f"  {pred['player_a']}: {pred['prob_a']*100:.1f}%")
        print(f"  {pred['player_b']}: {pred['prob_b']*100:.1f}%")
        print(f"Confidence: {pred['confidence']*100:.0f}%")
        print(f"Ratings: {pred['rating_a']:.0f} vs {pred['rating_b']:.0f}")

        if 'odds_a' in pred:
            print(f"\n{pred['player_a']} @ {pred['odds_a']:.2f}:")
            print(f"  Implied: {implied_probability(pred['odds_a'])*100:.1f}%")
            print(f"  Edge: {pred['edge_a']*100:.1f}%")
            print(f"  EV: {pred['ev_a']:+.1f}%")
            if pred['ev_a'] > 0:
                print(f"  >>> VALUE BET!")

        if 'odds_b' in pred:
            print(f"\n{pred['player_b']} @ {pred['odds_b']:.2f}:")
            print(f"  Implied: {implied_probability(pred['odds_b'])*100:.1f}%")
            print(f"  Edge: {pred['edge_b']*100:.1f}%")
            print(f"  EV: {pred['ev_b']:+.1f}%")
            if pred['ev_b'] > 0:
                print(f"  >>> VALUE BET!")
