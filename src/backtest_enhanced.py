"""
Enhanced backtester for comparing model variants.

Compares:
1. Baseline Elo
2. Enhanced Elo
3. Glicko-2
4. Elo + Serve/Return
5. Elo + H2H
6. Full Ensemble (Elo)
7. Full Ensemble (Glicko-2)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from tqdm import tqdm

from .elo import TennisElo
from .ratings.elo_enhanced import EnhancedElo
from .ratings.glicko2 import Glicko2System
from .features.serve_return import ServeReturnTracker
from .features.h2h import H2HAnalyzer
from .features.fatigue import FatigueTracker
from .features.surface import SurfaceAnalyzer
from .prediction.ensemble import EnsemblePredictor


@dataclass
class BacktestResult:
    """Results from a single backtest."""
    model_name: str
    total_bets: int = 0
    wins: int = 0
    losses: int = 0
    total_staked: float = 0.0
    total_profit: float = 0.0
    roi_percent: float = 0.0
    win_rate: float = 0.0
    avg_odds: float = 0.0
    avg_ev: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    accuracy: float = 0.0  # Prediction accuracy (not just bets)
    log_loss: float = 0.0  # Calibration metric


def calculate_ev(model_prob: float, decimal_odds: float) -> float:
    """Calculate expected value percentage."""
    profit_if_win = decimal_odds - 1
    prob_lose = 1 - model_prob
    ev = (model_prob * profit_if_win) - (prob_lose * 1)
    return ev * 100


def implied_probability(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if decimal_odds <= 1:
        return 1.0
    return 1 / decimal_odds


class ModelVariant:
    """Wrapper for a model variant to standardize interface."""

    def __init__(self, name: str, model_type: str, **kwargs):
        self.name = name
        self.model_type = model_type
        self.kwargs = kwargs

        # Initialize based on type
        if model_type == 'baseline_elo':
            self.model = TennisElo()
        elif model_type == 'enhanced_elo':
            self.model = EnhancedElo()
        elif model_type == 'glicko2':
            self.model = Glicko2System()
        elif model_type == 'ensemble_elo':
            self.model = EnsemblePredictor(use_glicko=False)
        elif model_type == 'ensemble_glicko':
            self.model = EnsemblePredictor(use_glicko=True)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def predict(self, player_a: str, player_b: str, surface: str, **kwargs) -> float:
        """Get win probability for player_a."""
        if self.model_type in ['baseline_elo', 'enhanced_elo']:
            pred = self.model.predict_match(player_a, player_b, surface)
            return pred[player_a]
        elif self.model_type == 'glicko2':
            pred = self.model.predict_match(player_a, player_b, surface)
            return pred[player_a]
        elif self.model_type in ['ensemble_elo', 'ensemble_glicko']:
            pred = self.model.predict(player_a, player_b, surface, **kwargs)
            return pred[player_a]
        return 0.5

    def update(self, winner: str, loser: str, surface: str, match_data: dict, match_date: datetime):
        """Update model after a match."""
        tourney_level = match_data.get('tourney_level', 'default')

        if self.model_type == 'baseline_elo':
            self.model.update_ratings(winner, loser, surface, tourney_level)
        elif self.model_type == 'enhanced_elo':
            self.model.update_ratings(winner, loser, surface, tourney_level, match_date)
        elif self.model_type == 'glicko2':
            self.model.update_single_match(winner, loser, surface, match_date)
        elif self.model_type in ['ensemble_elo', 'ensemble_glicko']:
            # Update all components
            self.model.elo.update_ratings(winner, loser, surface, tourney_level, match_date)
            self.model.glicko.update_single_match(winner, loser, surface, match_date)
            self.model.serve.process_match(match_data)
            self.model.fatigue.process_match(match_data)
            self.model.surface.update_player_surface(winner, surface, match_date, True)
            self.model.surface.update_player_surface(loser, surface, match_date, False)


class EnhancedBacktester:
    """
    Run backtests comparing multiple model variants.
    """

    def __init__(
        self,
        min_ev: float = 3.0,
        min_odds: float = 1.20,
        max_odds: float = 5.00,
        min_prob: float = 0.15,
        stake_size: float = 100.0,
        odds_column: str = 'Avg'
    ):
        self.min_ev = min_ev
        self.min_odds = min_odds
        self.max_odds = max_odds
        self.min_prob = min_prob
        self.stake_size = stake_size
        self.odds_column = odds_column

    def create_model_variants(self, matches_df: pd.DataFrame) -> Dict[str, ModelVariant]:
        """Create all model variants for comparison."""
        variants = {
            'Baseline Elo': ModelVariant('Baseline Elo', 'baseline_elo'),
            'Enhanced Elo': ModelVariant('Enhanced Elo', 'enhanced_elo'),
            'Glicko-2': ModelVariant('Glicko-2', 'glicko2'),
            'Ensemble (Elo)': ModelVariant('Ensemble (Elo)', 'ensemble_elo'),
            'Ensemble (Glicko)': ModelVariant('Ensemble (Glicko)', 'ensemble_glicko'),
        }

        # Initialize H2H analyzer for ensembles
        for name in ['Ensemble (Elo)', 'Ensemble (Glicko)']:
            variants[name].model.h2h.set_matches(matches_df)

        return variants

    def run_single_backtest(
        self,
        model: ModelVariant,
        matches: pd.DataFrame,
        verbose: bool = True
    ) -> BacktestResult:
        """Run backtest for a single model variant."""
        # Determine odds columns
        winner_odds_col = f'{self.odds_column}W'
        loser_odds_col = f'{self.odds_column}L'

        if winner_odds_col not in matches.columns:
            # Try alternatives
            for prefix in ['Avg', 'Max', 'B365', 'PS']:
                if f'{prefix}W' in matches.columns:
                    winner_odds_col = f'{prefix}W'
                    loser_odds_col = f'{prefix}L'
                    break

        # Sort by date
        matches = matches.copy()
        if 'tourney_date' in matches.columns:
            matches = matches.sort_values('tourney_date')

        # Remove matches without odds
        matches = matches.dropna(subset=[winner_odds_col, loser_odds_col])

        result = BacktestResult(model_name=model.name)
        bankroll = 10000.0
        peak_bankroll = bankroll
        returns = []

        # Track prediction accuracy
        correct_predictions = 0
        total_predictions = 0
        log_loss_sum = 0.0

        iterator = tqdm(matches.iterrows(), total=len(matches), desc=model.name) if verbose else matches.iterrows()

        for _, match in iterator:
            winner = match['winner_name']
            loser = match['loser_name']
            surface = match.get('surface', 'Hard')

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

            winner_odds = match[winner_odds_col]
            loser_odds = match[loser_odds_col]

            # Skip invalid odds
            if pd.isna(winner_odds) or pd.isna(loser_odds):
                model.update(winner, loser, surface, match.to_dict(), match_date)
                continue
            if winner_odds < 1.01 or loser_odds < 1.01:
                model.update(winner, loser, surface, match.to_dict(), match_date)
                continue

            # Get prediction
            prob_winner = model.predict(winner, loser, surface,
                                       tournament=match.get('tourney_name', ''),
                                       match_date=match_date)
            prob_loser = 1 - prob_winner

            # Track accuracy
            total_predictions += 1
            if prob_winner > 0.5:
                correct_predictions += 1

            # Log loss (for calibration)
            epsilon = 1e-15
            prob_winner_clipped = max(epsilon, min(1-epsilon, prob_winner))
            log_loss_sum += -np.log(prob_winner_clipped)

            # Check for value bets on both sides
            for player, prob, odds, won in [
                (winner, prob_winner, winner_odds, True),
                (loser, prob_loser, loser_odds, False)
            ]:
                if prob < self.min_prob:
                    continue
                if odds < self.min_odds or odds > self.max_odds:
                    continue

                ev = calculate_ev(prob, odds)

                if ev >= self.min_ev:
                    stake = self.stake_size

                    if won:
                        profit = stake * (odds - 1)
                        result.wins += 1
                    else:
                        profit = -stake
                        result.losses += 1

                    # Track
                    bankroll += profit
                    peak_bankroll = max(peak_bankroll, bankroll)
                    drawdown = peak_bankroll - bankroll
                    result.max_drawdown = max(result.max_drawdown, drawdown)

                    returns.append(profit / stake)

                    result.total_bets += 1
                    result.total_staked += stake
                    result.total_profit += profit
                    result.avg_odds = (result.avg_odds * (result.total_bets - 1) + odds) / result.total_bets
                    result.avg_ev = (result.avg_ev * (result.total_bets - 1) + ev) / result.total_bets

            # Update model after processing
            model.update(winner, loser, surface, match.to_dict(), match_date)

        # Calculate final metrics
        if result.total_bets > 0:
            result.roi_percent = (result.total_profit / result.total_staked) * 100
            result.win_rate = (result.wins / result.total_bets) * 100

            if returns and len(returns) > 1:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                if std_return > 0:
                    result.sharpe_ratio = avg_return / std_return * np.sqrt(252)  # Annualized

        if total_predictions > 0:
            result.accuracy = correct_predictions / total_predictions * 100
            result.log_loss = log_loss_sum / total_predictions

        return result

    def run_comparison(
        self,
        matches: pd.DataFrame,
        verbose: bool = True
    ) -> Dict[str, BacktestResult]:
        """Run backtests for all model variants and compare."""
        print("\n" + "=" * 70)
        print("MODEL COMPARISON BACKTEST")
        print("=" * 70)
        print(f"Settings: Min EV={self.min_ev}%, Odds={self.min_odds}-{self.max_odds}")
        print(f"Matches: {len(matches):,}")
        print("=" * 70)

        variants = self.create_model_variants(matches)
        results = {}

        for name, model in variants.items():
            print(f"\n--- {name} ---")
            results[name] = self.run_single_backtest(model, matches, verbose)

        return results

    def print_comparison(self, results: Dict[str, BacktestResult]):
        """Print comparison table."""
        print("\n" + "=" * 90)
        print("COMPARISON RESULTS")
        print("=" * 90)
        print(f"{'Model':<20} {'Bets':<8} {'ROI':>8} {'Win%':>8} {'Accuracy':>10} {'MaxDD':>10} {'Sharpe':>8}")
        print("-" * 90)

        sorted_results = sorted(results.items(), key=lambda x: -x[1].roi_percent)

        for name, res in sorted_results:
            print(f"{name:<20} {res.total_bets:<8} "
                  f"{res.roi_percent:>+7.2f}% "
                  f"{res.win_rate:>7.1f}% "
                  f"{res.accuracy:>9.1f}% "
                  f"${res.max_drawdown:>8,.0f} "
                  f"{res.sharpe_ratio:>7.2f}")

        print("=" * 90)

        # Best model
        best = sorted_results[0]
        print(f"\nBest model: {best[0]} with ROI {best[1].roi_percent:+.2f}%")

    def run_surface_comparison(
        self,
        matches: pd.DataFrame,
        verbose: bool = False
    ) -> Dict[str, Dict[str, BacktestResult]]:
        """Run comparison broken down by surface."""
        surfaces = ['Hard', 'Clay', 'Grass']
        results = {}

        for surface in surfaces:
            surface_matches = matches[matches['surface'] == surface]
            if len(surface_matches) > 100:
                print(f"\n=== {surface.upper()} COURT ({len(surface_matches):,} matches) ===")
                results[surface] = self.run_comparison(surface_matches, verbose)
                self.print_comparison(results[surface])

        return results


def quick_backtest(
    matches_df: pd.DataFrame,
    min_ev: float = 3.0,
    min_odds: float = 1.20,
    max_odds: float = 4.0,
    verbose: bool = True
) -> Dict[str, BacktestResult]:
    """
    Quick function to run model comparison backtest.

    Args:
        matches_df: DataFrame with matches and odds
        min_ev: Minimum EV% threshold
        min_odds: Minimum odds to bet
        max_odds: Maximum odds to bet
        verbose: Show progress

    Returns:
        Dictionary of model name -> BacktestResult
    """
    bt = EnhancedBacktester(
        min_ev=min_ev,
        min_odds=min_odds,
        max_odds=max_odds
    )

    results = bt.run_comparison(matches_df, verbose)
    bt.print_comparison(results)

    return results
