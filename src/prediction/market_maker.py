"""
Market Maker for Tennis Odds.

Generates fair prices for Moneyline, Spreads, and Totals using Monte Carlo simulation.
Identifies value in niche markets.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from .simulation import MatchSimulator, get_implied_serve_prob

class MarketMaker:
    def __init__(self):
        pass

    def estimate_serve_probs(
        self,
        p1_stats: Dict,
        p2_stats: Dict,
        surface: str
    ) -> Tuple[float, float]:
        """
        Estimate point-on-serve probabilities for both players.

        Uses a weighted average approach:
        - Player's service points won % (their strength on serve)
        - Opponent's return points lost % (100% - their return won %)

        The final serve probability blends these two factors.
        """
        base_serve = get_implied_serve_prob(surface, 'atp')
        base_return = 1 - base_serve

        # Player A Serve
        s_a = p1_stats.get('service_points_won_pct', base_serve)
        if s_a == 0:  # Fallback if no data
            s_a = base_serve

        # Player B Return - get their return ability
        r_b = p2_stats.get('return_points_won_pct', base_return)
        if r_b == 0:
            r_b = base_return

        # Blend: A's serve strength vs B's return weakness
        # p_serve_a = weighted avg of (A serve%) and (100% - B return%)
        # This balances serve dominance against return skill
        p_serve_a = 0.6 * s_a + 0.4 * (1 - r_b)

        # Player B Serve
        s_b = p2_stats.get('service_points_won_pct', base_serve)
        if s_b == 0:
            s_b = base_serve

        r_a = p1_stats.get('return_points_won_pct', base_return)
        if r_a == 0:
            r_a = base_return

        p_serve_b = 0.6 * s_b + 0.4 * (1 - r_a)

        # Bound probabilities to realistic ATP ranges
        p_serve_a = max(0.55, min(0.80, p_serve_a))
        p_serve_b = max(0.55, min(0.80, p_serve_b))

        return p_serve_a, p_serve_b

    def make_market(
        self,
        player_a: str,
        player_b: str,
        surface: str,
        p1_stats: Dict,
        p2_stats: Dict,
        best_of: int = 3,
        n_sims: int = 5000
    ) -> Dict:
        """
        Generate fair lines for a match.
        """
        # 1. Estimate Serve Probs
        pa, pb = self.estimate_serve_probs(p1_stats, p2_stats, surface)
        
        # 2. Run Simulation
        sim = MatchSimulator(pa, pb, best_of)
        res = sim.run(n_sims)
        
        # 3. Calculate Derivative Markets
        
        # -- Moneyline --
        ml_prob = res.winner_a_prob
        fair_odds_a = 1 / ml_prob if ml_prob > 0 else 999
        fair_odds_b = 1 / (1 - ml_prob) if ml_prob < 1 else 999
        
        # -- Game Spread --
        # Find the spread X where P(Diff > X) approx 0.5
        # We scan the distribution
        spreads = sorted(res.game_diff_dist.keys())
        cum_prob = 0
        fair_spread = 0
        for s in spreads:
            prob = res.game_diff_dist[s]
            if cum_prob + prob >= 0.5:
                fair_spread = s
                break
            cum_prob += prob
            
        # -- Totals --
        totals = sorted(res.total_games_dist.keys())
        cum_prob_t = 0
        fair_total = 0
        for t in totals:
            prob = res.total_games_dist[t]
            if cum_prob_t + prob >= 0.5:
                fair_total = t
                break
            cum_prob_t += prob
            
        # -- Set Betting --
        # Handle both best-of-3 and best-of-5 formats
        sets_to_win = (best_of // 2) + 1
        if best_of == 3:
            set_scores = {
                '2-0': res.set_score_dist.get('2-0', 0),
                '2-1': res.set_score_dist.get('2-1', 0),
                '0-2': res.set_score_dist.get('0-2', 0),
                '1-2': res.set_score_dist.get('1-2', 0)
            }
        else:  # best_of == 5
            set_scores = {
                '3-0': res.set_score_dist.get('3-0', 0),
                '3-1': res.set_score_dist.get('3-1', 0),
                '3-2': res.set_score_dist.get('3-2', 0),
                '0-3': res.set_score_dist.get('0-3', 0),
                '1-3': res.set_score_dist.get('1-3', 0),
                '2-3': res.set_score_dist.get('2-3', 0)
            }

        return {
            'match': f"{player_a} vs {player_b}",
            'surface': surface,
            'serve_probs': (pa, pb),
            'moneyline': {
                'prob_a': ml_prob,
                'fair_odds_a': round(fair_odds_a, 2),
                'fair_odds_b': round(fair_odds_b, 2)
            },
            'spread': {
                'fair_handicap_a': fair_spread, # e.g. +3 or -2
                'prob_cover_minus_2_5': self._calc_prob_greater(res.game_diff_dist, 2.5),
                'prob_cover_plus_2_5': self._calc_prob_greater(res.game_diff_dist, -2.5),
            },
            'total': {
                'fair_line': fair_total,
                'median': res.median_total_games,
                'prob_over_21_5': self._calc_prob_greater_total(res.total_games_dist, 21.5),
                'prob_over_22_5': self._calc_prob_greater_total(res.total_games_dist, 22.5),
            },
            'sets': set_scores
        }

    def _calc_prob_greater(self, dist: Dict[int, float], threshold: float) -> float:
        """Calculate probability that value > threshold."""
        prob = 0.0
        for val, p in dist.items():
            if val > threshold:
                prob += p
        return prob

    def _calc_prob_greater_total(self, dist: Dict[int, float], threshold: float) -> float:
        """Calculate probability that total > threshold."""
        prob = 0.0
        for val, p in dist.items():
            if val > threshold:
                prob += p
        return prob
