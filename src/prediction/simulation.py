"""
Monte Carlo Simulation Engine for Tennis Matches.

Converts point-level probabilities into match, set, game, and handicap probabilities
to price derivative markets (Spreads, Totals).
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import random

@dataclass
class SimulationResult:
    winner_a_prob: float
    expected_total_games: float
    total_games_dist: Dict[int, float]
    game_diff_dist: Dict[int, float]  # (Games A - Games B)
    set_score_dist: Dict[str, float]  # e.g., "2-0": 0.45
    median_total_games: float
    median_game_diff: int

class MatchSimulator:
    def __init__(self, p_serve_a: float, p_serve_b: float, best_of: int = 3):
        """
        Args:
            p_serve_a: Probability Player A wins point on serve
            p_serve_b: Probability Player B wins point on serve
            best_of: 3 or 5 sets
        """
        self.p_serve_a = p_serve_a
        self.p_serve_b = p_serve_b
        self.best_of = best_of
        self.sets_to_win = (best_of // 2) + 1

    def simulate_game(self, server_prob: float) -> Tuple[bool, int, int]:
        """
        Simulate a single game.
        Returns: (server_won, points_server, points_receiver)
        Note: Simplified to just return winner and approximate score impact is negligible 
        for totals, but for exact points we'd need more detail. 
        For game totals/spreads, we just need who won the game.
        """
        # We can use the analytical solution for a game to save compute
        # P(win game) = p^4 * (1 + 4(1-p) + 10(1-p)^2) / (1 - 20p^3(1-p)^3) ... complicated
        # Monte carlo for the game itself is fast enough and easier to read
        
        server_points = 0
        receiver_points = 0
        
        while True:
            # Win point?
            if random.random() < server_prob:
                server_points += 1
            else:
                receiver_points += 1
            
            # Game over conditions
            if server_points >= 4 and server_points >= receiver_points + 2:
                return True, server_points, receiver_points
            if receiver_points >= 4 and receiver_points >= server_points + 2:
                return False, server_points, receiver_points

    def simulate_tiebreak(self, p_serve_a: float, p_serve_b: float, final_set: bool = False) -> Tuple[bool, int, int]:
        """
        Simulate a tiebreak (first to 7, win by 2).
        For some grand slams, final set is to 10.
        """
        target = 10 if (final_set and self.best_of == 5) else 7
        
        points_a = 0
        points_b = 0
        
        # Serving order: A, B, B, A, A, B, B...
        # 0, 1, 2, 3, 4, 5, 6...
        # A, B, B, A, A, B, B...
        
        server_is_a = True # A serves first point
        points_served = 0
        
        while True:
            # Determine current server
            prob = p_serve_a if server_is_a else p_serve_b
            
            # Play point
            if random.random() < prob:
                if server_is_a: points_a += 1
                else: points_b += 1
            else:
                if server_is_a: points_b += 1
                else: points_a += 1
            
            points_served += 1
            
            # Switch server every 2 points after the first
            if points_served % 2 == 1:
                server_is_a = not server_is_a
                
            # Check win condition
            if (points_a >= target or points_b >= target) and abs(points_a - points_b) >= 2:
                return (points_a > points_b), points_a, points_b

    def simulate_set(self, start_server_a: bool, is_final_set: bool = False) -> Tuple[bool, int, int, bool]:
        """
        Simulate a set.
        Returns: (a_won, games_a, games_b, next_server_is_a)
        """
        games_a = 0
        games_b = 0
        server_is_a = start_server_a
        
        while True:
            # Check for set end (6-x or 7-5)
            if (games_a >= 6 or games_b >= 6) and abs(games_a - games_b) >= 2:
                return (games_a > games_b), games_a, games_b, server_is_a
            
            # Check for tiebreak (6-6)
            if games_a == 6 and games_b == 6:
                # Tiebreak
                # Note: Some GS use super tiebreak in final set, ignoring for simplicity unless needed
                a_won_tb, _, _ = self.simulate_tiebreak(self.p_serve_a, self.p_serve_b, is_final_set)
                if a_won_tb:
                    return True, 7, 6, not server_is_a # Switch server after set
                else:
                    return False, 6, 7, not server_is_a
            
            # Play game
            prob = self.p_serve_a if server_is_a else self.p_serve_b
            server_won, _, _ = self.simulate_game(prob)
            
            if server_is_a:
                if server_won: games_a += 1
                else: games_b += 1
            else:
                if server_won: games_b += 1
                else: games_a += 1
            
            # Switch server
            server_is_a = not server_is_a

    def simulate_match(self) -> Dict:
        """Simulate a full match."""
        sets_a = 0
        sets_b = 0
        total_games_a = 0
        total_games_b = 0
        server_is_a = True # Assume A serves first (can be randomized)
        
        # Randomize first server (50/50)
        if random.random() > 0.5:
            server_is_a = False
            
        score_str = []
        
        while sets_a < self.sets_to_win and sets_b < self.sets_to_win:
            is_final = (sets_a + sets_b) == (self.best_of - 1)
            
            a_won_set, ga, gb, next_server = self.simulate_set(server_is_a, is_final)
            
            total_games_a += ga
            total_games_b += gb
            server_is_a = next_server
            
            if a_won_set:
                sets_a += 1
            else:
                sets_b += 1
            
            score_str.append(f"{ga}-{gb}")
            
        return {
            'winner_a': sets_a > sets_b,
            'sets_a': sets_a,
            'sets_b': sets_b,
            'games_a': total_games_a,
            'games_b': total_games_b,
            'total_games': total_games_a + total_games_b,
            'game_diff': total_games_a - total_games_b,
            'score': " ".join(score_str)
        }

    def run(self, n_sims: int = 5000) -> SimulationResult:
        """Run N simulations and aggregate results."""
        wins_a = 0
        total_games_list = []
        game_diff_list = []
        set_scores = {}
        
        for _ in range(n_sims):
            res = self.simulate_match()
            
            if res['winner_a']:
                wins_a += 1
                score_key = f"{res['sets_a']}-{res['sets_b']}" # e.g. "2-0"
            else:
                score_key = f"{res['sets_a']}-{res['sets_b']}" # e.g. "1-2"
                
            set_scores[score_key] = set_scores.get(score_key, 0) + 1
            total_games_list.append(res['total_games'])
            game_diff_list.append(res['game_diff'])
            
        # Aggregate
        prob_a = wins_a / n_sims
        
        # Distributions
        total_dist = {}
        for g in total_games_list:
            total_dist[g] = total_dist.get(g, 0) + 1
        total_dist = {k: v/n_sims for k, v in total_dist.items()}
        
        diff_dist = {}
        for d in game_diff_list:
            diff_dist[d] = diff_dist.get(d, 0) + 1
        diff_dist = {k: v/n_sims for k, v in diff_dist.items()}
        
        score_dist = {k: v/n_sims for k, v in set_scores.items()}
        
        return SimulationResult(
            winner_a_prob=prob_a,
            expected_total_games=np.mean(total_games_list),
            total_games_dist=total_dist,
            game_diff_dist=diff_dist,
            set_score_dist=score_dist,
            median_total_games=np.median(total_games_list),
            median_game_diff=int(np.median(game_diff_list))
        )

def get_implied_serve_prob(surface: str, gender: str = 'atp') -> float:
    """Get baseline serve win probability for surface."""
    # Approximate ATP averages (2024)
    if gender == 'atp':
        if surface.lower() == 'clay': return 0.63
        if surface.lower() == 'grass': return 0.67
        return 0.65 # Hard
    else:
        # WTA
        if surface.lower() == 'clay': return 0.57
        if surface.lower() == 'grass': return 0.61
        return 0.59 # Hard
