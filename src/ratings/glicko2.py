"""
Glicko-2 rating system for tennis players.

Advantages over Elo:
- Tracks rating deviation (uncertainty)
- Volatility adapts to inconsistent players
- More accurate for players with few recent matches
"""

import math
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


# Glicko-2 constants
GLICKO2_SCALE = 173.7178  # Converts Glicko-1 scale to Glicko-2


@dataclass
class Glicko2Rating:
    """Glicko-2 rating with uncertainty tracking."""
    mu: float = 0.0           # Rating in Glicko-2 scale (0 = 1500 Glicko-1)
    phi: float = 2.0148       # Rating deviation (350/173.7178 in Glicko-2 scale)
    sigma: float = 0.06       # Rating volatility
    matches_played: int = 0
    last_match_date: Optional[datetime] = None

    @property
    def rating(self) -> float:
        """Convert to Glicko-1 scale (1500 baseline)."""
        return self.mu * GLICKO2_SCALE + 1500

    @property
    def rd(self) -> float:
        """Rating deviation in Glicko-1 scale."""
        return self.phi * GLICKO2_SCALE

    @classmethod
    def from_glicko1(cls, rating: float = 1500, rd: float = 350, vol: float = 0.06):
        """Create from Glicko-1 scale values."""
        return cls(
            mu=(rating - 1500) / GLICKO2_SCALE,
            phi=rd / GLICKO2_SCALE,
            sigma=vol
        )


@dataclass
class SurfaceGlicko2:
    """Surface-specific Glicko-2 ratings for a player."""
    overall: Glicko2Rating = field(default_factory=Glicko2Rating)
    hard: Glicko2Rating = field(default_factory=Glicko2Rating)
    clay: Glicko2Rating = field(default_factory=Glicko2Rating)
    grass: Glicko2Rating = field(default_factory=Glicko2Rating)

    def get_surface_rating(self, surface: str) -> Glicko2Rating:
        """Get rating for a specific surface."""
        surface = surface.lower()
        if 'hard' in surface:
            return self.hard
        elif 'clay' in surface:
            return self.clay
        elif 'grass' in surface:
            return self.grass
        return self.overall


class Glicko2System:
    """
    Glicko-2 rating system implementation.

    Reference: http://www.glicko.net/glicko/glicko2.pdf
    """

    TAU = 0.5           # System constant (constrains volatility change)
    EPSILON = 0.000001  # Convergence tolerance

    def __init__(self, initial_rating: float = 1500, initial_rd: float = 350):
        self.initial_rating = initial_rating
        self.initial_rd = initial_rd
        self.ratings: Dict[str, SurfaceGlicko2] = {}

    def get_player_rating(self, player: str) -> SurfaceGlicko2:
        """Get or create a player's rating."""
        if player not in self.ratings:
            self.ratings[player] = SurfaceGlicko2(
                overall=Glicko2Rating.from_glicko1(self.initial_rating, self.initial_rd),
                hard=Glicko2Rating.from_glicko1(self.initial_rating, self.initial_rd),
                clay=Glicko2Rating.from_glicko1(self.initial_rating, self.initial_rd),
                grass=Glicko2Rating.from_glicko1(self.initial_rating, self.initial_rd)
            )
        return self.ratings[player]

    def g(self, phi: float) -> float:
        """g(phi) function in Glicko-2."""
        return 1 / math.sqrt(1 + 3 * phi**2 / math.pi**2)

    def E(self, mu: float, mu_j: float, phi_j: float) -> float:
        """Expected score function E(mu, mu_j, phi_j)."""
        exponent = -self.g(phi_j) * (mu - mu_j)
        # Clip to avoid overflow
        exponent = max(-20, min(20, exponent))
        return 1 / (1 + math.exp(exponent))

    def compute_variance(
        self,
        rating: Glicko2Rating,
        opponents: List[Tuple[Glicko2Rating, float]]  # (opponent_rating, score)
    ) -> float:
        """Compute variance v."""
        if not opponents:
            return float('inf')

        total = 0.0
        for opp_rating, _ in opponents:
            g_phi = self.g(opp_rating.phi)
            e = self.E(rating.mu, opp_rating.mu, opp_rating.phi)
            total += g_phi**2 * e * (1 - e)

        return 1 / total if total > 0 else float('inf')

    def compute_delta(
        self,
        rating: Glicko2Rating,
        opponents: List[Tuple[Glicko2Rating, float]],
        v: float
    ) -> float:
        """Compute improvement delta."""
        total = 0.0
        for opp_rating, score in opponents:
            g_phi = self.g(opp_rating.phi)
            e = self.E(rating.mu, opp_rating.mu, opp_rating.phi)
            total += g_phi * (score - e)

        return v * total

    def compute_new_volatility(
        self,
        sigma: float,
        phi: float,
        v: float,
        delta: float
    ) -> float:
        """Compute new volatility sigma' using iterative algorithm."""
        a = math.log(sigma**2)

        def f(x):
            ex = math.exp(x)
            d2 = delta**2
            p2 = phi**2
            t2 = self.TAU**2

            num = ex * (d2 - p2 - v - ex)
            den = 2 * (p2 + v + ex)**2

            return (num / den) - ((x - a) / t2)

        # Initial bounds
        A = a
        if delta**2 > phi**2 + v:
            B = math.log(delta**2 - phi**2 - v)
        else:
            k = 1
            while f(a - k * self.TAU) < 0:
                k += 1
            B = a - k * self.TAU

        # Bisection method
        fA = f(A)
        fB = f(B)

        while abs(B - A) > self.EPSILON:
            C = A + (A - B) * fA / (fB - fA)
            fC = f(C)

            if fC * fB <= 0:
                A = B
                fA = fB
            else:
                fA = fA / 2

            B = C
            fB = fC

        return math.exp(A / 2)

    def update_rating(
        self,
        player: str,
        opponents: List[Tuple[str, float]],  # (opponent_name, score 0/0.5/1)
        surface: str,
        match_date: datetime = None
    ) -> Glicko2Rating:
        """
        Update player rating after one or more games.

        Args:
            player: Player name
            opponents: List of (opponent_name, score) where score is 1 for win, 0 for loss
            surface: Surface type
            match_date: Date of match(es)
        """
        player_ratings = self.get_player_rating(player)
        rating = player_ratings.get_surface_rating(surface)

        if not opponents:
            # No games: increase RD due to inactivity
            phi_star = math.sqrt(rating.phi**2 + rating.sigma**2)
            rating.phi = min(phi_star, 2.0148)  # Cap at initial RD
            return rating

        # Get opponent ratings
        opp_data = []
        for opp_name, score in opponents:
            opp_ratings = self.get_player_rating(opp_name)
            opp_rating = opp_ratings.get_surface_rating(surface)
            opp_data.append((opp_rating, score))

        # Step 3: Compute variance v
        v = self.compute_variance(rating, opp_data)

        # Step 4: Compute delta
        delta = self.compute_delta(rating, opp_data, v)

        # Step 5: Compute new volatility
        sigma_new = self.compute_new_volatility(rating.sigma, rating.phi, v, delta)

        # Step 6: Update rating deviation
        phi_star = math.sqrt(rating.phi**2 + sigma_new**2)

        # Step 7: Update rating and RD
        phi_new = 1 / math.sqrt(1 / phi_star**2 + 1 / v)

        mu_change = 0.0
        for opp_rating, score in opp_data:
            g_phi = self.g(opp_rating.phi)
            e = self.E(rating.mu, opp_rating.mu, opp_rating.phi)
            mu_change += g_phi * (score - e)

        mu_new = rating.mu + phi_new**2 * mu_change

        # Update the rating
        rating.mu = mu_new
        rating.phi = phi_new
        rating.sigma = sigma_new
        rating.matches_played += len(opponents)
        rating.last_match_date = match_date

        return rating

    def update_single_match(
        self,
        winner: str,
        loser: str,
        surface: str,
        match_date: datetime = None
    ):
        """Update ratings for a single match."""
        self.update_rating(winner, [(loser, 1.0)], surface, match_date)
        self.update_rating(loser, [(winner, 0.0)], surface, match_date)

    def predict_match(
        self,
        player_a: str,
        player_b: str,
        surface: str
    ) -> Dict[str, float]:
        """
        Predict match with confidence based on rating deviations.
        """
        ratings_a = self.get_player_rating(player_a)
        ratings_b = self.get_player_rating(player_b)

        r_a = ratings_a.get_surface_rating(surface)
        r_b = ratings_b.get_surface_rating(surface)

        # Win probability
        prob_a = self.E(r_a.mu, r_b.mu, r_b.phi)

        # Confidence based on combined uncertainty
        # Lower RD = more confident
        combined_rd = math.sqrt(r_a.rd**2 + r_b.rd**2)
        # Scale: RD of 50 each (combined ~70) = high confidence
        # RD of 350 each (combined ~495) = low confidence
        confidence = max(0, min(1, 1 - (combined_rd - 100) / 600))

        return {
            player_a: prob_a,
            player_b: 1 - prob_a,
            'confidence': confidence,
            f'{player_a}_rating': r_a.rating,
            f'{player_b}_rating': r_b.rating,
            f'{player_a}_rd': r_a.rd,
            f'{player_b}_rd': r_b.rd,
            f'{player_a}_matches': r_a.matches_played,
            f'{player_b}_matches': r_b.matches_played
        }

    def process_matches(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Process historical matches to build ratings."""
        if 'tourney_date' in matches_df.columns:
            matches_df = matches_df.sort_values('tourney_date')

        results = []

        for _, match in matches_df.iterrows():
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

            # Get pre-match prediction
            pred = self.predict_match(winner, loser, surface)
            expected = pred[winner]

            # Update ratings
            self.update_single_match(winner, loser, surface, match_date)

            results.append({
                'winner': winner,
                'loser': loser,
                'surface': surface,
                'winner_pre_rating': pred[f'{winner}_rating'],
                'loser_pre_rating': pred[f'{loser}_rating'],
                'winner_rd': pred[f'{winner}_rd'],
                'loser_rd': pred[f'{loser}_rd'],
                'predicted_winner_prob': expected,
                'prediction_correct': expected > 0.5,
                'confidence': pred['confidence']
            })

        return pd.DataFrame(results)

    def get_top_players(self, n: int = 50, surface: Optional[str] = None) -> pd.DataFrame:
        """Get top N players by Glicko-2 rating."""
        data = []
        for player, ratings in self.ratings.items():
            if surface:
                r = ratings.get_surface_rating(surface)
            else:
                r = ratings.overall

            data.append({
                'player': player,
                'rating': r.rating,
                'rd': r.rd,
                'volatility': r.sigma,
                'matches': r.matches_played,
                'hard': ratings.hard.rating,
                'clay': ratings.clay.rating,
                'grass': ratings.grass.rating
            })

        df = pd.DataFrame(data)
        if len(df) > 0:
            return df.nlargest(n, 'rating')
        return df
