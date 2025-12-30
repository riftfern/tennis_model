"""
Backtesting module for tennis betting model.

Simulates historical betting performance by:
1. Building Elo ratings chronologically (only using past data)
2. Finding +EV bets based on historical odds
3. Tracking profit/loss and ROI
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm
import requests

from .elo import TennisElo, calculate_ev, implied_probability


# Jeff Sackmann's betting data repository
BETTING_DATA_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_betting/master"


@dataclass
class BetResult:
    """Single bet result."""
    match_date: str
    tournament: str
    player_bet: str
    opponent: str
    surface: str
    model_prob: float
    odds: float
    implied_prob: float
    ev_percent: float
    stake: float
    won: bool
    profit: float


@dataclass
class BacktestResults:
    """Aggregated backtest results."""
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
    bets: List[BetResult] = field(default_factory=list)

    def summary(self) -> str:
        """Return formatted summary."""
        return f"""
========================================
BACKTEST RESULTS
========================================
Total Bets:     {self.total_bets}
Record:         {self.wins}W - {self.losses}L ({self.win_rate:.1f}%)
Total Staked:   ${self.total_staked:,.2f}
Total Profit:   ${self.total_profit:+,.2f}
ROI:            {self.roi_percent:+.2f}%
Avg Odds:       {self.avg_odds:.2f}
Avg EV:         {self.avg_ev:.1f}%
Max Drawdown:   ${self.max_drawdown:,.2f}
========================================
"""


def download_betting_data(
    start_year: int = 2010,
    end_year: int = 2024,
    data_dir: str = 'data'
) -> List[Path]:
    """
    Download historical betting/odds data from Jeff Sackmann's repository.
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    downloaded = []

    for year in tqdm(range(start_year, end_year + 1), desc="Downloading odds data"):
        filename = f"{year}.csv"
        url = f"{BETTING_DATA_BASE}/{filename}"
        local_path = data_path / f"odds_{year}.csv"

        if local_path.exists():
            downloaded.append(local_path)
            continue

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            with open(local_path, 'wb') as f:
                f.write(response.content)

            downloaded.append(local_path)
        except requests.RequestException as e:
            # Try alternative format
            pass

    return downloaded


def load_odds_data(
    data_dir: str = 'data',
    start_year: int = 2010,
    end_year: int = 2024
) -> pd.DataFrame:
    """Load historical odds data."""
    data_path = Path(data_dir)
    all_data = []

    for year in range(start_year, end_year + 1):
        filepath = data_path / f"odds_{year}.csv"
        if filepath.exists():
            try:
                df = pd.read_csv(filepath, low_memory=False)
                all_data.append(df)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


def merge_matches_with_odds(
    matches: pd.DataFrame,
    odds: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge match data with odds data.

    The odds data from Sackmann includes columns like:
    - AvgW, AvgL: Average odds for winner/loser
    - MaxW, MaxL: Best odds for winner/loser
    - B365W, B365L: Bet365 odds
    - PSW, PSL: Pinnacle odds
    """
    # If odds data has similar structure to match data, they might already be merged
    # Check for odds columns in matches
    odds_columns = ['AvgW', 'AvgL', 'MaxW', 'MaxL', 'B365W', 'B365L', 'PSW', 'PSL']

    if any(col in matches.columns for col in odds_columns):
        return matches

    # Otherwise need to merge on match identifiers
    # This is complex due to name variations - return matches as-is for now
    return matches


class Backtester:
    """
    Backtests the Elo model against historical data with odds.
    """

    def __init__(
        self,
        min_ev: float = 3.0,
        min_odds: float = 1.20,
        max_odds: float = 5.00,
        min_prob: float = 0.15,
        stake_size: float = 100.0,
        kelly_fraction: float = 0.0,  # 0 = flat betting, >0 = fractional Kelly
        odds_column: str = 'Max'  # 'Max', 'Avg', 'B365', 'PS' (Pinnacle)
    ):
        """
        Initialize backtester.

        Args:
            min_ev: Minimum EV% to place bet
            min_odds: Minimum odds to bet on
            max_odds: Maximum odds to bet on (avoid long shots)
            min_prob: Minimum model probability
            stake_size: Fixed stake per bet (if kelly_fraction=0)
            kelly_fraction: Fraction of Kelly criterion (0.25 = quarter Kelly)
            odds_column: Which odds to use ('Max', 'Avg', 'B365', 'PS')
        """
        self.min_ev = min_ev
        self.min_odds = min_odds
        self.max_odds = max_odds
        self.min_prob = min_prob
        self.stake_size = stake_size
        self.kelly_fraction = kelly_fraction
        self.odds_column = odds_column

    def calculate_kelly_stake(
        self,
        prob: float,
        odds: float,
        bankroll: float
    ) -> float:
        """Calculate Kelly criterion stake."""
        # Kelly formula: f* = (bp - q) / b
        # where b = odds - 1, p = probability of winning, q = 1 - p
        b = odds - 1
        q = 1 - prob
        kelly = (b * prob - q) / b

        if kelly <= 0:
            return 0

        # Apply fractional Kelly
        stake = bankroll * kelly * self.kelly_fraction

        # Cap at reasonable amount
        return min(stake, bankroll * 0.05)

    def run(
        self,
        matches: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        surface_filter: Optional[str] = None,
        verbose: bool = True
    ) -> BacktestResults:
        """
        Run backtest on historical matches.

        Args:
            matches: DataFrame with match data and odds
            start_date: Start date for backtest (YYYYMMDD)
            end_date: End date for backtest
            surface_filter: Only bet on specific surface
            verbose: Show progress bar
        """
        # Check for required odds columns
        winner_odds_col = f'{self.odds_column}W'
        loser_odds_col = f'{self.odds_column}L'

        if winner_odds_col not in matches.columns:
            # Try alternative column names
            if 'AvgW' in matches.columns:
                winner_odds_col = 'AvgW'
                loser_odds_col = 'AvgL'
            elif 'avg_odds_winner' in matches.columns:
                winner_odds_col = 'avg_odds_winner'
                loser_odds_col = 'avg_odds_loser'
            else:
                raise ValueError(f"Odds columns not found. Available: {matches.columns.tolist()}")

        # Sort by date
        matches = matches.copy()
        if 'tourney_date' in matches.columns:
            matches = matches.sort_values('tourney_date')

        # Filter by date range
        if start_date and 'tourney_date' in matches.columns:
            matches = matches[matches['tourney_date'] >= int(start_date)]
        if end_date and 'tourney_date' in matches.columns:
            matches = matches[matches['tourney_date'] <= int(end_date)]

        # Filter by surface
        if surface_filter:
            matches = matches[matches['surface'].str.lower() == surface_filter.lower()]

        # Remove matches without odds
        matches = matches.dropna(subset=[winner_odds_col, loser_odds_col])

        # Initialize Elo system
        elo = TennisElo()

        # Track results
        results = BacktestResults()
        bankroll = 10000.0  # Starting bankroll for Kelly
        peak_bankroll = bankroll

        iterator = tqdm(matches.iterrows(), total=len(matches), desc="Backtesting") if verbose else matches.iterrows()

        for _, match in iterator:
            winner = match['winner_name']
            loser = match['loser_name']
            surface = match.get('surface', 'Hard')
            tourney_level = match.get('tourney_level', 'default')
            tourney_name = match.get('tourney_name', 'Unknown')
            match_date = str(match.get('tourney_date', ''))

            winner_odds = match[winner_odds_col]
            loser_odds = match[loser_odds_col]

            # Skip invalid odds
            if pd.isna(winner_odds) or pd.isna(loser_odds):
                elo.update_ratings(winner, loser, surface, tourney_level)
                continue

            if winner_odds < 1.01 or loser_odds < 1.01:
                elo.update_ratings(winner, loser, surface, tourney_level)
                continue

            # Get pre-match prediction
            pred = elo.predict_match(winner, loser, surface)
            winner_prob = pred[winner]
            loser_prob = pred[loser]

            # Check for value bets on both sides
            for player, prob, odds, won in [
                (winner, winner_prob, winner_odds, True),
                (loser, loser_prob, loser_odds, False)
            ]:
                # Apply filters
                if prob < self.min_prob:
                    continue
                if odds < self.min_odds or odds > self.max_odds:
                    continue

                ev = calculate_ev(prob, odds)
                implied = implied_probability(odds)

                if ev >= self.min_ev:
                    # Place bet
                    if self.kelly_fraction > 0:
                        stake = self.calculate_kelly_stake(prob, odds, bankroll)
                    else:
                        stake = self.stake_size

                    if stake <= 0:
                        continue

                    # Calculate profit
                    if won:
                        profit = stake * (odds - 1)
                        results.wins += 1
                    else:
                        profit = -stake
                        results.losses += 1

                    # Update bankroll
                    bankroll += profit
                    peak_bankroll = max(peak_bankroll, bankroll)
                    drawdown = peak_bankroll - bankroll
                    results.max_drawdown = max(results.max_drawdown, drawdown)

                    # Record bet
                    opponent = loser if player == winner else winner
                    bet = BetResult(
                        match_date=match_date,
                        tournament=tourney_name,
                        player_bet=player,
                        opponent=opponent,
                        surface=surface,
                        model_prob=prob,
                        odds=odds,
                        implied_prob=implied,
                        ev_percent=ev,
                        stake=stake,
                        won=won,
                        profit=profit
                    )
                    results.bets.append(bet)
                    results.total_bets += 1
                    results.total_staked += stake
                    results.total_profit += profit

            # Update Elo after processing match
            elo.update_ratings(winner, loser, surface, tourney_level)

        # Calculate final stats
        if results.total_bets > 0:
            results.roi_percent = (results.total_profit / results.total_staked) * 100
            results.win_rate = (results.wins / results.total_bets) * 100
            results.avg_odds = sum(b.odds for b in results.bets) / len(results.bets)
            results.avg_ev = sum(b.ev_percent for b in results.bets) / len(results.bets)

        return results

    def analyze_by_surface(
        self,
        matches: pd.DataFrame,
        verbose: bool = False
    ) -> Dict[str, BacktestResults]:
        """Run separate backtests for each surface."""
        surfaces = ['Hard', 'Clay', 'Grass']
        results = {}

        for surface in surfaces:
            surface_matches = matches[matches['surface'] == surface]
            if len(surface_matches) > 0:
                results[surface] = self.run(surface_matches, verbose=verbose)

        return results

    def analyze_by_odds_range(
        self,
        matches: pd.DataFrame,
        ranges: List[Tuple[float, float]] = None,
        verbose: bool = False
    ) -> Dict[str, BacktestResults]:
        """Analyze performance by odds range."""
        if ranges is None:
            ranges = [(1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0)]

        results = {}
        original_min, original_max = self.min_odds, self.max_odds

        for min_o, max_o in ranges:
            self.min_odds = min_o
            self.max_odds = max_o
            key = f"{min_o:.1f}-{max_o:.1f}"
            results[key] = self.run(matches, verbose=verbose)

        self.min_odds, self.max_odds = original_min, original_max
        return results

    def analyze_by_ev_threshold(
        self,
        matches: pd.DataFrame,
        thresholds: List[float] = None,
        verbose: bool = False
    ) -> Dict[str, BacktestResults]:
        """Analyze performance by EV threshold."""
        if thresholds is None:
            thresholds = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

        results = {}
        original_min_ev = self.min_ev

        for ev in thresholds:
            self.min_ev = ev
            results[f"EV≥{ev:.0f}%"] = self.run(matches, verbose=verbose)

        self.min_ev = original_min_ev
        return results


def generate_report(results: BacktestResults, detailed: bool = False) -> str:
    """Generate a detailed backtest report."""
    report = [results.summary()]

    if detailed and results.bets:
        # Monthly breakdown
        bets_df = pd.DataFrame([
            {
                'date': b.match_date,
                'player': b.player_bet,
                'odds': b.odds,
                'ev': b.ev_percent,
                'profit': b.profit,
                'won': b.won
            }
            for b in results.bets
        ])

        if 'date' in bets_df.columns and len(bets_df) > 0:
            bets_df['year_month'] = bets_df['date'].astype(str).str[:6]
            monthly = bets_df.groupby('year_month').agg({
                'profit': 'sum',
                'won': ['sum', 'count']
            }).round(2)

            report.append("\nMonthly Performance:")
            report.append("-" * 40)
            for idx, row in monthly.iterrows():
                profit = row[('profit', 'sum')]
                wins = int(row[('won', 'sum')])
                total = int(row[('won', 'count')])
                report.append(f"{idx}: ${profit:+.2f} ({wins}/{total} wins)")

    return '\n'.join(report)


if __name__ == "__main__":
    # Example usage
    from data_loader import load_matches, download_sackmann_data

    print("Loading match data...")
    matches = load_matches(start_year=2019, end_year=2024)

    if len(matches) == 0:
        print("No data found. Downloading...")
        download_sackmann_data(2019, 2024)
        matches = load_matches(start_year=2019, end_year=2024)

    # Check for odds columns
    odds_cols = [c for c in matches.columns if 'odds' in c.lower() or c in ['AvgW', 'AvgL', 'MaxW', 'MaxL']]
    print(f"Available odds columns: {odds_cols}")

    if odds_cols:
        print("\nRunning backtest...")
        bt = Backtester(min_ev=3.0, min_odds=1.30, max_odds=4.0)
        results = bt.run(matches)
        print(results.summary())
