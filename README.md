# Tennis Match Prediction Model

A quantitative sports betting model that identifies edge in ATP tennis markets through ensemble prediction and Monte Carlo simulation. The system combines multiple orthogonal signals (Elo ratings, serve/return metrics, head-to-head, fatigue, surface affinity) and extends beyond simple moneyline prediction into derivative markets (spreads, totals, set betting).

**Backtest Results:** +10.78% ROI, 2.60 Sharpe ratio (2019-2024, hard court, n≈800 bets)

## Core Capabilities

### 1. Ensemble Prediction Engine
Combines five independent signals with learned weights to generate calibrated win probabilities:

| Signal | Weight | Description |
|--------|--------|-------------|
| Elo Rating | 50% | Surface-specific with temporal decay and form adjustment |
| Serve/Return | 15% | Point-level dominance metrics from match statistics |
| Head-to-Head | 15% | Historical matchup effects and stylistic adjustments |
| Fatigue | 10% | Schedule density, travel, and match load modeling |
| Surface Form | 10% | Specialist vs. generalist performance patterns |

### 2. Monte Carlo Simulation Engine
Point-by-point match simulation for pricing derivative markets:

```python
# Simulate 5,000 matches at point granularity
simulator = MatchSimulator(p_serve_a=0.65, p_serve_b=0.62, best_of=3)
result = simulator.run(n_sims=5000)

# Output distributions for:
# - Total games (for Over/Under markets)
# - Game differential (for Spread markets)
# - Exact set scores (for Set Betting markets)
```

The simulator models tennis scoring rules precisely: deuce games, tiebreaks (including super-tiebreaks for Grand Slam final sets), and proper service rotation.

### 3. Market Maker Module
Converts simulation distributions into fair betting lines:

```bash
python main.py market "Sinner" "Alcaraz" --surface Hard --best-of 5

# Output:
# [MONEYLINE] Sinner: 1.86 (53.8%) | Alcaraz: 2.17
# [SPREAD] Fair Line: Sinner +1 Games | -2.5: 42% | +2.5: 65%
# [TOTAL] Fair Line: 41 | Over 22.5: 99.8%
# [SETS] 3-0: 14.4% | 3-1: 19.3% | 3-2: 20.2% | ...
```

### 4. Momentum & "Vibes" Analysis
Quantifies intangible factors that traditional models miss:

| Metric | Description |
|--------|-------------|
| **Clutch Rating** | Tiebreak win % (last 52 weeks) — pressure performance |
| **Grit Rating** | Deciding set win % — mental fortitude in long matches |
| **Dominance Rating** | % of sets won 6-0, 6-1, 6-2 — "in the zone" indicator |
| **Recent Form** | Last 5 match win % — current trajectory |

## Architecture

```
tennis_model/
├── main.py                         # CLI: predict, backtest, market analysis
├── src/
│   ├── data_loader.py              # Sackmann data ingestion + odds loading
│   ├── backtest.py                 # Chronological backtesting engine
│   ├── ratings/
│   │   ├── elo_enhanced.py         # Elo with decay, form, K-factor scaling
│   │   └── glicko2.py              # Glicko-2 with rating deviation
│   ├── features/
│   │   ├── serve_return.py         # Rolling serve/return statistics
│   │   ├── h2h.py                  # Head-to-head analysis
│   │   ├── fatigue.py              # Schedule and travel modeling
│   │   ├── surface.py              # Surface-specific performance
│   │   └── momentum.py             # Clutch, grit, dominance tracking
│   └── prediction/
│       ├── ensemble.py             # Multi-signal aggregation
│       ├── simulation.py           # Monte Carlo match simulator
│       ├── market_maker.py         # Fair line generation
│       └── upcoming.py             # Live prediction interface
└── data/                           # Match data (gitignored)
```

## Technical Design

### Rating System Implementation

Three rating systems with different trade-offs:

**Enhanced Elo** — Addresses two classic Elo weaknesses:
- *Staleness*: Ratings decay toward 1500 after 180 days of inactivity
- *Form blindness*: ±50 point adjustment based on recent performance vs. expectation

**Glicko-2** — Tracks rating deviation (uncertainty) alongside rating. Players with sparse match history produce more conservative predictions. Implements the [Glickman algorithm](http://www.glicko.net/glicko/glicko2.pdf).

**Surface-Specific** — Maintains separate Hard/Clay/Grass ratings per player. K-factors scale by tournament importance (Grand Slam: 32, Masters: 28, ATP 500: 24, ATP 250: 20).

### Simulation Engine Details

The Monte Carlo simulator operates at point granularity:

```
Point → Game → Set → Match

For each point:
  - Determine server (proper rotation, tiebreak rules)
  - Sample outcome from Bernoulli(p_serve)
  - Update game state (0-15-30-40-deuce-advantage)
  - Handle game/set/match transitions
```

Serve probabilities are estimated by blending player service stats against opponent return stats:
```
P(A wins serve vs B) = 0.6 × A_serve_won% + 0.4 × (1 - B_return_won%)
```

### Backtesting Methodology

Chronological processing prevents look-ahead bias:

```python
for match in sorted_by_date(matches):
    prediction = model.predict(match)      # Uses only prior data
    if expected_value(prediction, odds) > threshold:
        place_bet(match)
    model.update(match.outcome)            # Then update ratings
```

## Usage

```bash
# Install
git clone https://github.com/[username]/tennis_model.git && cd tennis_model
pip install -r requirements.txt

# Single match prediction
python main.py predict --player-a "Sinner" --player-b "Alcaraz" --surface Hard

# With value detection (compare to bookmaker odds)
python main.py predict --player-a "Sinner" --player-b "Alcaraz" \
  --surface Hard --odds-a 1.65 --odds-b 2.30

# Full market analysis (moneyline, spread, totals, set betting)
python main.py market "Sinner" "Alcaraz" --surface Hard --best-of 3

# Historical backtest
python main.py backtest --min-ev 7 --min-odds 1.10 --max-odds 1.60

# Player analysis
python main.py analyze-player "Jannik Sinner"
```

## Backtest Results

Optimal parameters via grid search:

| Parameter | Value |
|-----------|-------|
| Surface | Hard |
| Odds range | 1.10–1.60 |
| Min EV threshold | 7% |

| Metric | Value |
|--------|-------|
| Sample size | ~800 bets |
| Win rate | 56.2% |
| ROI | +10.78% |
| Sharpe ratio | 2.60 |
| Max drawdown | -8.3% |

The model performs best on hard court favorites where form signals are most reliable and market liquidity is highest.

## Data Sources

- [Jeff Sackmann / Tennis Abstract](https://github.com/JeffSackmann/tennis_atp) — Historical ATP match data with point-level statistics
- [tennis-data.co.uk](http://tennis-data.co.uk/) — Historical bookmaker odds (Pinnacle, Bet365, market average)
- [The Odds API](https://the-odds-api.com/) — Live odds integration (optional)

## Limitations & Future Work

**Current limitations:**
- ATP only; WTA would require separate calibration
- No injury/motivation modeling
- Serve probability estimation assumes independent points (ignores momentum within games)

**Potential extensions:**
- In-play betting using point-by-point simulation updates
- Player embedding models for style matchup prediction
- Bayesian ensemble weight optimization

## License

MIT

## Acknowledgments

- Jeff Sackmann for maintaining the most comprehensive public tennis dataset
- Mark Glickman for the Glicko-2 rating system
- Isaac Rose-Berman for insights on attacking inefficient derivative markets
