# Tennis Match Prediction Model

A professional-grade tennis betting and match prediction system built with Python. Combines multiple rating systems, feature signals, and ensemble methods to predict ATP match outcomes and identify value betting opportunities.

**Key Result:** Achieved **+10.78% ROI** with a **2.60 Sharpe ratio** on historical backtests (2019-2024, hard court matches).

## Features

- **Multiple Rating Systems** — Elo, Enhanced Elo (with form/decay), and Glicko-2 implementations
- **Ensemble Predictions** — Combines 5 independent signals: ratings, serve/return stats, head-to-head, fatigue, and surface performance
- **Backtesting Engine** — Chronological simulation with no look-ahead bias for honest performance evaluation
- **Value Detection** — Identifies positive expected value (+EV) opportunities by comparing model probabilities to market odds
- **Live Odds Integration** — Fetches real-time odds via The Odds API for active tournaments
- **Tournament Simulation** — Predicts full tournament brackets round-by-round

## Tech Stack

- **Python 3.10+**
- pandas, numpy — Data manipulation and numerical computing
- scikit-learn — Machine learning utilities
- requests, beautifulsoup4, selenium — Data fetching and web scraping
- matplotlib — Visualization
- argparse — Command-line interface

## Project Structure

```
tennis_model/
├── main.py                     # CLI entry point
├── src/
│   ├── elo.py                  # Basic Elo rating system
│   ├── data_loader.py          # Data fetching and cleaning
│   ├── odds_scraper.py         # Live odds API integration
│   ├── backtest.py             # Historical backtesting engine
│   ├── ratings/
│   │   ├── elo_enhanced.py     # Elo with form adjustment and decay
│   │   └── glicko2.py          # Glicko-2 rating system
│   ├── features/
│   │   ├── serve_return.py     # Serve/return statistics
│   │   ├── h2h.py              # Head-to-head analysis
│   │   ├── fatigue.py          # Match load and travel fatigue
│   │   └── surface.py          # Surface-specific performance
│   └── prediction/
│       ├── ensemble.py         # Weighted signal combination
│       └── upcoming.py         # Live tournament predictions
└── data/                       # Historical match data (gitignored)
```

## Installation

```bash
git clone https://github.com/riftfern/tennis_model.git
cd tennis_model
pip install -r requirements.txt
```

## Usage

```bash
# Train ratings on historical data
python main.py

# Predict a match
python main.py predict --player-a "Sinner" --player-b "Alcaraz" --surface Hard

# Predict with odds for value detection
python main.py predict --player-a "Sinner" --player-b "Alcaraz" \
  --surface Hard --odds-a 1.65 --odds-b 2.30

# Run historical backtest
python main.py backtest --min-ev 7 --min-odds 1.10 --max-odds 1.60

# Compare rating systems
python main.py backtest-compare --start-year 2020

# Analyze a player's ratings and stats
python main.py analyze-player "Jannik Sinner"
```

## How It Works

### Ensemble Prediction System

The model combines five independent signals with learned weights:

| Signal | Weight | Description |
|--------|--------|-------------|
| Base Rating | 50% | Elo/Glicko-2 win probability |
| Serve/Return | 15% | Rolling serve dominance differential |
| Head-to-Head | 15% | Historical matchup adjustment |
| Fatigue | 10% | Match load and travel penalties |
| Surface | 10% | Surface specialist bonus/transitions |

### Rating Systems

**Elo Rating**
- Surface-specific ratings (Hard, Clay, Grass)
- K-factor adjustment by tournament level (Grand Slam: 32, Masters: 28, etc.)
- New player acceleration for faster convergence

**Enhanced Elo**
- Inactivity decay (ratings regress after 180 days)
- Form adjustment based on recent performance vs. expectation
- Upset bonus rewards underdog victories

**Glicko-2**
- Tracks rating uncertainty (RD) — wider for inactive players
- Volatility adaptation for inconsistent performers
- More statistically principled for sparse player histories

### Value Betting

Expected value calculation:
```
EV = (model_prob × (odds - 1)) - ((1 - model_prob) × stake)
```

The model identifies value when:
- Model probability > implied odds probability
- EV exceeds configurable threshold (default: 7%)

## Backtest Results

Best performing strategy on 2019-2024 ATP data:

| Metric | Value |
|--------|-------|
| Surface | Hard court |
| Odds Range | 1.10 - 1.60 (favorites) |
| Min EV Threshold | 7% |
| Win Rate | 54-57% |
| **ROI** | **+10.78%** |
| **Sharpe Ratio** | **2.60** |

## Example Output

```
Jannik Sinner vs Carlos Alcaraz (Hard court)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model probability: 68.2% vs 31.8%
Confidence: 70%
Elo ratings: 1886 vs 1734

At odds 1.65 (implied 60.6%):
Edge: +7.6%
Expected Value: +12.5%  → VALUE BET
```

## Data Sources

- **Match Data:** [Jeff Sackmann's Tennis Abstract](https://github.com/JeffSackmann/tennis_atp) (2010-2024)
- **Historical Odds:** [tennis-data.co.uk](http://tennis-data.co.uk/)
- **Live Odds:** [The Odds API](https://the-odds-api.com/) (optional, requires API key)

## Skills Demonstrated

- **Algorithm Implementation** — Elo and Glicko-2 rating systems from scratch
- **Feature Engineering** — Domain-specific signals (serve stats, fatigue, surface transitions)
- **Ensemble Methods** — Weighted combination of uncorrelated predictors
- **Backtesting** — Chronological simulation avoiding look-ahead bias
- **Data Pipeline** — Web scraping, API integration, data cleaning
- **CLI Design** — argparse with subcommands, progress tracking, formatted output
- **Statistical Validation** — ROI, Sharpe ratio, performance breakdowns

## Author

Jack ([@riftfern](https://github.com/riftfern))

## License

MIT

## Acknowledgments

- Jeff Sackmann for the comprehensive tennis match dataset
- Glicko-2 algorithm by Mark Glickman
