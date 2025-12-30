# Tennis Model Usage Guide

## Setup

```bash
cd /home/jg/Documents/dev/tennis_model
pip install -r requirements.txt
```

## Quick Start

```bash
# Download historical data (first time only)
python main.py

# Make a prediction
python main.py predict --player-a "Jannik Sinner" --player-b "Carlos Alcaraz" --surface Hard
```

---

## Commands

### 1. Basic Model Training

```bash
python main.py
```

Downloads ATP data (2020-2024), trains Elo ratings, shows top 20 players and sample predictions.

### 2. Predict a Match

```bash
# Basic prediction
python main.py predict --player-a "Jannik Sinner" --player-b "Carlos Alcaraz" --surface Hard

# With odds (shows EV calculation)
python main.py predict --player-a "Jannik Sinner" --player-b "Carlos Alcaraz" --surface Hard --odds-a 1.65 --odds-b 2.30

# Specify tournament
python main.py predict --player-a "Novak Djokovic" --player-b "Rafael Nadal" --surface Clay --tournament "Roland Garros"
```

**Surfaces:** `Hard`, `Clay`, `Grass`

### 3. Backtest (Original Elo)

```bash
# Full backtest with default settings
python main.py backtest

# Custom parameters
python main.py backtest --min-ev 5 --min-odds 1.20 --max-odds 3.0 --start-year 2022
```

### 4. Model Comparison Backtest

```bash
# Compare all 5 model variants
python main.py backtest-compare --start-year 2020
```

Compares:
- Baseline Elo
- Enhanced Elo (recency decay, form adjustment)
- Glicko-2 (uncertainty tracking)
- Ensemble (Elo) - all features combined
- Ensemble (Glicko) - all features with Glicko-2 base

### 5. Analyze a Player

```bash
python main.py analyze-player "Jannik Sinner"
python main.py analyze-player "Carlos Alcaraz"
```

Shows: ratings, surface breakdown, recent form, serve/return stats.

### 6. Live Odds (Requires API Key)

```bash
# Get predictions with live odds
python main.py --odds-api-key YOUR_KEY

# During active tournaments only (off-season = no data)
```

Get API key at: https://the-odds-api.com/

---

## Best Betting Strategy (From Backtesting)

Our backtests found the most profitable approach:

| Parameter | Value |
|-----------|-------|
| Model | Ensemble (Elo) |
| Surface | Hard court |
| Odds range | 1.10 - 1.60 (favorites) |
| Min EV | 7%+ |
| **ROI** | **+10.78%** |
| Sharpe | 2.60 |

### Interpreting Predictions

```
Jannik Sinner vs Carlos Alcaraz
--------------------------------------------------
Model probability:
  Jannik Sinner: 68.2%      <- Model's win probability
  Carlos Alcaraz: 31.8%
Confidence: 70%              <- Based on data quality
Ratings: 1886 vs 1734        <- Elo ratings

Jannik Sinner @ 1.65:
  Implied: 60.6%             <- Bookmaker's implied probability
  Edge: 7.6%                 <- Model prob - Implied prob
  EV: +12.5%                 <- Expected value
  >>> VALUE BET!             <- Flags when EV > threshold
```

**When to bet:**
- EV > 7%
- Edge > 5%
- Confidence > 50%
- Odds between 1.10-1.60

---

## Model Components

### Ensemble Predictor Weights

| Component | Weight | Description |
|-----------|--------|-------------|
| Base Elo | 50% | Surface-specific Elo rating |
| Serve/Return | 15% | Rolling serve dominance differential |
| Head-to-Head | 15% | Historical matchup adjustment |
| Fatigue | 10% | Match load, travel penalties |
| Surface | 10% | Surface specialist adjustment |

### Data Sources

- **Match data:** Jeff Sackmann's GitHub (tennis_atp)
- **Odds data:** tennis-data.co.uk
- **Live odds:** The Odds API

---

## File Structure

```
tennis_model/
├── main.py                 # CLI entry point
├── src/
│   ├── elo.py              # Basic Elo system
│   ├── data_loader.py      # Data download/loading
│   ├── backtest.py         # Original backtester
│   ├── backtest_enhanced.py # Model comparison
│   ├── odds_scraper.py     # Live odds API
│   ├── ratings/
│   │   ├── elo_enhanced.py # Enhanced Elo
│   │   └── glicko2.py      # Glicko-2 system
│   ├── features/
│   │   ├── serve_return.py # Serve/return stats
│   │   ├── h2h.py          # Head-to-head
│   │   ├── fatigue.py      # Fatigue tracking
│   │   └── surface.py      # Surface analysis
│   └── prediction/
│       ├── ensemble.py     # Combines all signals
│       └── upcoming.py     # Live predictions
└── data/                   # Downloaded CSV files (gitignored)
```

---

## Examples

### Pre-Tournament Analysis

```bash
# Before Australian Open - analyze top contenders
python main.py analyze-player "Jannik Sinner"
python main.py analyze-player "Novak Djokovic"
python main.py analyze-player "Carlos Alcaraz"

# Predict potential final
python main.py predict --player-a "Jannik Sinner" --player-b "Novak Djokovic" --surface Hard --odds-a 1.80 --odds-b 2.00
```

### Finding Value Bets

```bash
# Run backtest to validate strategy
python main.py backtest --min-ev 7 --min-odds 1.10 --max-odds 1.60 --start-year 2023

# Then apply same filters when betting live matches
```

### Testing Model Accuracy

```bash
# Compare all models
python main.py backtest-compare --start-year 2020

# Check accuracy by surface (modify code or filter data)
```

---

## Troubleshooting

**"No matches found"** - Data not downloaded yet. Run `python main.py` first.

**"No odds data"** - Tennis is in off-season (Dec-Jan between seasons). Live odds only available during tournaments.

**Low prediction confidence** - Player has limited recent matches. Model uses defaults for unknown players.

**Negative ROI in backtest** - Try tighter parameters: higher min EV, narrower odds range, specific surface.
