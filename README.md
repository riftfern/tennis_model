# Tennis Match Prediction Model

A Python model for predicting professional tennis match outcomes using historical match data and player statistics.

## Features

- **Match outcome prediction** using ELO ratings and surface-adjusted performance
- **Data pipeline** for fetching and processing ATP/WTA match data
- **Backtesting framework** to evaluate model accuracy against historical results

## Tech Stack

- Python 3.10+
- pandas, numpy
- scikit-learn
- [Add any other libraries from your requirements.txt]

## Project Structure

```
tennis_model/
├── src/
│   ├── data/          # Data fetching and preprocessing
│   ├── features/      # Feature engineering
│   └── model/         # Prediction model
├── main.py            # Entry point
├── requirements.txt
└── USAGE.md
```

## Quick Start

```bash
# Clone the repo
git clone https://github.com/riftfern/tennis_model.git
cd tennis_model

# Install dependencies
pip install -r requirements.txt

# Run predictions
python main.py
```

## How It Works

1. Fetches historical match data from [your data source]
2. Calculates rolling ELO ratings adjusted for surface type
3. Engineers features like head-to-head record, recent form, and surface win rate
4. Outputs win probabilities for upcoming matches

## Example Output

```
Sinner vs Alcaraz (Hard) → Sinner 54.2% | Alcaraz 45.8%
```

## Roadmap

- [ ] Add serve/return statistics
- [ ] Integrate live odds for +EV detection
- [ ] Tournament bracket simulation

## Author

Jack ([@riftfern](https://github.com/riftfern))
