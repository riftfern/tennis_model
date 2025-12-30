#!/usr/bin/env python3
"""
Tennis Betting Model - Main Script

This script:
1. Downloads historical tennis data
2. Builds Elo ratings for all players
3. Shows current top players and ratings
4. Finds +EV betting opportunities
5. Backtests the model against historical odds
"""

import pandas as pd
from pathlib import Path
import argparse

from src.data_loader import (
    download_sackmann_data, load_matches, get_head_to_head,
    download_odds_data, load_matches_with_odds
)
from src.elo import TennisElo, calculate_ev, implied_probability, find_value_bets
from src.odds_scraper import OddsAPIClient, remove_vig
from src.backtest import Backtester, generate_report

# Enhanced model imports
from src.ratings.elo_enhanced import EnhancedElo
from src.ratings.glicko2 import Glicko2System
from src.features.serve_return import ServeReturnTracker
from src.features.h2h import H2HAnalyzer
from src.features.fatigue import FatigueTracker
from src.features.surface import SurfaceAnalyzer
from src.prediction.ensemble import EnsemblePredictor
from src.prediction.upcoming import UpcomingMatchPredictor
from src.backtest_enhanced import EnhancedBacktester, quick_backtest


def main():
    parser = argparse.ArgumentParser(description='Tennis Betting Model')
    parser.add_argument('--download', action='store_true', help='Download fresh data')
    parser.add_argument('--start-year', type=int, default=2015, help='Start year for data')
    parser.add_argument('--end-year', type=int, default=2024, help='End year for data')
    parser.add_argument('--tour', choices=['atp', 'wta', 'both'], default='atp')
    parser.add_argument('--odds-api-key', type=str, help='The Odds API key')
    args = parser.parse_args()

    print("=" * 60)
    print("TENNIS BETTING MODEL")
    print("=" * 60)

    # Step 1: Download/Load Data
    if args.download:
        print(f"\n[1] Downloading {args.tour.upper()} data ({args.start_year}-{args.end_year})...")
        if args.tour in ['atp', 'both']:
            download_sackmann_data(args.start_year, args.end_year, 'atp')
        if args.tour in ['wta', 'both']:
            download_sackmann_data(args.start_year, args.end_year, 'wta')

    print(f"\n[1] Loading match data ({args.start_year}-{args.end_year})...")
    matches = load_matches(start_year=args.start_year, end_year=args.end_year, tour=args.tour)

    if len(matches) == 0:
        print("No data found. Run with --download to fetch data.")
        return

    print(f"    Loaded {len(matches):,} matches")

    # Step 2: Build Elo Ratings
    print("\n[2] Building Elo ratings...")
    elo = TennisElo()
    results = elo.process_matches(matches)

    # Calculate accuracy
    accuracy = results['prediction_correct'].mean() * 100
    print(f"    Model accuracy: {accuracy:.1f}%")

    # Step 3: Show Top Players
    print("\n[3] Current Top 20 Players (Overall Elo):")
    print("-" * 60)
    top_players = elo.get_top_players(n=20)
    for i, row in top_players.iterrows():
        rank = top_players.index.get_loc(i) + 1
        print(f"    {rank:2}. {row['player']:<25} Elo: {row['elo']:.0f}  "
              f"(H:{row['hard']:.0f} C:{row['clay']:.0f} G:{row['grass']:.0f})")

    # Step 4: Surface-specific leaders
    print("\n[4] Surface Specialists:")
    for surface in ['Hard', 'Clay', 'Grass']:
        top_surface = elo.get_top_players(n=5, surface=surface)
        print(f"\n    Top 5 on {surface}:")
        for i, row in top_surface.head(5).iterrows():
            rank = list(top_surface.head(5).index).index(i) + 1
            print(f"      {rank}. {row['player']:<25} {row[surface.lower()]:.0f}")

    # Step 5: Interactive Predictions
    print("\n" + "=" * 60)
    print("MATCH PREDICTION")
    print("=" * 60)

    # Example prediction
    example_matches = [
        ("Jannik Sinner", "Carlos Alcaraz", "Hard"),
        ("Novak Djokovic", "Rafael Nadal", "Clay"),
        ("Carlos Alcaraz", "Daniil Medvedev", "Grass"),
    ]

    print("\nExample predictions:")
    for p1, p2, surface in example_matches:
        pred = elo.predict_match(p1, p2, surface)

        # Check if players exist
        if pred[p1] == 0.5 and elo.get_player_rating(p1).matches_played == 0:
            print(f"\n  {p1} vs {p2} on {surface}: Players not found in database")
            continue

        print(f"\n  {p1} vs {p2} on {surface}:")
        print(f"    {p1}: {pred[p1]*100:.1f}% (Elo: {pred.get(f'{p1}_elo', 'N/A'):.0f})")
        print(f"    {p2}: {pred[p2]*100:.1f}% (Elo: {pred.get(f'{p2}_elo', 'N/A'):.0f})")

        # If odds were available, calculate EV
        # Example: if bookmaker offers 2.10 on player 1
        sample_odds = 2.10
        ev = calculate_ev(pred[p1], sample_odds)
        if ev > 0:
            print(f"    At odds {sample_odds}: EV = +{ev:.1f}% ✓")
        else:
            print(f"    At odds {sample_odds}: EV = {ev:.1f}%")

    # Step 6: Fetch Live Odds (if API key provided)
    if args.odds_api_key:
        print("\n" + "=" * 60)
        print("LIVE ODDS & VALUE BETS")
        print("=" * 60)

        odds_client = OddsAPIClient(args.odds_api_key)
        live_odds = odds_client.get_tennis_odds()

        if live_odds:
            odds_df = odds_client.parse_odds_response(live_odds)
            print(f"\nFound {len(odds_df)} upcoming matches with odds")

            # Calculate value for each match
            value_bets = []

            for _, match in odds_df.iterrows():
                p1, p2 = match['player_a'], match['player_b']

                # Try to get prediction
                pred = elo.predict_match(p1, p2, 'Hard')  # Default to hard

                for player, odds, prob in [
                    (p1, match['odds_a'], pred.get(p1, 0.5)),
                    (p2, match['odds_b'], pred.get(p2, 0.5))
                ]:
                    ev = calculate_ev(prob, odds)
                    if ev > 2.0:  # Minimum 2% EV
                        value_bets.append({
                            'match': f"{p1} vs {p2}",
                            'bet_on': player,
                            'model_prob': prob,
                            'odds': odds,
                            'ev': ev
                        })

            if value_bets:
                print("\n+EV Betting Opportunities:")
                for bet in sorted(value_bets, key=lambda x: -x['ev']):
                    print(f"\n  {bet['match']}")
                    print(f"    Bet on: {bet['bet_on']}")
                    print(f"    Model prob: {bet['model_prob']*100:.1f}%")
                    print(f"    Odds: {bet['odds']}")
                    print(f"    Expected Value: +{bet['ev']:.1f}%")
            else:
                print("\nNo value bets found with >2% edge")
        else:
            print("Failed to fetch live odds")
    else:
        print("\n[TIP] Add --odds-api-key YOUR_KEY to fetch live odds")
        print("      Get a free API key at: https://the-odds-api.com/")

    print("\n" + "=" * 60)


def predict_match_interactive():
    """Interactive match prediction."""
    print("\nLoading data for predictions...")
    matches = load_matches(start_year=2015, end_year=2024)

    if len(matches) == 0:
        print("No data found. Run: python main.py --download")
        return

    elo = TennisElo()
    elo.process_matches(matches)

    print("\nEnter match details (or 'quit' to exit):\n")

    while True:
        player_a = input("Player A: ").strip()
        if player_a.lower() == 'quit':
            break

        player_b = input("Player B: ").strip()
        surface = input("Surface (hard/clay/grass): ").strip() or 'Hard'
        odds_input = input("Odds for Player A (optional): ").strip()

        pred = elo.predict_match(player_a, player_b, surface)

        print(f"\n--- Prediction ---")
        print(f"{player_a}: {pred[player_a]*100:.1f}%")
        print(f"{player_b}: {pred[player_b]*100:.1f}%")

        if odds_input:
            try:
                odds = float(odds_input)
                ev = calculate_ev(pred[player_a], odds)
                implied = implied_probability(odds) * 100
                edge = pred[player_a] * 100 - implied

                print(f"\nAt odds {odds}:")
                print(f"  Implied probability: {implied:.1f}%")
                print(f"  Your edge: {edge:+.1f}%")
                print(f"  Expected Value: {ev:+.1f}%")

                if ev > 0:
                    print("  → This is a +EV bet! ✓")
                else:
                    print("  → Not a value bet")
            except ValueError:
                print("Invalid odds format")

        print()


def run_backtest():
    """Run historical backtest with odds data."""
    parser = argparse.ArgumentParser(description='Tennis Model Backtest')
    parser.add_argument('--download', action='store_true', help='Download odds data')
    parser.add_argument('--start-year', type=int, default=2019, help='Start year')
    parser.add_argument('--end-year', type=int, default=2024, help='End year')
    parser.add_argument('--min-ev', type=float, default=3.0, help='Minimum EV%% to bet')
    parser.add_argument('--min-odds', type=float, default=1.30, help='Minimum odds')
    parser.add_argument('--max-odds', type=float, default=4.0, help='Maximum odds')
    parser.add_argument('--stake', type=float, default=100.0, help='Stake per bet')
    parser.add_argument('--odds-source', choices=['Avg', 'Max', 'B365', 'PS'],
                        default='Avg', help='Which odds to use')
    parser.add_argument('--surface', type=str, help='Filter by surface')
    parser.add_argument('--detailed', action='store_true', help='Show detailed report')

    # Parse only backtest args (skip 'backtest' command itself)
    import sys
    args = parser.parse_args(sys.argv[2:])

    print("=" * 60)
    print("TENNIS MODEL BACKTEST")
    print("=" * 60)

    # Download odds data if requested
    if args.download:
        print(f"\n[1] Downloading odds data ({args.start_year}-{args.end_year})...")
        download_odds_data(args.start_year, args.end_year)

    # Load matches with odds
    print(f"\n[1] Loading matches with odds ({args.start_year}-{args.end_year})...")
    matches = load_matches_with_odds(
        start_year=args.start_year,
        end_year=args.end_year,
        download=args.download
    )

    if len(matches) == 0:
        print("No odds data found. Run with --download to fetch data.")
        print("\nExample: python main.py backtest --download")
        return

    print(f"    Loaded {len(matches):,} matches with odds")

    # Show odds columns available
    odds_cols = [c for c in matches.columns if any(x in c for x in ['W', 'L']) and
                 any(x in c for x in ['Avg', 'Max', 'B365', 'PS'])]
    print(f"    Available odds: {', '.join(odds_cols[:8])}")

    # Initialize backtester
    print(f"\n[2] Running backtest with settings:")
    print(f"    Min EV: {args.min_ev}%")
    print(f"    Odds range: {args.min_odds} - {args.max_odds}")
    print(f"    Stake: ${args.stake}")
    print(f"    Odds source: {args.odds_source}")
    if args.surface:
        print(f"    Surface filter: {args.surface}")

    bt = Backtester(
        min_ev=args.min_ev,
        min_odds=args.min_odds,
        max_odds=args.max_odds,
        stake_size=args.stake,
        odds_column=args.odds_source
    )

    # Run backtest
    print("\n[3] Processing matches...")
    results = bt.run(matches, surface_filter=args.surface, verbose=True)

    # Print results
    print(generate_report(results, detailed=args.detailed))

    # Surface breakdown
    if not args.surface:
        print("\n[4] Performance by Surface:")
        print("-" * 40)
        surface_results = bt.analyze_by_surface(matches, verbose=False)
        for surface, res in surface_results.items():
            if res.total_bets > 0:
                print(f"    {surface:6} | {res.total_bets:4} bets | "
                      f"ROI: {res.roi_percent:+.1f}% | "
                      f"Win: {res.win_rate:.0f}%")

    # EV threshold analysis
    print("\n[5] Performance by EV Threshold:")
    print("-" * 40)
    ev_results = bt.analyze_by_ev_threshold(matches, verbose=False)
    for ev_thresh, res in ev_results.items():
        if res.total_bets > 0:
            print(f"    {ev_thresh:8} | {res.total_bets:4} bets | "
                  f"ROI: {res.roi_percent:+.1f}% | "
                  f"Win: {res.win_rate:.0f}%")

    # Odds range analysis
    print("\n[6] Performance by Odds Range:")
    print("-" * 40)
    odds_results = bt.analyze_by_odds_range(matches, verbose=False)
    for odds_range, res in odds_results.items():
        if res.total_bets > 0:
            print(f"    {odds_range:8} | {res.total_bets:4} bets | "
                  f"ROI: {res.roi_percent:+.1f}% | "
                  f"Win: {res.win_rate:.0f}%")

    print("\n" + "=" * 60)

    # Show some sample bets
    if results.bets and args.detailed:
        print("\n[7] Sample Recent Bets:")
        print("-" * 60)
        for bet in results.bets[-10:]:
            result = "WIN" if bet.won else "LOSS"
            print(f"    {bet.match_date[:10] if len(bet.match_date) >= 10 else bet.match_date} | "
                  f"{bet.player_bet[:20]:<20} | "
                  f"@{bet.odds:.2f} | "
                  f"EV:{bet.ev_percent:.1f}% | "
                  f"{result:4} | ${bet.profit:+.2f}")


def run_backtest_compare():
    """Run model comparison backtest."""
    parser = argparse.ArgumentParser(description='Tennis Model Comparison Backtest')
    parser.add_argument('--download', action='store_true', help='Download odds data')
    parser.add_argument('--start-year', type=int, default=2019, help='Start year')
    parser.add_argument('--end-year', type=int, default=2024, help='End year')
    parser.add_argument('--min-ev', type=float, default=3.0, help='Minimum EV%')
    parser.add_argument('--min-odds', type=float, default=1.30, help='Minimum odds')
    parser.add_argument('--max-odds', type=float, default=4.0, help='Maximum odds')
    parser.add_argument('--by-surface', action='store_true', help='Break down by surface')

    import sys
    args = parser.parse_args(sys.argv[2:])

    print("=" * 70)
    print("ENHANCED MODEL COMPARISON")
    print("=" * 70)

    # Download if needed
    if args.download:
        print(f"\n[1] Downloading odds data ({args.start_year}-{args.end_year})...")
        download_odds_data(args.start_year, args.end_year)

    # Load data
    print(f"\n[1] Loading matches with odds ({args.start_year}-{args.end_year})...")
    matches = load_matches_with_odds(
        start_year=args.start_year,
        end_year=args.end_year,
        download=args.download
    )

    if len(matches) == 0:
        print("No odds data found. Run with --download to fetch data.")
        return

    print(f"    Loaded {len(matches):,} matches with odds")

    # Run comparison
    bt = EnhancedBacktester(
        min_ev=args.min_ev,
        min_odds=args.min_odds,
        max_odds=args.max_odds
    )

    if args.by_surface:
        bt.run_surface_comparison(matches, verbose=True)
    else:
        results = bt.run_comparison(matches, verbose=True)
        bt.print_comparison(results)


def run_predict():
    """Run predictions for upcoming matches or manual input."""
    parser = argparse.ArgumentParser(description='Tennis Match Predictions')
    parser.add_argument('--tournament', type=str, help='Tournament name')
    parser.add_argument('--surface', type=str, default='Hard', help='Surface type')
    parser.add_argument('--player-a', type=str, help='First player name')
    parser.add_argument('--player-b', type=str, help='Second player name')
    parser.add_argument('--odds-a', type=float, help='Odds for player A')
    parser.add_argument('--odds-b', type=float, help='Odds for player B')
    parser.add_argument('--odds-api-key', type=str, help='The Odds API key for live odds')
    parser.add_argument('--start-year', type=int, default=2020, help='Training data start year')

    import sys
    args = parser.parse_args(sys.argv[2:])

    print("=" * 60)
    print("ENHANCED TENNIS PREDICTOR")
    print("=" * 60)

    # Load training data
    print(f"\n[1] Loading training data ({args.start_year}-2024)...")
    matches = load_matches(start_year=args.start_year, end_year=2024)

    if len(matches) == 0:
        print("No data found. Run: python main.py --download")
        return

    print(f"    Loaded {len(matches):,} matches")

    # Build ensemble predictor
    print("\n[2] Building ensemble model...")
    ensemble = EnsemblePredictor(use_glicko=False)
    ensemble.initialize_from_matches(matches)
    print("    Model ready")

    # Create predictor
    predictor = UpcomingMatchPredictor(ensemble)

    if args.player_a and args.player_b:
        # Manual prediction
        print(f"\n[3] Predicting: {args.player_a} vs {args.player_b} on {args.surface}")

        pred = predictor.predict_manual(
            args.player_a,
            args.player_b,
            args.surface,
            args.odds_a,
            args.odds_b,
            args.tournament
        )

        predictor.print_single_prediction(pred)

    elif args.odds_api_key:
        # Live odds prediction
        print("\n[3] Fetching live odds...")
        predictor.set_odds_client(args.odds_api_key)

        predictions = predictor.get_predictions(
            tournament_name=args.tournament,
            surface=args.surface,
            min_ev=2.0
        )

        if len(predictions) > 0:
            predictor.print_predictions(predictions, show_breakdown=True)

            value_bets = predictions[predictions['is_value']]
            if len(value_bets) > 0:
                print(f"\n*** Found {len(value_bets)} value bet(s)! ***")
        else:
            print("No upcoming matches found.")

    else:
        # Interactive mode
        print("\n[3] Interactive prediction mode")
        print("    Enter match details (or 'quit' to exit)\n")

        while True:
            player_a = input("Player A: ").strip()
            if player_a.lower() == 'quit':
                break

            player_b = input("Player B: ").strip()
            surface = input("Surface (hard/clay/grass) [hard]: ").strip() or 'Hard'
            odds_input = input("Odds for Player A (optional): ").strip()

            odds_a = float(odds_input) if odds_input else None
            odds_b_input = input("Odds for Player B (optional): ").strip()
            odds_b = float(odds_b_input) if odds_b_input else None

            pred = predictor.predict_manual(player_a, player_b, surface, odds_a, odds_b)
            predictor.print_single_prediction(pred)
            print()


def analyze_player_cmd():
    """Analyze a specific player's profile."""
    parser = argparse.ArgumentParser(description='Player Analysis')
    parser.add_argument('player', type=str, help='Player name')
    parser.add_argument('--start-year', type=int, default=2020, help='Start year')

    import sys
    args = parser.parse_args(sys.argv[2:])

    print(f"\n=== PLAYER ANALYSIS: {args.player} ===\n")

    # Load data
    matches = load_matches(start_year=args.start_year, end_year=2024)
    if len(matches) == 0:
        print("No data found.")
        return

    # Build models
    elo = EnhancedElo()
    elo.process_matches(matches)

    surface_analyzer = SurfaceAnalyzer()
    surface_analyzer.process_matches_df(matches)

    serve_tracker = ServeReturnTracker()
    serve_tracker.process_matches_df(matches)

    # Get player rating
    rating = elo.get_player_rating(args.player)

    print(f"Overall Elo: {rating.overall:.0f}")
    print(f"  Hard:  {rating.hard:.0f} ({rating.hard_matches} matches)")
    print(f"  Clay:  {rating.clay:.0f} ({rating.clay_matches} matches)")
    print(f"  Grass: {rating.grass:.0f} ({rating.grass_matches} matches)")
    print(f"Total matches: {rating.matches_played}")
    print(f"Form adjustment: {rating.form_adjustment:+.0f}")

    # Surface profile
    print(f"\nSurface Performance:")
    for surface in ['hard', 'clay', 'grass']:
        profile = surface_analyzer.get_surface_profile(args.player, surface)
        specialist = " (SPECIALIST)" if profile.is_specialist else ""
        print(f"  {surface.capitalize()}: {profile.win_rate*100:.1f}% ({profile.matches} matches){specialist}")

    # Serve stats
    print(f"\nServe Statistics (rolling {serve_tracker.window} matches):")
    for surface in ['hard', 'clay', 'grass']:
        stats = serve_tracker.get_player_stats(args.player, surface)
        if stats.matches_in_sample > 0:
            print(f"\n  {surface.capitalize()}:")
            print(f"    1st Serve %: {stats.first_serve_pct*100:.1f}%")
            print(f"    1st Serve Won: {stats.first_serve_won_pct*100:.1f}%")
            print(f"    2nd Serve Won: {stats.second_serve_won_pct*100:.1f}%")
            print(f"    Service Hold: {stats.service_hold_pct*100:.1f}%")
            print(f"    BP Save: {stats.bp_save_pct*100:.1f}%")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'interactive':
            predict_match_interactive()
        elif cmd == 'backtest':
            run_backtest()
        elif cmd == 'backtest-compare':
            run_backtest_compare()
        elif cmd == 'predict':
            run_predict()
        elif cmd == 'analyze-player':
            analyze_player_cmd()
        else:
            main()
    else:
        main()
