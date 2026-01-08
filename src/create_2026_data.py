"""
Create 2026 ATP match data from actual tournament results.

This script creates match data for the 2026 season based on actual
tournament results. It uses the player database from historical data
to ensure proper player IDs, country codes, and other metadata.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import the player database builder from convert_2025_data
sys.path.insert(0, str(Path(__file__).parent))
from convert_2025_data import build_player_database, get_player_info


def create_2026_data():
    """Create 2026 season data from actual tournament results."""
    print("Creating 2026 Season Data with correct results...")

    data_path = Path('data')

    # Build player database
    print("Building player database from historical data...")
    player_db = build_player_database(data_path)
    print(f"  Loaded {len(player_db)} player entries")

    # Sackmann columns
    columns = [
        'tourney_id', 'tourney_name', 'surface', 'draw_size', 'tourney_level',
        'tourney_date', 'match_num', 'winner_id', 'winner_seed', 'winner_entry',
        'winner_name', 'winner_hand', 'winner_ht', 'winner_ioc', 'winner_age',
        'loser_id', 'loser_seed', 'loser_entry', 'loser_name', 'loser_hand',
        'loser_ht', 'loser_ioc', 'loser_age', 'score', 'best_of', 'round',
        'minutes', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
        'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt', 'l_1stIn',
        'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced',
        'winner_rank', 'winner_rank_points', 'loser_rank', 'loser_rank_points'
    ]

    matches = []
    match_num = 1

    def get_player_full_info(name_input):
        """Look up player info, trying various name formats."""
        # Try direct full name lookup first
        name = str(name_input).strip()

        # Build abbreviated form for lookup
        parts = name.split()
        if len(parts) >= 2:
            first_name = parts[0]
            last_name = ' '.join(parts[1:])
            abbrev = f"{last_name} {first_name[0]}."

            info = get_player_info(abbrev, player_db)
            if info['ioc'] != 'UNK':
                return info

        # Try with last name first
        if len(parts) >= 2:
            first_name = parts[-1]
            last_name = ' '.join(parts[:-1])
            abbrev = f"{last_name} {first_name[0]}."

            info = get_player_info(abbrev, player_db)
            if info['ioc'] != 'UNK':
                return info

        # Fallback with hash-based ID
        return {
            'full_name': name,
            'id': abs(hash(name)) % 100000 + 200000,
            'hand': 'R',
            'ht': None,
            'ioc': 'UNK'
        }

    def add_match(tourney_id, tourney_name, surface, date, winner, loser, score, round_name):
        nonlocal match_num

        winner_info = get_player_full_info(winner)
        loser_info = get_player_full_info(loser)

        matches.append({
            'tourney_id': tourney_id,
            'tourney_name': tourney_name,
            'surface': surface,
            'draw_size': 32,
            'tourney_level': 'A',
            'tourney_date': date,
            'match_num': match_num,
            'winner_id': winner_info['id'],
            'winner_seed': None,
            'winner_entry': None,
            'winner_name': winner_info['full_name'],
            'winner_hand': winner_info['hand'],
            'winner_ht': winner_info['ht'],
            'winner_ioc': winner_info['ioc'],
            'winner_age': None,
            'loser_id': loser_info['id'],
            'loser_seed': None,
            'loser_entry': None,
            'loser_name': loser_info['full_name'],
            'loser_hand': loser_info['hand'],
            'loser_ht': loser_info['ht'],
            'loser_ioc': loser_info['ioc'],
            'loser_age': None,
            'score': score,
            'best_of': 3,
            'round': round_name,
            'minutes': None,
            'w_ace': None, 'w_df': None, 'w_svpt': None, 'w_1stIn': None,
            'w_1stWon': None, 'w_2ndWon': None, 'w_SvGms': None,
            'w_bpSaved': None, 'w_bpFaced': None,
            'l_ace': None, 'l_df': None, 'l_svpt': None, 'l_1stIn': None,
            'l_1stWon': None, 'l_2ndWon': None, 'l_SvGms': None,
            'l_bpSaved': None, 'l_bpFaced': None,
            'winner_rank': None, 'winner_rank_points': None,
            'loser_rank': None, 'loser_rank_points': None,
        })
        match_num += 1

    # =========================================================================
    # BRISBANE 2026 (Hard, ATP 250) - Corrected Results from ATP Tour
    # =========================================================================
    tourney_id = "2026-0339"
    tourney_name = "Brisbane International"
    surface = "Hard"

    # Round of 32 - Sun, 04 January
    add_match(tourney_id, tourney_name, surface, "20260104",
              "Learner Tien", "Camilo Ugo Carabelli", "7-6(4) 6-3", "R32")
    add_match(tourney_id, tourney_name, surface, "20260104",
              "Frances Tiafoe", "Aleksandar Vukic", "6-2 6-2", "R32")

    # Round of 32 - Mon, 05 January
    add_match(tourney_id, tourney_name, surface, "20260105",
              "Daniil Medvedev", "Marton Fucsovics", "6-2 6-3", "R32")
    add_match(tourney_id, tourney_name, surface, "20260105",
              "Jiri Lehecka", "Tomas Machac", "6-4 6-7(5) 6-2", "R32")
    add_match(tourney_id, tourney_name, surface, "20260105",
              "Quentin Halys", "Alexei Popyrin", "5-7 6-3 6-4", "R32")
    add_match(tourney_id, tourney_name, surface, "20260105",
              "Sebastian Korda", "Valentin Vacherot", "7-6(1) 6-3", "R32")
    add_match(tourney_id, tourney_name, surface, "20260105",
              "Kamil Majchrzak", "Daniel Altmaier", "7-6(4) 6-0", "R32")
    add_match(tourney_id, tourney_name, surface, "20260105",
              "Alex Michelsen", "James Duckworth", "6-7(4) 7-6(2) 6-3", "R32")
    add_match(tourney_id, tourney_name, surface, "20260105",
              "Reilly Opelka", "Dane Sweeny", "6-3 7-5", "R32")

    # Round of 32 - Tue, 06 January
    add_match(tourney_id, tourney_name, surface, "20260106",
              "Brandon Nakashima", "Alejandro Davidovich Fokina", "7-6(4) 6-4", "R32")
    add_match(tourney_id, tourney_name, surface, "20260106",
              "Giovanni Mpetshi Perricard", "Tommy Paul", "7-6(2) 3-6 7-6(6)", "R32")
    add_match(tourney_id, tourney_name, surface, "20260106",
              "Raphael Collignon", "Denis Shapovalov", "6-4 6-2", "R32")
    add_match(tourney_id, tourney_name, surface, "20260106",
              "Cameron Norrie", "Ugo Humbert", "1-6 7-6(6) 7-5", "R32")
    add_match(tourney_id, tourney_name, surface, "20260106",
              "Grigor Dimitrov", "Pablo Carreno Busta", "6-3 6-2", "R32")
    add_match(tourney_id, tourney_name, surface, "20260106",
              "Rinky Hijikata", "Adam Walton", "6-3 6-2", "R32")
    add_match(tourney_id, tourney_name, surface, "20260106",
              "Aleksandar Kovacevic", "Nick Kyrgios", "6-3 6-4", "R32")

    # Round of 16 - Wed, 07 January
    add_match(tourney_id, tourney_name, surface, "20260107",
              "Daniil Medvedev", "Frances Tiafoe", "6-3 6-2", "R16")
    add_match(tourney_id, tourney_name, surface, "20260107",
              "Sebastian Korda", "Jiri Lehecka", "6-3 1-2 RET", "R16")
    add_match(tourney_id, tourney_name, surface, "20260107",
              "Alex Michelsen", "Learner Tien", "6-4 6-2", "R16")
    add_match(tourney_id, tourney_name, surface, "20260107",
              "Kamil Majchrzak", "Reilly Opelka", "6-7(2) 7-6(7) 7-6(8)", "R16")

    # =========================================================================
    # HONG KONG 2026 (Hard, ATP 250) - Corrected Results from ATP Tour
    # =========================================================================
    tourney_id = "2026-0336"
    tourney_name = "Hong Kong Tennis Open"
    surface = "Hard"

    # Round of 32 - Sun, 05 January
    add_match(tourney_id, tourney_name, surface, "20260105",
              "Nuno Borges", "Damir Dzumhur", "6-4 6-3", "R32")
    add_match(tourney_id, tourney_name, surface, "20260105",
              "Marin Cilic", "Adrian Mannarino", "6-3 6-2", "R32")
    add_match(tourney_id, tourney_name, surface, "20260105",
              "Yibing Wu", "Fabian Marozsan", "6-4 6-2", "R32")

    # Round of 32 - Mon, 06 January
    add_match(tourney_id, tourney_name, surface, "20260106",
              "Lorenzo Sonego", "Rei Sakamoto", "6-2 7-6(4)", "R32")
    add_match(tourney_id, tourney_name, surface, "20260106",
              "Gabriel Diallo", "Jesper De Jong", "6-4 7-6(7)", "R32")
    add_match(tourney_id, tourney_name, surface, "20260106",
              "Alexandre Muller", "Miomir Kecmanovic", "7-5 6-4", "R32")
    add_match(tourney_id, tourney_name, surface, "20260106",
              "Tomas Martin Etcheverry", "Valentin Royer", "6-4 7-5", "R32")
    add_match(tourney_id, tourney_name, surface, "20260106",
              "Marcos Giron", "Laslo Djere", "6-2 6-0", "R32")
    add_match(tourney_id, tourney_name, surface, "20260106",
              "Michael Mmoh", "Alejandro Tabilo", "7-5 6-4", "R32")
    add_match(tourney_id, tourney_name, surface, "20260106",
              "Juncheng Shang", "Francisco Comesana", "6-4 6-4", "R32")
    add_match(tourney_id, tourney_name, surface, "20260106",
              "Botic Van De Zandschulp", "Jan Lennard Struff", "6-3 1-6 6-1", "R32")
    add_match(tourney_id, tourney_name, surface, "20260106",
              "Coleman Wong", "Mariano Navone", "6-3 7-5", "R32")

    # Round of 16 - Tue, 07 January
    add_match(tourney_id, tourney_name, surface, "20260107",
              "Andrey Rublev", "Yibing Wu", "3-6 6-2 6-1", "R16")
    add_match(tourney_id, tourney_name, surface, "20260107",
              "Lorenzo Musetti", "Tomas Martin Etcheverry", "6-7(3) 6-2 6-4", "R16")
    add_match(tourney_id, tourney_name, surface, "20260107",
              "Coleman Wong", "Gabriel Diallo", "1-6 7-5 7-5", "R16")
    add_match(tourney_id, tourney_name, surface, "20260107",
              "Nuno Borges", "Marin Cilic", "7-5 6-3", "R16")

    # Create DataFrame
    df = pd.DataFrame(matches, columns=columns)

    # Save
    output_path = data_path / "atp_matches_2026.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSuccessfully created {output_path} with {len(df)} matches.")

    # Show sample
    print("\nSample of created data:")
    sample_cols = ['tourney_name', 'winner_name', 'loser_name', 'score', 'round']
    print(df[sample_cols].head(10).to_string())


if __name__ == "__main__":
    create_2026_data()
