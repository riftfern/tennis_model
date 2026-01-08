"""
Convert 2025 tennis-data.co.uk odds data to Sackmann format.

This script:
1. Loads historical Sackmann data to build a player lookup database
2. Reads the 2025 odds Excel file
3. Converts abbreviated names to full names with proper IDs
4. Maps round names to Sackmann format (R32, R16, QF, SF, F)
5. Creates proper tourney_id values
6. Outputs in the standard Sackmann CSV format
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re


def build_player_database(data_path: Path, start_year: int = 2020, end_year: int = 2024) -> dict:
    """
    Build a player database from historical Sackmann data.
    Maps abbreviated names (e.g., 'Vukic A.') to full player info.
    """
    all_players = {}

    for year in range(start_year, end_year + 1):
        filepath = data_path / f'atp_matches_{year}.csv'
        if filepath.exists():
            df = pd.read_csv(filepath, low_memory=False)

            for _, row in df.iterrows():
                # Process winner
                if pd.notna(row.get('winner_name')) and pd.notna(row.get('winner_id')):
                    name = str(row['winner_name']).strip()
                    parts = name.split()
                    if len(parts) >= 2:
                        first_name = parts[0]
                        last_name = ' '.join(parts[1:])
                        abbrev = f"{last_name} {first_name[0]}."

                        all_players[abbrev.lower()] = {
                            'full_name': name,
                            'id': int(row['winner_id']),
                            'hand': row.get('winner_hand', 'R'),
                            'ht': row.get('winner_ht'),
                            'ioc': row.get('winner_ioc', 'UNK')
                        }

                # Process loser
                if pd.notna(row.get('loser_name')) and pd.notna(row.get('loser_id')):
                    name = str(row['loser_name']).strip()
                    parts = name.split()
                    if len(parts) >= 2:
                        first_name = parts[0]
                        last_name = ' '.join(parts[1:])
                        abbrev = f"{last_name} {first_name[0]}."

                        all_players[abbrev.lower()] = {
                            'full_name': name,
                            'id': int(row['loser_id']),
                            'hand': row.get('loser_hand', 'R'),
                            'ht': row.get('loser_ht'),
                            'ioc': row.get('loser_ioc', 'UNK')
                        }

    # Add manual mappings for edge cases and new players
    manual_mappings = {
        # O'Connell variations
        "o connell c.": {'full_name': 'Christopher Oconnell', 'id': 106331, 'hand': 'R', 'ht': 183.0, 'ioc': 'AUS'},
        "oconnell c.": {'full_name': 'Christopher Oconnell', 'id': 106331, 'hand': 'R', 'ht': 183.0, 'ioc': 'AUS'},
        # Mpetshi Perricard
        "mpetshi g.": {'full_name': 'Giovanni Mpetshi Perricard', 'id': 210196, 'hand': 'R', 'ht': 203.0, 'ioc': 'FRA'},
        "mpetshi perricard g.": {'full_name': 'Giovanni Mpetshi Perricard', 'id': 210196, 'hand': 'R', 'ht': 203.0, 'ioc': 'FRA'},
        # Auger-Aliassime
        "auger aliassime f.": {'full_name': 'Felix Auger Aliassime', 'id': 200000, 'hand': 'R', 'ht': 193.0, 'ioc': 'CAN'},
        "auger-aliassime f.": {'full_name': 'Felix Auger Aliassime', 'id': 200000, 'hand': 'R', 'ht': 193.0, 'ioc': 'CAN'},
        # Davidovich Fokina
        "davidovich fokina a.": {'full_name': 'Alejandro Davidovich Fokina', 'id': 200221, 'hand': 'R', 'ht': 183.0, 'ioc': 'ESP'},
        # Carballes Baena
        "carballes baena r.": {'full_name': 'Roberto Carballes Baena', 'id': 105155, 'hand': 'R', 'ht': 180.0, 'ioc': 'ESP'},
        # Moro Canas
        "moro canas a.": {'full_name': 'Alejandro Moro Canas', 'id': 209325, 'hand': 'R', 'ht': 180.0, 'ioc': 'ESP'},
        # Huesler (Swiss)
        "huesler m.a.": {'full_name': 'Marc Andrea Huesler', 'id': 125796, 'hand': 'L', 'ht': 196.0, 'ioc': 'SUI'},
        # Bu Yunchaokete
        "bu y.": {'full_name': 'Yunchaokete Bu', 'id': 210220, 'hand': 'R', 'ht': 188.0, 'ioc': 'CHN'},
        # Struff
        "struff j.l.": {'full_name': 'Jan Lennard Struff', 'id': 105024, 'hand': 'R', 'ht': 196.0, 'ioc': 'GER'},
        "struff j-l.": {'full_name': 'Jan Lennard Struff', 'id': 105024, 'hand': 'R', 'ht': 196.0, 'ioc': 'GER'},
        # Tien Learner
        "tien l.": {'full_name': 'Learner Tien', 'id': 210283, 'hand': 'R', 'ht': 185.0, 'ioc': 'USA'},
        # Mensik
        "mensik j.": {'full_name': 'Jakub Mensik', 'id': 210169, 'hand': 'R', 'ht': 193.0, 'ioc': 'CZE'},
        # Cazaux
        "cazaux a.": {'full_name': 'Arthur Cazaux', 'id': 209808, 'hand': 'R', 'ht': 178.0, 'ioc': 'FRA'},
        # Zhang Zhizhen
        "zhang zh.": {'full_name': 'Zhizhen Zhang', 'id': 111190, 'hand': 'R', 'ht': 193.0, 'ioc': 'CHN'},
        # Gomez Federico
        "gomez f.": {'full_name': 'Federico Gomez', 'id': 210305, 'hand': 'R', 'ht': 185.0, 'ioc': 'ARG'},
        # Basavareddy
        "basavareddy n.": {'full_name': 'Nishesh Basavareddy', 'id': 210388, 'hand': 'R', 'ht': 180.0, 'ioc': 'USA'},
        # Passaro Francesco
        "passaro f.": {'full_name': 'Francesco Passaro', 'id': 209433, 'hand': 'R', 'ht': 183.0, 'ioc': 'ITA'},
        # Wong Coleman
        "wong c.": {'full_name': 'Coleman Wong', 'id': 210159, 'hand': 'R', 'ht': 183.0, 'ioc': 'HKG'},
        # Diaz Acosta Facundo
        "diaz acosta f.": {'full_name': 'Facundo Diaz Acosta', 'id': 209730, 'hand': 'R', 'ht': 180.0, 'ioc': 'ARG'},
        # Comesana
        "comesana f.": {'full_name': 'Francisco Comesana', 'id': 209815, 'hand': 'R', 'ht': 188.0, 'ioc': 'ARG'},
        # Carreno Busta
        "carreno busta p.": {'full_name': 'Pablo Carreno Busta', 'id': 105297, 'hand': 'R', 'ht': 188.0, 'ioc': 'ESP'},
        # Ugo Carabelli
        "ugo carabelli c.": {'full_name': 'Camilo Ugo Carabelli', 'id': 209421, 'hand': 'R', 'ht': 185.0, 'ioc': 'ARG'},
        # Seyboth Wild
        "seyboth wild t.": {'full_name': 'Thiago Seyboth Wild', 'id': 205734, 'hand': 'R', 'ht': 185.0, 'ioc': 'BRA'},
        # Schoolkate
        "schoolkate t.": {'full_name': 'Tristan Schoolkate', 'id': 210392, 'hand': 'R', 'ht': 190.0, 'ioc': 'AUS'},
        # Becroft Isaac
        "becroft i.": {'full_name': 'Isaac Becroft', 'id': 210547, 'hand': 'R', 'ht': 185.0, 'ioc': 'NZL'},
        # Kukushkin
        "kukushkin m.": {'full_name': 'Mikhail Kukushkin', 'id': 105159, 'hand': 'R', 'ht': 183.0, 'ioc': 'KAZ'},
        # Van Assche
        "van assche l.": {'full_name': 'Luca Van Assche', 'id': 209414, 'hand': 'R', 'ht': 178.0, 'ioc': 'FRA'},
        # Huesler
        "huesler m.a.": {'full_name': 'Marc Andrea Huesler', 'id': 125796, 'hand': 'L', 'ht': 196.0, 'ioc': 'SUI'},
        # Fucsovics
        "fucsovics m.": {'full_name': 'Marton Fucsovics', 'id': 105916, 'hand': 'R', 'ht': 188.0, 'ioc': 'HUN'},
        # Fils
        "fils a.": {'full_name': 'Arthur Fils', 'id': 210117, 'hand': 'R', 'ht': 190.0, 'ioc': 'FRA'},
    }

    all_players.update(manual_mappings)

    return all_players


def get_player_info(abbrev_name: str, player_db: dict) -> dict:
    """
    Look up player info from abbreviated name.
    Returns default values if not found.
    """
    lookup_key = str(abbrev_name).lower().strip()

    if lookup_key in player_db:
        return player_db[lookup_key]

    # Try alternative formats
    # Remove extra spaces
    alt_key = ' '.join(lookup_key.split())
    if alt_key in player_db:
        return player_db[alt_key]

    # Try with hyphen instead of space
    alt_key = lookup_key.replace(' ', '-')
    if alt_key in player_db:
        return player_db[alt_key]

    # If not found, create a fallback entry
    return {
        'full_name': abbrev_name,  # Keep original as fallback
        'id': abs(hash(abbrev_name)) % 100000 + 200000,
        'hand': 'R',
        'ht': None,
        'ioc': 'UNK'
    }


def map_round_name(round_name: str, draw_size: int = 32) -> str:
    """
    Map tennis-data.co.uk round names to Sackmann format.
    """
    round_map = {
        '1st Round': 'R32' if draw_size >= 32 else 'R16',
        '2nd Round': 'R16' if draw_size >= 32 else 'QF',
        '3rd Round': 'R16',
        '4th Round': 'QF',
        'Quarterfinals': 'QF',
        'Semifinals': 'SF',
        'The Final': 'F',
        'Round Robin': 'RR',
    }
    return round_map.get(round_name, round_name)


def create_tourney_id(location: str, date: pd.Timestamp) -> str:
    """Create a tourney_id in Sackmann format: YYYY-XXXX"""
    year = date.year
    # Generate a hash-based number from the location
    loc_hash = abs(hash(location.lower())) % 10000
    return f"{year}-{loc_hash:04d}"


def construct_score(row: pd.Series) -> str:
    """Construct a score string from set scores."""
    score_parts = []
    for i in range(1, 6):
        w = row.get(f'W{i}')
        l = row.get(f'L{i}')
        if pd.notna(w) and pd.notna(l):
            try:
                w_score = int(w)
                l_score = int(l)
                # Check for tiebreak (7-6)
                if w_score == 7 and l_score == 6:
                    score_parts.append(f"7-6")
                elif w_score == 6 and l_score == 7:
                    score_parts.append(f"6-7")
                else:
                    score_parts.append(f"{w_score}-{l_score}")
            except (ValueError, TypeError):
                pass

    result = " ".join(score_parts)

    # Check for retirement or walkover in comment
    comment = str(row.get('Comment', '')).lower()
    if 'retired' in comment or 'ret' in comment:
        result += " RET"
    elif 'walkover' in comment or 'w/o' in comment:
        result = "W/O"

    return result


def convert_odds_to_sackmann():
    """Convert 2025 odds data to Sackmann format."""
    print("Converting 2025 Odds Data to Match Data format...")

    data_path = Path('data')

    # Build player database
    print("Building player database from historical data...")
    player_db = build_player_database(data_path)
    print(f"  Loaded {len(player_db)} player entries")

    # Load the odds data
    try:
        df_odds = pd.read_excel(data_path / 'odds_2025.xlsx')
    except Exception as e:
        print(f"Error loading Excel: {e}")
        return

    print(f"  Loaded {len(df_odds)} matches from odds file")

    # Track unmatched players
    unmatched_players = set()

    # Process each row
    matches = []
    match_num = 1

    for _, row in df_odds.iterrows():
        # Parse date
        date = pd.to_datetime(row['Date'])

        # Get player info
        winner_abbrev = str(row['Winner']).strip()
        loser_abbrev = str(row['Loser']).strip()

        winner_info = get_player_info(winner_abbrev, player_db)
        loser_info = get_player_info(loser_abbrev, player_db)

        if winner_info['ioc'] == 'UNK':
            unmatched_players.add(winner_abbrev)
        if loser_info['ioc'] == 'UNK':
            unmatched_players.add(loser_abbrev)

        # Map series to tourney level
        series = str(row.get('Series', 'ATP250'))
        series_map = {
            'ATP250': 'A',
            'ATP500': 'A',
            'Grand Slam': 'G',
            'Masters 1000': 'M',
            'Masters Cup': 'F',
        }
        tourney_level = series_map.get(series, 'A')

        # Determine draw size
        if tourney_level == 'G':
            draw_size = 128
        elif tourney_level == 'M':
            draw_size = 64
        else:
            draw_size = 32

        # Build match record
        match = {
            'tourney_id': create_tourney_id(str(row['Location']), date),
            'tourney_name': str(row['Tournament']),
            'surface': str(row['Surface']),
            'draw_size': draw_size,
            'tourney_level': tourney_level,
            'tourney_date': date.strftime('%Y%m%d'),
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
            'score': construct_score(row),
            'best_of': int(row['Best of']) if pd.notna(row.get('Best of')) else 3,
            'round': map_round_name(str(row['Round']), draw_size),
            'minutes': None,
            # Stats - not available in odds data
            'w_ace': None, 'w_df': None, 'w_svpt': None, 'w_1stIn': None,
            'w_1stWon': None, 'w_2ndWon': None, 'w_SvGms': None,
            'w_bpSaved': None, 'w_bpFaced': None,
            'l_ace': None, 'l_df': None, 'l_svpt': None, 'l_1stIn': None,
            'l_1stWon': None, 'l_2ndWon': None, 'l_SvGms': None,
            'l_bpSaved': None, 'l_bpFaced': None,
            # Rankings
            'winner_rank': int(row['WRank']) if pd.notna(row.get('WRank')) else None,
            'winner_rank_points': int(row['WPts']) if pd.notna(row.get('WPts')) else None,
            'loser_rank': int(row['LRank']) if pd.notna(row.get('LRank')) else None,
            'loser_rank_points': int(row['LPts']) if pd.notna(row.get('LPts')) else None,
        }

        matches.append(match)
        match_num += 1

    # Create DataFrame
    df_new = pd.DataFrame(matches)

    # Report unmatched players
    if unmatched_players:
        print(f"\nWarning: {len(unmatched_players)} unmatched players (using fallback):")
        for p in sorted(unmatched_players)[:10]:
            print(f"  - {p}")
        if len(unmatched_players) > 10:
            print(f"  ... and {len(unmatched_players) - 10} more")

    # Save
    output_path = data_path / 'atp_matches_2025.csv'
    df_new.to_csv(output_path, index=False)
    print(f"\nSuccessfully saved {len(df_new)} matches to {output_path}")

    # Show sample
    print("\nSample of converted data:")
    sample_cols = ['tourney_id', 'tourney_name', 'winner_name', 'loser_name', 'score', 'round']
    print(df_new[sample_cols].head().to_string())


if __name__ == "__main__":
    convert_odds_to_sackmann()
