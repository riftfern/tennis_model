"""
Momentum and "Vibes" Analysis.

Captures qualitative factors like clutch performance, recent dominance, 
and mental toughness (tiebreaks/deciders).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List

class MomentumTracker:
    def __init__(self, window_weeks: int = 52):
        self.window_weeks = window_weeks
        self.matches = pd.DataFrame()

    def process_matches_df(self, matches_df: pd.DataFrame):
        """Load match history."""
        self.matches = matches_df.copy()
        if 'tourney_date' in self.matches.columns:
            self.matches['tourney_date'] = pd.to_datetime(self.matches['tourney_date'])
            
    def get_momentum_stats(self, player: str, current_date: datetime = None) -> Dict:
        """
        Get momentum and 'vibes' stats for a player.
        """
        if current_date is None:
            current_date = datetime.now()
            
        start_date = current_date - timedelta(weeks=self.window_weeks)
        
        # Filter player matches
        pm = self.matches[
            ((self.matches['winner_name'] == player) | (self.matches['loser_name'] == player)) &
            (self.matches['tourney_date'] >= start_date) &
            (self.matches['tourney_date'] < current_date)
        ].copy()
        
        if len(pm) == 0:
            return self._empty_stats()
            
        # 1. Clutch: Tiebreak Performance
        tb_played = 0
        tb_won = 0
        
        # 2. Deciders: 3rd set (in bo3) or 5th set (in bo5)
        deciders_played = 0
        deciders_won = 0
        
        # 3. Dominance: Sets won 6-0, 6-1, or 6-2
        total_sets = 0
        dominance_sets = 0
        
        # 4. Recent Form (Last 5 matches)
        # Sort by date
        pm = pm.sort_values('tourney_date', ascending=False)
        last_5 = pm.head(5)
        l5_matches_won = 0
        l5_sets_won = 0
        l5_sets_played = 0
        
        for idx, row in pm.iterrows():
            is_winner = row['winner_name'] == player
            score = str(row['score'])
            
            # Simple score parsing (robustness needed for real dirty data)
            # e.g. "6-4 3-6 7-6(5)"
            sets = score.split(' ')
            
            # Check Decider
            is_decider = False
            best_of = row.get('best_of', 3)
            if len(sets) == best_of: # Took full distance
                is_decider = True
                deciders_played += 1
                if is_winner:
                    deciders_won += 1
            
            for s in sets:
                if 'RET' in s or 'W/O' in s: continue
                
                # Tiebreak check
                if '7-6' in s or '6-7' in s:
                    tb_played += 1
                    # Who won the TB?
                    # If 7-6 and winner, player won TB
                    # If 6-7 and winner, player lost TB
                    # Need to parse carefully
                    
                    # Logic: 
                    # If player is winner: 7-6 is win, 6-7 is loss
                    # If player is loser: 7-6 is loss, 6-7 is win
                    
                    if '7-6' in s:
                        if is_winner: tb_won += 1
                    elif '6-7' in s:
                        if not is_winner: tb_won += 1
                        
                # Dominance check
                # Look for "6-0", "6-1", "6-2"
                # If player is winner: these are dominance sets
                # If player is loser: check "0-6", "1-6", "2-6"
                total_sets += 1

                # Parse the set score (e.g., "6-4" or "6-7(5)")
                clean_set = s.split('(')[0]  # Remove tiebreak score
                parts = clean_set.split('-')
                if len(parts) == 2:
                    try:
                        games_1, games_2 = int(parts[0]), int(parts[1])
                        # Dominance sets: 6-0, 6-1, 6-2 (or reverse if loser)
                        if is_winner:
                            # Winner's dominant set: won 6-0, 6-1, or 6-2
                            if games_1 == 6 and games_2 <= 2:
                                dominance_sets += 1
                        else:
                            # Loser's dominant set: won 0-6, 1-6, 2-6
                            if games_2 == 6 and games_1 <= 2:
                                dominance_sets += 1
                    except ValueError:
                        pass
                
        # Recent Form L5
        for idx, row in last_5.iterrows():
            is_winner = row['winner_name'] == player
            if is_winner: l5_matches_won += 1
            
        clutch_rating = (tb_won / tb_played) if tb_played > 0 else 0.5
        decider_rating = (deciders_won / deciders_played) if deciders_played > 0 else 0.5
        dominance_rating = (dominance_sets / total_sets) if total_sets > 0 else 0.0

        return {
            'matches_played': len(pm),
            'clutch_rating': clutch_rating, # Win % in Tiebreaks
            'decider_rating': decider_rating, # Win % in Final Sets
            'dominance_rating': dominance_rating, # % of sets that were dominant wins
            'recent_form_rating': l5_matches_won / 5, # L5 Win %
            'tb_record': f"{tb_won}-{tb_played - tb_won}",
            'decider_record': f"{deciders_won}-{deciders_played - deciders_won}",
            'dominance_sets': dominance_sets,
            'total_sets': total_sets
        }

    def _empty_stats(self):
        return {
            'matches_played': 0,
            'clutch_rating': 0.5,
            'decider_rating': 0.5,
            'dominance_rating': 0.0,
            'recent_form_rating': 0.5,
            'tb_record': "0-0",
            'decider_record': "0-0",
            'dominance_sets': 0,
            'total_sets': 0
        }
