"""
Data Cleaning and Feature Engineering Module
Prepares raw match data for model training
"""

import pandas as pd
import numpy as np
from datetime import datetime

class FootballDataCleaner:
    def __init__(self):
        self.df = None
        
    def load_data(self, filepath):
        """Load raw match data"""
        self.df = pd.read_csv(filepath)
        self.df['date'] = pd.to_datetime(self.df['date'])
        return self.df
    
    def clean_basic(self):
        """Basic data cleaning"""
        print("Running basic cleaning...")
        
        # Remove duplicates
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=['match_id'])
        print(f"  Removed {initial_count - len(self.df)} duplicates")
        
        # Remove rows with missing goals
        self.df = self.df.dropna(subset=['home_goals', 'away_goals'])
        
        # Ensure goals are integers
        self.df['home_goals'] = self.df['home_goals'].astype(int)
        self.df['away_goals'] = self.df['away_goals'].astype(int)
        
        # Sort by date
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        print("✓ Basic cleaning complete")
        return self.df
    
    def create_target_variable(self):
        """
        Create outcome labels:
        H = Home Win
        D = Draw
        A = Away Win
        """
        def get_outcome(row):
            if row['home_goals'] > row['away_goals']:
                return 'H'
            elif row['home_goals'] < row['away_goals']:
                return 'A'
            else:
                return 'D'
        
        self.df['outcome'] = self.df.apply(get_outcome, axis=1)
        
        # Numeric encoding for modeling
        outcome_map = {'H': 1, 'D': 0, 'A': 2}
        self.df['outcome_numeric'] = self.df['outcome'].map(outcome_map)
        
        print("✓ Created target variable")
        return self.df
    
    def engineer_team_form_features(self, window=5):
        """
        Calculate rolling team statistics based on previous matches
        
        Args:
            window (int): Number of previous matches to consider
        """
        print(f"Engineering team form features (last {window} matches)...")
        
        # Sort by date to ensure chronological order
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        # Initialize feature columns
        features = [
            'home_team_form_points', 'away_team_form_points',
            'home_team_goals_scored_avg', 'away_team_goals_scored_avg',
            'home_team_goals_conceded_avg', 'away_team_goals_conceded_avg',
            'home_team_win_rate', 'away_team_win_rate'
        ]
        
        for feat in features:
            self.df[feat] = np.nan
        
        # Calculate features for each match
        for idx, row in self.df.iterrows():
            match_date = row['date']
            home_team = row['home_team_id']
            away_team = row['away_team_id']
            
            # Get previous matches for home team (before current match)
            home_prev = self.df[
                (self.df['date'] < match_date) & 
                ((self.df['home_team_id'] == home_team) | 
                 (self.df['away_team_id'] == home_team))
            ].tail(window)
            
            # Get previous matches for away team
            away_prev = self.df[
                (self.df['date'] < match_date) & 
                ((self.df['home_team_id'] == away_team) | 
                 (self.df['away_team_id'] == away_team))
            ].tail(window)
            
            # Calculate home team features
            if len(home_prev) > 0:
                home_points = self._calculate_points(home_prev, home_team)
                home_gf = self._calculate_goals_for(home_prev, home_team)
                home_ga = self._calculate_goals_against(home_prev, home_team)
                home_wins = self._calculate_win_rate(home_prev, home_team)
                
                self.df.at[idx, 'home_team_form_points'] = home_points
                self.df.at[idx, 'home_team_goals_scored_avg'] = home_gf
                self.df.at[idx, 'home_team_goals_conceded_avg'] = home_ga
                self.df.at[idx, 'home_team_win_rate'] = home_wins
            
            # Calculate away team features
            if len(away_prev) > 0:
                away_points = self._calculate_points(away_prev, away_team)
                away_gf = self._calculate_goals_for(away_prev, away_team)
                away_ga = self._calculate_goals_against(away_prev, away_team)
                away_wins = self._calculate_win_rate(away_prev, away_team)
                
                self.df.at[idx, 'away_team_form_points'] = away_points
                self.df.at[idx, 'away_team_goals_scored_avg'] = away_gf
                self.df.at[idx, 'away_team_goals_conceded_avg'] = away_ga
                self.df.at[idx, 'away_team_win_rate'] = away_wins
        
        print("✓ Team form features created")
        return self.df
    
    def _calculate_points(self, matches, team_id):
        """Calculate total points from previous matches"""
        points = 0
        for _, match in matches.iterrows():
            if match['home_team_id'] == team_id:
                if match['home_goals'] > match['away_goals']:
                    points += 3
                elif match['home_goals'] == match['away_goals']:
                    points += 1
            else:  # Away team
                if match['away_goals'] > match['home_goals']:
                    points += 3
                elif match['away_goals'] == match['home_goals']:
                    points += 1
        return points
    
    def _calculate_goals_for(self, matches, team_id):
        """Calculate average goals scored"""
        goals = []
        for _, match in matches.iterrows():
            if match['home_team_id'] == team_id:
                goals.append(match['home_goals'])
            else:
                goals.append(match['away_goals'])
        return np.mean(goals) if goals else 0
    
    def _calculate_goals_against(self, matches, team_id):
        """Calculate average goals conceded"""
        goals = []
        for _, match in matches.iterrows():
            if match['home_team_id'] == team_id:
                goals.append(match['away_goals'])
            else:
                goals.append(match['home_goals'])
        return np.mean(goals) if goals else 0
    
    def _calculate_win_rate(self, matches, team_id):
        """Calculate win rate from previous matches"""
        wins = 0
        for _, match in matches.iterrows():
            if match['home_team_id'] == team_id:
                if match['home_goals'] > match['away_goals']:
                    wins += 1
            else:
                if match['away_goals'] > match['home_goals']:
                    wins += 1
        return wins / len(matches) if len(matches) > 0 else 0
    
    def create_head_to_head_features(self, h2h_window=5):
        """Calculate head-to-head statistics between teams"""
        print(f"Creating head-to-head features (last {h2h_window} meetings)...")
        
        self.df['h2h_home_wins'] = np.nan
        self.df['h2h_draws'] = np.nan
        self.df['h2h_away_wins'] = np.nan
        
        for idx, row in self.df.iterrows():
            match_date = row['date']
            home_team = row['home_team_id']
            away_team = row['away_team_id']
            
            # Get previous meetings
            h2h = self.df[
                (self.df['date'] < match_date) &
                (((self.df['home_team_id'] == home_team) & (self.df['away_team_id'] == away_team)) |
                 ((self.df['home_team_id'] == away_team) & (self.df['away_team_id'] == home_team)))
            ].tail(h2h_window)
            
            if len(h2h) > 0:
                home_wins = len(h2h[
                    ((h2h['home_team_id'] == home_team) & (h2h['home_goals'] > h2h['away_goals'])) |
                    ((h2h['away_team_id'] == home_team) & (h2h['away_goals'] > h2h['home_goals']))
                ])
                
                draws = len(h2h[h2h['home_goals'] == h2h['away_goals']])
                
                away_wins = len(h2h) - home_wins - draws
                
                self.df.at[idx, 'h2h_home_wins'] = home_wins
                self.df.at[idx, 'h2h_draws'] = draws
                self.df.at[idx, 'h2h_away_wins'] = away_wins
        
        print("✓ Head-to-head features created")
        return self.df
    
    def remove_early_season_matches(self, min_matches=5):
        """Remove matches where teams don't have enough history"""
        print(f"Removing matches with insufficient history (< {min_matches} previous matches)...")
        
        initial_count = len(self.df)
        
        # Drop rows where form features are NaN (insufficient history)
        self.df = self.df.dropna(subset=['home_team_form_points', 'away_team_form_points'])
        
        print(f"  Removed {initial_count - len(self.df)} matches")
        print("✓ Dataset ready for modeling")
        return self.df
    
    def get_clean_dataset(self):
        """Return the cleaned dataset"""
        return self.df
    
    def save_cleaned_data(self, filepath='data/processed/cleaned_matches.csv'):
        """Save cleaned data"""
        self.df.to_csv(filepath, index=False)
        print(f"✓ Saved cleaned data to {filepath}")


# Example usage
if __name__ == "__main__":
    cleaner = FootballDataCleaner()
    
    # Load raw data
    df = cleaner.load_data('data/raw/premier_league_matches.csv')
    print(f"Loaded {len(df)} matches")
    
    # Run cleaning pipeline
    df = cleaner.clean_basic()
    df = cleaner.create_target_variable()
    df = cleaner.engineer_team_form_features(window=5)
    df = cleaner.create_head_to_head_features(h2h_window=5)
    df = cleaner.remove_early_season_matches(min_matches=5)
    
    # Check outcome distribution
    print("\nOutcome distribution:")
    print(df['outcome'].value_counts())
    
    # Save
    cleaner.save_cleaned_data()
    
    print(f"\nFinal dataset: {len(df)} matches with {len(df.columns)} features")