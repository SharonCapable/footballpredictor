"""
Multi-League Data Preprocessing Pipeline
Processes data from Football-Data.co.uk for multiple leagues with advanced features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MultiLeaguePreprocessor:
    """Advanced preprocessing for multiple football leagues"""
    
    LEAGUES = ['england', 'spain', 'germany', 'italy', 'france', 'portugal']
    
    def __init__(self, raw_dir='data/raw', processed_dir='data/processed'):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def load_raw_data(self, league):
        """Load raw data for a specific league"""
        file_path = self.raw_dir / f"{league}_matches.csv"
        
        if not file_path.exists():
            print(f"⚠️  {league} data not found at {file_path}")
            return None
        
        df = pd.read_csv(file_path)
        print(f"✓ Loaded {league}: {len(df)} matches")
        return df
    
    def standardize_columns(self, df, league):
        """Standardize column names across different data sources"""
        
        # Football-Data.co.uk format
        if 'FTHG' in df.columns:
            column_mapping = {
                'Date': 'date',
                'HomeTeam': 'home_team',
                'AwayTeam': 'away_team',
                'FTHG': 'home_score',
                'FTAG': 'away_score',
                'FTR': 'result',
                'HTHG': 'ht_home_score',
                'HTAG': 'ht_away_score',
                'HS': 'home_shots',
                'AS': 'away_shots',
                'HST': 'home_shots_on_target',
                'AST': 'away_shots_on_target',
                'HF': 'home_fouls',
                'AF': 'away_fouls',
                'HC': 'home_corners',
                'AC': 'away_corners',
                'HY': 'home_yellow_cards',
                'AY': 'away_yellow_cards',
                'HR': 'home_red_cards',
                'AR': 'away_red_cards'
            }
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Portugal format (if different)
        if league == 'portugal' and 'home_team' not in df.columns:
            # Add any Portugal-specific mappings here
            pass
        
        # Ensure league column exists
        if 'League' not in df.columns:
            df['League'] = league
        
        return df
    
    def clean_data(self, df):
        """Clean and validate data"""
        print("  🧹 Cleaning data...")
        
        initial_count = len(df)
        
        # Remove rows with missing critical columns
        required_cols = ['home_team', 'away_team', 'home_score', 'away_score']
        df = df.dropna(subset=required_cols)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['date', 'home_team', 'away_team'], keep='first')
        
        # Convert scores to numeric
        df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
        df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')
        
        # Remove invalid scores (negative or NaN)
        df = df[(df['home_score'] >= 0) & (df['away_score'] >= 0)]
        
        removed = initial_count - len(df)
        if removed > 0:
            print(f"  ⚠️  Removed {removed} invalid/duplicate rows")
        
        return df
    
    def create_outcome(self, df):
        """Create outcome variable: 0=Away Win, 1=Draw, 2=Home Win"""
        print("  🎯 Creating outcome variable...")
        
        def determine_outcome(row):
            if row['home_score'] > row['away_score']:
                return 2  # Home win
            elif row['home_score'] < row['away_score']:
                return 0  # Away win
            else:
                return 1  # Draw
        
        df['outcome'] = df.apply(determine_outcome, axis=1)
        
        # Print distribution
        outcome_counts = df['outcome'].value_counts().sort_index()
        print(f"     Away Win: {outcome_counts.get(0, 0)} ({outcome_counts.get(0, 0)/len(df)*100:.1f}%)")
        print(f"     Draw:     {outcome_counts.get(1, 0)} ({outcome_counts.get(1, 0)/len(df)*100:.1f}%)")
        print(f"     Home Win: {outcome_counts.get(2, 0)} ({outcome_counts.get(2, 0)/len(df)*100:.1f}%)")
        
        return df
    
    def engineer_basic_features(self, df):
        """Create basic engineered features"""
        print("  ⚙️  Engineering basic features...")
        
        # Goal difference
        df['goal_diff'] = df['home_score'] - df['away_score']
        
        # Total goals
        df['total_goals'] = df['home_score'] + df['away_score']
        
        # High scoring game flag
        df['high_scoring'] = (df['total_goals'] >= 3).astype(int)
        
        # Home advantage (will be calculated from historical data)
        df['home_advantage'] = 1  # Placeholder
        
        # Shot efficiency (if available)
        if 'home_shots' in df.columns and 'home_shots_on_target' in df.columns:
            df['home_shot_accuracy'] = df['home_shots_on_target'] / (df['home_shots'] + 1)
            df['away_shot_accuracy'] = df['away_shots_on_target'] / (df['away_shots'] + 1)
        
        # Discipline (cards per game)
        if 'home_yellow_cards' in df.columns:
            df['home_discipline'] = df['home_yellow_cards'] + (df.get('home_red_cards', 0) * 3)
            df['away_discipline'] = df['away_yellow_cards'] + (df.get('away_red_cards', 0) * 3)
        
        return df
    
    def engineer_rolling_features(self, df, window=5):
        """Create rolling average features for team form"""
        print(f"  📊 Engineering rolling features (last {window} games)...")
        
        # Sort by date
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values('date').reset_index(drop=True)
        
        # Initialize form columns
        form_features = [
            'home_goals_scored_avg', 'home_goals_conceded_avg',
            'away_goals_scored_avg', 'away_goals_conceded_avg',
            'home_win_rate', 'away_win_rate',
            'home_form_points', 'away_form_points'
        ]
        
        for feature in form_features:
            df[feature] = 0.0
        
        # Calculate rolling statistics for each team
        teams = pd.concat([df['home_team'], df['away_team']]).unique()
        
        for team in teams:
            # Home games
            home_mask = df['home_team'] == team
            home_indices = df[home_mask].index
            
            for idx in home_indices:
                # Get previous games (both home and away)
                prev_home = df[(df['home_team'] == team) & (df.index < idx)].tail(window)
                prev_away = df[(df['away_team'] == team) & (df.index < idx)].tail(window)
                
                if len(prev_home) + len(prev_away) > 0:
                    # Goals scored
                    home_goals = prev_home['home_score'].tolist()
                    away_goals = prev_away['away_score'].tolist()
                    all_goals_scored = home_goals + away_goals
                    df.at[idx, 'home_goals_scored_avg'] = np.mean(all_goals_scored) if all_goals_scored else 0
                    
                    # Goals conceded
                    home_conceded = prev_home['away_score'].tolist()
                    away_conceded = prev_away['home_score'].tolist()
                    all_goals_conceded = home_conceded + away_conceded
                    df.at[idx, 'home_goals_conceded_avg'] = np.mean(all_goals_conceded) if all_goals_conceded else 0
                    
                    # Win rate
                    home_wins = (prev_home['outcome'] == 2).sum()
                    away_wins = (prev_away['outcome'] == 0).sum()
                    total_games = len(prev_home) + len(prev_away)
                    df.at[idx, 'home_win_rate'] = (home_wins + away_wins) / total_games if total_games > 0 else 0
                    
                    # Form points (3 for win, 1 for draw, 0 for loss)
                    home_points = (prev_home['outcome'] == 2) * 3 + (prev_home['outcome'] == 1) * 1
                    away_points = (prev_away['outcome'] == 0) * 3 + (prev_away['outcome'] == 1) * 1
                    df.at[idx, 'home_form_points'] = (home_points.sum() + away_points.sum()) / total_games if total_games > 0 else 0
            
            # Away games
            away_mask = df['away_team'] == team
            away_indices = df[away_mask].index
            
            for idx in away_indices:
                prev_home = df[(df['home_team'] == team) & (df.index < idx)].tail(window)
                prev_away = df[(df['away_team'] == team) & (df.index < idx)].tail(window)
                
                if len(prev_home) + len(prev_away) > 0:
                    home_goals = prev_home['home_score'].tolist()
                    away_goals = prev_away['away_score'].tolist()
                    all_goals_scored = home_goals + away_goals
                    df.at[idx, 'away_goals_scored_avg'] = np.mean(all_goals_scored) if all_goals_scored else 0
                    
                    home_conceded = prev_home['away_score'].tolist()
                    away_conceded = prev_away['home_score'].tolist()
                    all_goals_conceded = home_conceded + away_conceded
                    df.at[idx, 'away_goals_conceded_avg'] = np.mean(all_goals_conceded) if all_goals_conceded else 0
                    
                    home_wins = (prev_home['outcome'] == 2).sum()
                    away_wins = (prev_away['outcome'] == 0).sum()
                    total_games = len(prev_home) + len(prev_away)
                    df.at[idx, 'away_win_rate'] = (home_wins + away_wins) / total_games if total_games > 0 else 0
                    
                    home_points = (prev_home['outcome'] == 2) * 3 + (prev_home['outcome'] == 1) * 1
                    away_points = (prev_away['outcome'] == 0) * 3 + (prev_away['outcome'] == 1) * 1
                    df.at[idx, 'away_form_points'] = (home_points.sum() + away_points.sum()) / total_games if total_games > 0 else 0
        
        # Remove first N games where rolling stats aren't meaningful
        initial_len = len(df)
        df = df[df['home_form_points'] > 0].reset_index(drop=True)
        removed = initial_len - len(df)
        print(f"  ⚠️  Removed {removed} matches without sufficient history")
        
        return df
    
    def select_features(self, df):
        """Select final feature set for modeling"""
        print("  🎯 Selecting model features...")
        
        # Core features
        feature_cols = [
            'home_goals_scored_avg', 'home_goals_conceded_avg',
            'away_goals_scored_avg', 'away_goals_conceded_avg',
            'home_win_rate', 'away_win_rate',
            'home_form_points', 'away_form_points',
            'home_advantage'
        ]
        
        # Optional features (include if available)
        optional_features = [
            'home_shot_accuracy', 'away_shot_accuracy',
            'home_discipline', 'away_discipline'
        ]
        
        for feat in optional_features:
            if feat in df.columns:
                feature_cols.append(feat)
        
        # Ensure all features exist
        missing_features = [f for f in feature_cols if f not in df.columns]
        if missing_features:
            print(f"  ⚠️  Missing features: {missing_features}")
            feature_cols = [f for f in feature_cols if f in df.columns]
        
        # Create final dataset
        final_cols = feature_cols + ['outcome', 'League', 'date', 'home_team', 'away_team']
        df_final = df[final_cols].copy()
        
        print(f"  ✓ Final feature count: {len(feature_cols)}")
        return df_final
    
    def process_league(self, league):
        """Complete preprocessing pipeline for one league"""
        print(f"\n{'='*60}")
        print(f"🏆 PROCESSING: {league.upper()}")
        print(f"{'='*60}")
        
        # Load data
        df = self.load_raw_data(league)
        if df is None:
            return None
        
        # Standardize columns
        df = self.standardize_columns(df, league)
        
        # Clean data
        df = self.clean_data(df)
        
        # Create outcome
        df = self.create_outcome(df)
        
        # Engineer features
        df = self.engineer_basic_features(df)
        df = self.engineer_rolling_features(df, window=5)
        
        # Select features
        df = self.select_features(df)
        
        # Save processed data
        output_file = self.processed_dir / f"{league}_processed.csv"
        df.to_csv(output_file, index=False)
        
        print(f"\n✓ Processed {len(df)} matches")
        print(f"✓ Saved to: {output_file}")
        
        return df
    
    def process_all_leagues(self):
        """Process all available leagues"""
        print("\n" + "="*60)
        print("🌍 PROCESSING ALL LEAGUES")
        print("="*60)
        
        results = {}
        summary = []
        
        for league in self.LEAGUES:
            df = self.process_league(league)
            if df is not None:
                results[league] = df
                summary.append({
                    'League': league.title(),
                    'Matches': len(df),
                    'Features': len([c for c in df.columns if c not in ['outcome', 'League', 'date', 'home_team', 'away_team']])
                })
        
        # Print summary
        print("\n" + "="*60)
        print("📊 PREPROCESSING SUMMARY")
        print("="*60)
        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(index=False))
        print(f"\n✓ Total processed matches: {sum(s['Matches'] for s in summary)}")
        
        # Create combined dataset
        if results:
            print("\n🔄 Creating combined dataset...")
            combined_df = pd.concat(results.values(), ignore_index=True)
            combined_file = self.processed_dir / 'all_leagues_processed.csv'
            combined_df.to_csv(combined_file, index=False)
            print(f"✓ Combined dataset saved: {combined_file}")
            print(f"✓ Total matches: {len(combined_df)}")
        
        return results


def main():
    """Main execution"""
    print("\n🚀 MULTI-LEAGUE PREPROCESSING PIPELINE")
    print("="*60)
    
    preprocessor = MultiLeaguePreprocessor()
    
    # Process all leagues
    results = preprocessor.process_all_leagues()
    
    print("\n✅ PREPROCESSING COMPLETE!")
    print("\nNext steps:")
    print("  1. Run training: python complete_system.py")
    print("  2. Compare accuracy with previous models")
    print("  3. Analyze feature importance")


if __name__ == "__main__":
    main()