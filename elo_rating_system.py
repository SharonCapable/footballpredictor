"""
ELO Rating System for Football
Adds dynamic team strength ratings that update after each match
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class FootballEloSystem:
    """
    ELO rating system for football teams
    
    How it works:
    - Every team starts at 1500 rating
    - Winner gains points, loser loses points
    - Bigger upsets = bigger point swings
    - Home advantage included
    """
    
    def __init__(self, k_factor=20, home_advantage=100):
        """
        Args:
            k_factor: How much ratings change per match (default: 20)
            home_advantage: Rating boost for home team (default: 100)
        """
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings = defaultdict(lambda: 1500)  # All teams start at 1500
        
    def expected_score(self, rating_a, rating_b):
        """
        Calculate expected score (0-1) for team A vs team B
        
        Formula: E = 1 / (1 + 10^((Rb - Ra) / 400))
        """
        return 1 / (1 + 10**((rating_b - rating_a) / 400))
    
    def update_ratings(self, home_team, away_team, home_score, away_score):
        """
        Update ELO ratings based on match result
        
        Returns:
            dict with old ratings, new ratings, and rating changes
        """
        # Get current ratings
        home_rating = self.ratings[home_team]
        away_rating = self.ratings[away_team]
        
        # Apply home advantage
        home_rating_adj = home_rating + self.home_advantage
        
        # Expected scores
        home_expected = self.expected_score(home_rating_adj, away_rating)
        away_expected = 1 - home_expected
        
        # Actual scores (1 for win, 0.5 for draw, 0 for loss)
        if home_score > away_score:
            home_actual = 1.0
            away_actual = 0.0
        elif home_score < away_score:
            home_actual = 0.0
            away_actual = 1.0
        else:
            home_actual = 0.5
            away_actual = 0.5
        
        # Goal difference multiplier (bigger wins = bigger rating changes)
        goal_diff = abs(home_score - away_score)
        if goal_diff <= 1:
            multiplier = 1.0
        elif goal_diff == 2:
            multiplier = 1.5
        else:
            multiplier = (11 + goal_diff) / 8
        
        # Calculate rating changes
        home_change = self.k_factor * multiplier * (home_actual - home_expected)
        away_change = self.k_factor * multiplier * (away_actual - away_expected)
        
        # Update ratings
        new_home_rating = home_rating + home_change
        new_away_rating = away_rating + away_change
        
        self.ratings[home_team] = new_home_rating
        self.ratings[away_team] = new_away_rating
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_old_rating': home_rating,
            'away_old_rating': away_rating,
            'home_new_rating': new_home_rating,
            'away_new_rating': new_away_rating,
            'home_change': home_change,
            'away_change': away_change
        }
    
    def get_pre_match_ratings(self, home_team, away_team):
        """Get ratings BEFORE a match (for prediction)"""
        return {
            'home_elo': self.ratings[home_team],
            'away_elo': self.ratings[away_team],
            'elo_diff': self.ratings[home_team] - self.ratings[away_team],
            'home_win_prob': self.expected_score(
                self.ratings[home_team] + self.home_advantage,
                self.ratings[away_team]
            )
        }


class EloFeatureEngineer:
    """Add ELO ratings to existing match data"""
    
    def __init__(self, raw_dir='data/raw', processed_dir='data/processed'):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        
    def add_elo_features(self, league):
        """
        Add ELO ratings to a league's data
        
        Process:
        1. Load matches chronologically
        2. For each match:
           - Record pre-match ELO ratings
           - Update ELO after match
        3. Save enhanced dataset
        """
        print(f"\n{'='*60}")
        print(f"⚡ ADDING ELO RATINGS: {league.upper()}")
        print(f"{'='*60}")
        
        # Load raw data
        file_path = self.raw_dir / f"{league}_matches.csv"
        if not file_path.exists():
            print(f"⚠️  {league} not found")
            return None
        
        df = pd.read_csv(file_path)
        
        # Standardize columns
        if 'FTHG' in df.columns:
            df = df.rename(columns={
                'Date': 'date',
                'HomeTeam': 'home_team',
                'AwayTeam': 'away_team',
                'FTHG': 'home_score',
                'FTAG': 'away_score'
            })
        
        # Sort by date
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"✓ Loaded {len(df)} matches")
        
        # Initialize ELO system
        elo = FootballEloSystem(k_factor=20, home_advantage=100)
        
        # Add ELO columns
        df['home_elo_before'] = 0.0
        df['away_elo_before'] = 0.0
        df['elo_diff'] = 0.0
        df['home_win_prob'] = 0.0
        
        # Process each match
        print("📊 Calculating ELO ratings...")
        
        for idx, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            home_score = row['home_score']
            away_score = row['away_score']
            
            # Get pre-match ratings (for this match's features)
            pre_match = elo.get_pre_match_ratings(home_team, away_team)
            
            df.at[idx, 'home_elo_before'] = pre_match['home_elo']
            df.at[idx, 'away_elo_before'] = pre_match['away_elo']
            df.at[idx, 'elo_diff'] = pre_match['elo_diff']
            df.at[idx, 'home_win_prob'] = pre_match['home_win_prob']
            
            # Update ratings after this match
            elo.update_ratings(home_team, away_team, home_score, away_score)
        
        # Show ELO distribution
        print(f"\n📈 ELO Rating Distribution:")
        print(f"  Min:    {df['home_elo_before'].min():.0f}")
        print(f"  Max:    {df['home_elo_before'].max():.0f}")
        print(f"  Mean:   {df['home_elo_before'].mean():.0f}")
        print(f"  Median: {df['home_elo_before'].median():.0f}")
        
        # Show top teams (final ratings)
        final_ratings = pd.DataFrame([
            {'team': team, 'elo': rating}
            for team, rating in elo.ratings.items()
        ]).sort_values('elo', ascending=False)
        
        print(f"\n🏆 Top 10 Teams (Final ELO):")
        print(final_ratings.head(10).to_string(index=False))
        
        # Save enhanced data
        output_file = self.processed_dir / f"{league}_with_elo.csv"
        df.to_csv(output_file, index=False)
        
        print(f"\n✓ Saved to: {output_file}")
        return df
    
    def add_elo_to_all_leagues(self, leagues):
        """Add ELO to multiple leagues"""
        print("\n" + "="*60)
        print("🚀 ADDING ELO RATINGS TO ALL LEAGUES")
        print("="*60)
        
        results = {}
        
        for league in leagues:
            df = self.add_elo_features(league)
            if df is not None:
                results[league] = df
        
        # Combined dataset
        if results:
            print("\n" + "="*60)
            print("📊 SUMMARY")
            print("="*60)
            
            for league, df in results.items():
                print(f"{league.title():12} {len(df):5} matches")
            
            # Create combined file
            combined = pd.concat(results.values(), ignore_index=True)
            output_file = self.processed_dir / 'all_leagues_with_elo.csv'
            combined.to_csv(output_file, index=False)
            
            print(f"\n✓ Combined saved: {output_file}")
            print(f"✓ Total: {len(combined)} matches")
        
        return results


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("⚡ ELO RATING SYSTEM")
    print("="*60)
    print("\nELO ratings capture team strength dynamically!")
    print("\nHow it works:")
    print("  • All teams start at 1500")
    print("  • Winners gain points, losers lose points")
    print("  • Bigger upsets = bigger point swings")
    print("  • Updates after every match")
    
    print("\nExample:")
    print("  Man City (1800) beats Brighton (1550)")
    print("  → Expected, small rating change")
    print("\n  Brighton (1550) beats Man City (1800)")
    print("  → Upset! Large rating change")
    
    input("\nPress Enter to calculate ELO ratings...")
    
    engineer = EloFeatureEngineer()
    
    leagues = ['england', 'spain', 'germany', 'italy', 'france']
    results = engineer.add_elo_to_all_leagues(leagues)
    
    print("\n" + "="*60)
    print("✅ ELO RATINGS ADDED!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Retrain models with ELO features")
    print("  2. Compare accuracy (should improve 2-5%)")
    print("  3. ELO is now your STRONGEST feature!")


if __name__ == "__main__":
    main()