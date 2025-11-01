"""
Improved Feature Engineering - Goal-Focused Features
Creates features that actually predict scoring patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
sys.path.append('..')
from dotenv import load_dotenv
from database_manager import FootballDatabaseManager

load_dotenv()

def create_improved_features():
    """
    Create goal-focused features that better predict O/U and BTTS
    """
    
    # Connect to database
    db = FootballDatabaseManager(
        host=os.getenv('DB_HOST'),
        port=int(os.getenv('DB_PORT')),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    
    # Load all matches
    with db.get_cursor() as cursor:
        cursor.execute("""
            SELECT 
                m.match_id,
                m.fixture_date,
                m.league_id,
                m.season,
                m.home_team_id,
                m.away_team_id,
                ht.team_name as home_team,
                at.team_name as away_team,
                m.home_goals,
                m.away_goals
            FROM matches m
            JOIN teams ht ON m.home_team_id = ht.team_id
            JOIN teams at ON m.away_team_id = at.team_id
            ORDER BY m.fixture_date
        """)
        
        matches_df = pd.DataFrame(cursor.fetchall())
    
    db.close()
    
    matches_df['fixture_date'] = pd.to_datetime(matches_df['fixture_date'])
    matches_df['total_goals'] = matches_df['home_goals'] + matches_df['away_goals']
    
    print(f"Processing {len(matches_df)} matches...")
    
    # Initialize feature lists
    features_list = []
    
    for idx, match in matches_df.iterrows():
        if idx % 100 == 0:
            print(f"  Processing {idx}/{len(matches_df)}...")
        
        match_date = match['fixture_date']
        home_team_id = match['home_team_id']
        away_team_id = match['away_team_id']
        
        # Get previous matches (use last 10 instead of 5 for more data)
        home_prev = matches_df[
            ((matches_df['home_team_id'] == home_team_id) | 
             (matches_df['away_team_id'] == home_team_id)) &
            (matches_df['fixture_date'] < match_date)
        ].tail(10)
        
        away_prev = matches_df[
            ((matches_df['home_team_id'] == away_team_id) | 
             (matches_df['away_team_id'] == away_team_id)) &
            (matches_df['fixture_date'] < match_date)
        ].tail(10)
        
        # Skip if insufficient data (require at least 3 matches)
        if len(home_prev) < 3 or len(away_prev) < 3:
            continue
        
        # ============ HOME TEAM FEATURES ============
        
        # Get goals scored/conceded in last N matches
        home_goals_for = []
        home_goals_against = []
        home_total_goals = []
        
        for _, prev_match in home_prev.iterrows():
            if prev_match['home_team_id'] == home_team_id:
                home_goals_for.append(prev_match['home_goals'])
                home_goals_against.append(prev_match['away_goals'])
            else:
                home_goals_for.append(prev_match['away_goals'])
                home_goals_against.append(prev_match['home_goals'])
            
            home_total_goals.append(prev_match['total_goals'])
        
        # Home team specific (only home matches)
        home_home_only = matches_df[
            (matches_df['home_team_id'] == home_team_id) &
            (matches_df['fixture_date'] < match_date)
        ].tail(5)
        
        home_home_goals_for = home_home_only['home_goals'].tolist()
        home_home_goals_against = home_home_only['away_goals'].tolist()
        
        # ============ AWAY TEAM FEATURES ============
        
        away_goals_for = []
        away_goals_against = []
        away_total_goals = []
        
        for _, prev_match in away_prev.iterrows():
            if prev_match['away_team_id'] == away_team_id:
                away_goals_for.append(prev_match['away_goals'])
                away_goals_against.append(prev_match['home_goals'])
            else:
                away_goals_for.append(prev_match['home_goals'])
                away_goals_against.append(prev_match['away_goals'])
            
            away_total_goals.append(prev_match['total_goals'])
        
        # Away team specific (only away matches)
        away_away_only = matches_df[
            (matches_df['away_team_id'] == away_team_id) &
            (matches_df['fixture_date'] < match_date)
        ].tail(5)
        
        away_away_goals_for = away_away_only['away_goals'].tolist()
        away_away_goals_against = away_away_only['home_goals'].tolist()
        
        # ============ CREATE FEATURES ============
        
        features = {
            'match_id': match['match_id'],
            'fixture_date': match_date,
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            
            # Target variables
            'home_goals': match['home_goals'],
            'away_goals': match['away_goals'],
            'total_goals': match['total_goals'],
            'over_2_5': 1 if match['total_goals'] > 2.5 else 0,
            'btts': 1 if (match['home_goals'] > 0 and match['away_goals'] > 0) else 0,
            
            # Home team - GOAL SCORING
            'home_avg_goals_scored_l10': np.mean(home_goals_for),
            'home_avg_goals_scored_l5': np.mean(home_goals_for[-5:]),
            'home_avg_goals_scored_l3': np.mean(home_goals_for[-3:]),
            
            # Home team - GOAL CONCEDING
            'home_avg_goals_conceded_l10': np.mean(home_goals_against),
            'home_avg_goals_conceded_l5': np.mean(home_goals_against[-5:]),
            'home_avg_goals_conceded_l3': np.mean(home_goals_against[-3:]),
            
            # Home team - TOTAL GOALS IN MATCHES
            'home_avg_total_goals_l10': np.mean(home_total_goals),
            'home_avg_total_goals_l5': np.mean(home_total_goals[-5:]),
            
            # Home team - HOME VENUE SPECIFIC
            'home_home_avg_scored': np.mean(home_home_goals_for) if home_home_goals_for else 0,
            'home_home_avg_conceded': np.mean(home_home_goals_against) if home_home_goals_against else 0,
            
            # Home team - SCORING CONSISTENCY
            'home_scored_in_last_5': sum([1 for g in home_goals_for[-5:] if g > 0]),
            'home_conceded_in_last_5': sum([1 for g in home_goals_against[-5:] if g > 0]),
            
            # Home team - HIGH SCORING TENDENCY
            'home_high_scoring_rate': sum([1 for t in home_total_goals if t >= 3]) / len(home_total_goals),
            
            # Away team - GOAL SCORING
            'away_avg_goals_scored_l10': np.mean(away_goals_for),
            'away_avg_goals_scored_l5': np.mean(away_goals_for[-5:]),
            'away_avg_goals_scored_l3': np.mean(away_goals_for[-3:]),
            
            # Away team - GOAL CONCEDING
            'away_avg_goals_conceded_l10': np.mean(away_goals_against),
            'away_avg_goals_conceded_l5': np.mean(away_goals_against[-5:]),
            'away_avg_goals_conceded_l3': np.mean(away_goals_against[-3:]),
            
            # Away team - TOTAL GOALS IN MATCHES
            'away_avg_total_goals_l10': np.mean(away_total_goals),
            'away_avg_total_goals_l5': np.mean(away_total_goals[-5:]),
            
            # Away team - AWAY VENUE SPECIFIC
            'away_away_avg_scored': np.mean(away_away_goals_for) if away_away_goals_for else 0,
            'away_away_avg_conceded': np.mean(away_away_goals_against) if away_away_goals_against else 0,
            
            # Away team - SCORING CONSISTENCY
            'away_scored_in_last_5': sum([1 for g in away_goals_for[-5:] if g > 0]),
            'away_conceded_in_last_5': sum([1 for g in away_goals_against[-5:] if g > 0]),
            
            # Away team - HIGH SCORING TENDENCY
            'away_high_scoring_rate': sum([1 for t in away_total_goals if t >= 3]) / len(away_total_goals),
            
            # ============ MATCH-LEVEL FEATURES ============
            
            # Combined attacking strength
            'combined_avg_goals_scored': np.mean(home_goals_for + away_goals_for),
            
            # Combined defensive weakness
            'combined_avg_goals_conceded': np.mean(home_goals_against + away_goals_against),
            
            # Attack vs Defense matchup
            'home_attack_vs_away_defense': np.mean(home_goals_for[-5:]) + np.mean(away_goals_against[-5:]),
            'away_attack_vs_home_defense': np.mean(away_goals_for[-5:]) + np.mean(home_goals_against[-5:]),
            
            # Total expected goals (simple)
            'expected_total_goals': (np.mean(home_goals_for[-5:]) + 
                                    np.mean(away_goals_for[-5:]) + 
                                    np.mean(home_goals_against[-5:]) + 
                                    np.mean(away_goals_against[-5:])) / 2,
            
            # BTTS indicators
            'both_teams_scoring_trend': (sum([1 for g in home_goals_for[-5:] if g > 0]) + 
                                        sum([1 for g in away_goals_for[-5:] if g > 0])) / 10,
        }
        
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    
    print(f"\n✓ Created features for {len(features_df)} matches")
    print(f"✓ Total features: {len(features_df.columns) - 8}")  # Excluding IDs and targets
    
    # Save
    output_path = 'data/processed/improved_features.csv'
    features_df.to_csv(output_path, index=False)
    print(f"✓ Saved to: {output_path}")
    
    return features_df


if __name__ == "__main__":
    print("="*60)
    print("IMPROVED FEATURE ENGINEERING - GOAL-FOCUSED")
    print("="*60)
    
    df = create_improved_features()
    
    print("\n" + "="*60)
    print("FEATURE SUMMARY")
    print("="*60)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nSample features:")
    print(df[['home_avg_goals_scored_l5', 'away_avg_goals_scored_l5', 
              'expected_total_goals', 'over_2_5']].head(10))
    
    print(f"\nOver 2.5 distribution:")
    print(df['over_2_5'].value_counts())
    
    print(f"\nBTTS distribution:")
    print(df['btts'].value_counts())