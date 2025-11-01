"""
Odds-Independent Football Predictor
Works with OR without betting odds
Relies on intrinsic team quality features
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class OddsIndependentPredictor:
    """
    A model that doesn't rely on betting odds
    Uses intrinsic team/match features instead
    """
    
    def __init__(self, league_name='portugal'):
        self.league_name = league_name
        self.models = {}
        self.feature_names = []
        self.league_stats = {}  # Store league-wide statistics
        self.team_stats = {}    # Store team statistics
    
    def load_and_prepare_data(self, csv_path):
        """Load match data"""
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        
        # Create targets
        df['total_goals'] = df['FTHG'] + df['FTAG']
        df['over_2_5'] = (df['total_goals'] > 2.5).astype(int)
        df['btts'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)
        
        # Create outcome
        df['home_win'] = (df['FTHG'] > df['FTAG']).astype(int)
        df['draw'] = (df['FTHG'] == df['FTAG']).astype(int)
        df['away_win'] = (df['FTHG'] < df['FTAG']).astype(int)
        
        df = df.sort_values('Date').reset_index(drop=True)
        
        print(f"✓ Loaded {len(df)} matches")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        return df
    
    def calculate_league_standings(self, df, up_to_date):
        """
        Calculate league table up to a specific date
        Returns dict with team standings
        """
        matches_so_far = df[df['Date'] < up_to_date].copy()
        
        standings = {}
        
        for team in set(matches_so_far['HomeTeam'].unique()) | set(matches_so_far['AwayTeam'].unique()):
            team_matches = matches_so_far[
                (matches_so_far['HomeTeam'] == team) | (matches_so_far['AwayTeam'] == team)
            ]
            
            points = 0
            gf = 0
            ga = 0
            wins = 0
            draws = 0
            losses = 0
            
            for _, match in team_matches.iterrows():
                if match['HomeTeam'] == team:
                    gf += match['FTHG']
                    ga += match['FTAG']
                    if match['FTHG'] > match['FTAG']:
                        wins += 1
                        points += 3
                    elif match['FTHG'] == match['FTAG']:
                        draws += 1
                        points += 1
                    else:
                        losses += 1
                else:
                    gf += match['FTAG']
                    ga += match['FTHG']
                    if match['FTAG'] > match['FTHG']:
                        wins += 1
                        points += 3
                    elif match['FTAG'] == match['FTHG']:
                        draws += 1
                        points += 1
                    else:
                        losses += 1
            
            standings[team] = {
                'points': points,
                'played': len(team_matches),
                'wins': wins,
                'draws': draws,
                'losses': losses,
                'gf': gf,
                'ga': ga,
                'gd': gf - ga,
                'ppg': points / len(team_matches) if len(team_matches) > 0 else 0
            }
        
        # Calculate positions
        sorted_teams = sorted(standings.items(), 
                            key=lambda x: (x[1]['points'], x[1]['gd'], x[1]['gf']), 
                            reverse=True)
        
        for pos, (team, stats) in enumerate(sorted_teams, 1):
            standings[team]['position'] = pos
        
        return standings
    
    def engineer_rich_features(self, df):
        """
        Create rich features without relying on odds
        """
        print("\n" + "="*60)
        print("ENGINEERING ODDS-INDEPENDENT FEATURES")
        print("="*60)
        
        features_list = []
        
        # Calculate league average for normalization
        league_avg_goals = df['total_goals'].mean()
        
        for idx, match in df.iterrows():
            if idx % 100 == 0:
                print(f"  Processing {idx}/{len(df)}...")
            
            match_date = match['Date']
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            
            # Get league standings at this point
            standings = self.calculate_league_standings(df, match_date)
            
            # Skip if teams don't have history yet
            if home_team not in standings or away_team not in standings:
                continue
            
            if standings[home_team]['played'] < 5 or standings[away_team]['played'] < 5:
                continue
            
            # Get recent matches
            home_recent = df[
                ((df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team)) &
                (df['Date'] < match_date)
            ].tail(10)
            
            away_recent = df[
                ((df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team)) &
                (df['Date'] < match_date)
            ].tail(10)
            
            # Home-only and away-only
            home_home = df[(df['HomeTeam'] == home_team) & (df['Date'] < match_date)].tail(5)
            away_away = df[(df['AwayTeam'] == away_team) & (df['Date'] < match_date)].tail(5)
            
            # Extract goals
            home_gf_recent, home_ga_recent = self._extract_team_goals(home_recent, home_team)
            away_gf_recent, away_ga_recent = self._extract_team_goals(away_recent, away_team)
            
            # Calculate rest days (days since last match)
            home_rest_days = (match_date - home_recent.iloc[-1]['Date']).days if len(home_recent) > 0 else 7
            away_rest_days = (match_date - away_recent.iloc[-1]['Date']).days if len(away_recent) > 0 else 7
            
            # Calculate form (weighted recent matches more)
            home_form_l5 = self._calculate_weighted_form(home_recent.tail(5), home_team)
            away_form_l5 = self._calculate_weighted_form(away_recent.tail(5), away_team)
            
            # Head-to-head history
            h2h = df[
                (((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
                 ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))) &
                (df['Date'] < match_date)
            ].tail(5)
            
            h2h_home_wins = len(h2h[
                ((h2h['HomeTeam'] == home_team) & (h2h['FTHG'] > h2h['FTAG'])) |
                ((h2h['AwayTeam'] == home_team) & (h2h['FTAG'] > h2h['FTHG']))
            ]) if len(h2h) > 0 else 0
            
            features = {
                'match_idx': idx,
                'date': match_date,
                'home_team': home_team,
                'away_team': away_team,
                
                # Targets
                'over_2_5': match['over_2_5'],
                'btts': match['btts'],
                
                # ============ LEAGUE POSITION FEATURES ============
                'home_position': standings[home_team]['position'],
                'away_position': standings[away_team]['position'],
                'home_points': standings[home_team]['points'],
                'away_points': standings[away_team]['points'],
                'home_ppg': standings[home_team]['ppg'],
                'away_ppg': standings[away_team]['ppg'],
                'position_diff': standings[away_team]['position'] - standings[home_team]['position'],
                
                # ============ GOAL DIFFERENCE (KEY FEATURE!) ============
                'home_gd': standings[home_team]['gd'],
                'away_gd': standings[away_team]['gd'],
                'gd_diff': standings[home_team]['gd'] - standings[away_team]['gd'],
                
                # ============ RECENT FORM ============
                'home_gf_l5': np.mean(home_gf_recent[-5:]),
                'home_ga_l5': np.mean(home_ga_recent[-5:]),
                'home_gf_l10': np.mean(home_gf_recent),
                'home_ga_l10': np.mean(home_ga_recent),
                
                'away_gf_l5': np.mean(away_gf_recent[-5:]),
                'away_ga_l5': np.mean(away_ga_recent[-5:]),
                'away_gf_l10': np.mean(away_gf_recent),
                'away_ga_l10': np.mean(away_ga_recent),
                
                # ============ FORM POINTS (WEIGHTED) ============
                'home_form_l5': home_form_l5,
                'away_form_l5': away_form_l5,
                'form_diff': home_form_l5 - away_form_l5,
                
                # ============ HOME/AWAY SPECIFIC ============
                'home_home_gf': home_home['FTHG'].mean() if len(home_home) >= 3 else 0,
                'home_home_ga': home_home['FTAG'].mean() if len(home_home) >= 3 else 0,
                'away_away_gf': away_away['FTAG'].mean() if len(away_away) >= 3 else 0,
                'away_away_ga': away_away['FTHG'].mean() if len(away_away) >= 3 else 0,
                
                # ============ REST/FATIGUE ============
                'home_rest_days': home_rest_days,
                'away_rest_days': away_rest_days,
                'rest_advantage': home_rest_days - away_rest_days,
                
                # ============ HEAD-TO-HEAD ============
                'h2h_matches': len(h2h),
                'h2h_home_advantage': h2h_home_wins / len(h2h) if len(h2h) > 0 else 0.5,
                
                # ============ LEAGUE CONTEXT ============
                'home_top_6': 1 if standings[home_team]['position'] <= 6 else 0,
                'away_top_6': 1 if standings[away_team]['position'] <= 6 else 0,
                'home_bottom_6': 1 if standings[home_team]['position'] >= len(standings) - 5 else 0,
                'away_bottom_6': 1 if standings[away_team]['position'] >= len(standings) - 5 else 0,
                
                # ============ ATTACKING/DEFENSIVE STRENGTH ============
                'home_attack_strength': standings[home_team]['gf'] / standings[home_team]['played'] / league_avg_goals,
                'home_defense_strength': league_avg_goals / (standings[home_team]['ga'] / standings[home_team]['played']),
                'away_attack_strength': standings[away_team]['gf'] / standings[away_team]['played'] / league_avg_goals,
                'away_defense_strength': league_avg_goals / (standings[away_team]['ga'] / standings[away_team]['played']),
                
                # ============ INTERACTION FEATURES ============
                'expected_goals': (standings[home_team]['gf'] / standings[home_team]['played']) + 
                                 (standings[away_team]['gf'] / standings[away_team]['played']),
                'home_attack_vs_away_defense': (standings[home_team]['gf'] / standings[home_team]['played']) / 
                                               max((standings[away_team]['ga'] / standings[away_team]['played']), 0.5),
                'quality_gap': abs(standings[home_team]['ppg'] - standings[away_team]['ppg']),
            }
            
            # Add odds if available (optional enhancement)
            if pd.notna(match.get('home_odds_avg')):
                features['odds_available'] = 1
                features['home_odds'] = match['home_odds_avg']
                features['combined_attacking_odds'] = (1/match['home_odds_avg']) + (1/match['away_odds_avg'])
            else:
                features['odds_available'] = 0
                features['home_odds'] = 0
                features['combined_attacking_odds'] = 0
            
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        print(f"\n✓ Created {len(features_df)} feature sets")
        print(f"✓ Total features: {len([c for c in features_df.columns if c not in ['match_idx', 'date', 'home_team', 'away_team', 'over_2_5', 'btts']])}")
        
        return features_df
    
    def _extract_team_goals(self, matches, team):
        """Extract goals for/against"""
        gf, ga = [], []
        for _, m in matches.iterrows():
            if m['HomeTeam'] == team:
                gf.append(m['FTHG'])
                ga.append(m['FTAG'])
            else:
                gf.append(m['FTAG'])
                ga.append(m['FTHG'])
        return gf, ga
    
    def _calculate_weighted_form(self, recent_matches, team):
        """Calculate form with recent matches weighted more heavily"""
        if len(recent_matches) == 0:
            return 0
        
        points = []
        for _, match in recent_matches.iterrows():
            if match['HomeTeam'] == team:
                if match['FTHG'] > match['FTAG']:
                    points.append(3)
                elif match['FTHG'] == match['FTAG']:
                    points.append(1)
                else:
                    points.append(0)
            else:
                if match['FTAG'] > match['FTHG']:
                    points.append(3)
                elif match['FTAG'] == match['FTHG']:
                    points.append(1)
                else:
                    points.append(0)
        
        # Weight recent matches more: [0.5, 0.75, 1.0, 1.25, 1.5]
        weights = np.linspace(0.5, 1.5, len(points))
        weighted_points = np.average(points, weights=weights)
        
        return weighted_points
    
    def train_model(self, features_df, target='over_2_5'):
        """Train model"""
        print("\n" + "="*60)
        print(f"TRAINING: {target.upper()}")
        print("="*60)
        
        # Feature columns
        feature_cols = [col for col in features_df.columns if col not in [
            'match_idx', 'date', 'home_team', 'away_team', 'over_2_5', 'btts'
        ]]
        
        X = features_df[feature_cols].fillna(0)
        y = features_df[target]
        
        # Temporal split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Train: {len(X_train)} | Test: {len(X_test)}")
        
        # Train
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=3,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nAccuracy: {accuracy*100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 Most Important Features:")
        print(feature_importance.head(15).to_string(index=False))
        
        # Check if odds are in top 10
        odds_in_top_10 = len([f for f in feature_importance.head(10)['feature'] if 'odds' in f.lower()])
        print(f"\nOdds features in top 10: {odds_in_top_10}/10")
        
        if odds_in_top_10 <= 3:
            print("✓ Model is NOT overly dependent on odds!")
        else:
            print("⚠️ Model still relies heavily on odds")
        
        self.models[target] = model
        self.feature_names = feature_cols
        
        return accuracy, feature_importance
    
    def train_complete_system(self, csv_path):
        """Complete training"""
        print("="*60)
        print(f"ODDS-INDEPENDENT MODEL: {self.league_name.upper()}")
        print("="*60)
        
        df = self.load_and_prepare_data(csv_path)
        features_df = self.engineer_rich_features(df)
        
        results = {}
        for target in ['over_2_5', 'btts']:
            accuracy, importance = self.train_model(features_df, target)
            results[target] = {'accuracy': accuracy, 'importance': importance}
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        
        for target, data in results.items():
            print(f"{target.upper()}: {data['accuracy']*100:.2f}%")
        
        return results


if __name__ == "__main__":
    predictor = OddsIndependentPredictor('portugal')
    results = predictor.train_complete_system('data/odds/portugal_with_odds.csv')