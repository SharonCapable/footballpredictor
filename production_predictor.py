"""
Production Football Prediction System
Trains models, saves them, and predicts future matches
Supports multiple leagues
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

class FootballPredictor:
    def __init__(self, league_name='epl'):
        """
        Initialize predictor for a specific league
        
        Args:
            league_name: 'portugal', 'epl', 'la_liga', etc.
        """
        self.league_name = league_name
        self.models = {}
        self.feature_names = []
        self.team_stats = {}  # Store team statistics for predictions
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
    
    # ==================== STEP 1: PREPARE DATA ====================
    
    def prepare_league_data(self, csv_path=None):
        """
        Load and prepare data for a league
        
        Args:
            csv_path: Path to odds CSV file (e.g., 'data/odds/portugal_with_odds.csv')
        """
        print("="*60)
        print(f"PREPARING {self.league_name.upper()} DATA")
        print("="*60)
        
        if csv_path is None:
            csv_path = f'data/odds/{self.league_name}_with_odds.csv'
        
        # Load data
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        
        # Create targets
        df['total_goals'] = df['FTHG'] + df['FTAG']
        df['over_2_5'] = (df['total_goals'] > 2.5).astype(int)
        df['over_1_5'] = (df['total_goals'] > 1.5).astype(int)
        df['over_3_5'] = (df['total_goals'] > 3.5).astype(int)
        df['btts'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)
        
        df = df.sort_values('Date').reset_index(drop=True)
        
        print(f"✓ Loaded {len(df)} matches")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        return df
    
    def engineer_features(self, df, lookback_windows=[3, 5, 10]):
        """
        Create features with multiple lookback windows
        
        Args:
            df: DataFrame with match data
            lookback_windows: List of lookback periods [3, 5, 10, etc.]
        """
        print("\n" + "="*60)
        print("ENGINEERING FEATURES")
        print("="*60)
        
        features_list = []
        
        for idx, match in df.iterrows():
            if idx % 100 == 0:
                print(f"  Processing {idx}/{len(df)}...")
            
            match_date = match['Date']
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            
            # Get previous matches
            home_prev = df[
                ((df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team)) &
                (df['Date'] < match_date)
            ].tail(max(lookback_windows))
            
            away_prev = df[
                ((df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team)) &
                (df['Date'] < match_date)
            ].tail(max(lookback_windows))
            
            # Need minimum history
            if len(home_prev) < min(lookback_windows) or len(away_prev) < min(lookback_windows):
                continue
            
            # Extract goals
            home_gf, home_ga = self._extract_team_goals(home_prev, home_team)
            away_gf, away_ga = self._extract_team_goals(away_prev, away_team)
            
            # Home-only matches
            home_home = df[(df['HomeTeam'] == home_team) & (df['Date'] < match_date)].tail(5)
            away_away = df[(df['AwayTeam'] == away_team) & (df['Date'] < match_date)].tail(5)
            
            features = {
                'match_idx': idx,
                'date': match_date,
                'home_team': home_team,
                'away_team': away_team,
                
                # Targets
                'total_goals': match['total_goals'],
                'over_2_5': match['over_2_5'],
                'over_1_5': match['over_1_5'],
                'over_3_5': match['over_3_5'],
                'btts': match['btts'],
            }
            
            # Add features for each lookback window
            for window in lookback_windows:
                features.update({
                    f'home_gf_l{window}': np.mean(home_gf[-window:]),
                    f'home_ga_l{window}': np.mean(home_ga[-window:]),
                    f'away_gf_l{window}': np.mean(away_gf[-window:]),
                    f'away_ga_l{window}': np.mean(away_ga[-window:]),
                })
            
            # Home/Away specific
            if len(home_home) >= 3:
                features['home_home_gf'] = home_home['FTHG'].mean()
                features['home_home_ga'] = home_home['FTAG'].mean()
            else:
                features['home_home_gf'] = 0
                features['home_home_ga'] = 0
            
            if len(away_away) >= 3:
                features['away_away_gf'] = away_away['FTAG'].mean()
                features['away_away_ga'] = away_away['FTHG'].mean()
            else:
                features['away_away_gf'] = 0
                features['away_away_ga'] = 0
            
            # Odds features (if available)
            if pd.notna(match.get('home_odds_avg')):
                features.update({
                    'home_odds': match['home_odds_avg'],
                    'draw_odds': match['draw_odds_avg'],
                    'away_odds': match['away_odds_avg'],
                    'home_prob': match['home_prob_norm'],
                    'draw_prob': match['draw_prob_norm'],
                    'away_prob': match['away_prob_norm'],
                    'combined_attacking': (1/match['home_odds_avg']) + (1/match['away_odds_avg']),
                    'odds_diff': match['home_odds_avg'] - match['away_odds_avg'],
                })
            else:
                # No odds available
                features.update({
                    'home_odds': 0, 'draw_odds': 0, 'away_odds': 0,
                    'home_prob': 0, 'draw_prob': 0, 'away_prob': 0,
                    'combined_attacking': 0, 'odds_diff': 0,
                })
            
            # Interaction features
            features['expected_goals'] = features['home_gf_l5'] + features['away_gf_l5']
            features['home_attack_vs_away_defense'] = features['home_gf_l5'] / max(features['away_ga_l5'], 0.5)
            features['away_attack_vs_home_defense'] = features['away_gf_l5'] / max(features['home_ga_l5'], 0.5)
            
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        print(f"\n✓ Created {len(features_df)} feature sets")
        print(f"✓ Total features: {len(features_df.columns) - 8}")
        
        # Store team stats for future predictions
        self._store_team_stats(df)
        
        return features_df
    
    def _extract_team_goals(self, matches, team):
        """Extract goals for/against from team's perspective"""
        gf, ga = [], []
        
        for _, m in matches.iterrows():
            if m['HomeTeam'] == team:
                gf.append(m['FTHG'])
                ga.append(m['FTAG'])
            else:
                gf.append(m['FTAG'])
                ga.append(m['FTHG'])
        
        return gf, ga
    
    def _store_team_stats(self, df):
        """Store latest team statistics for making predictions"""
        for team in set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()):
            team_matches = df[
                (df['HomeTeam'] == team) | (df['AwayTeam'] == team)
            ].tail(10)
            
            gf, ga = self._extract_team_goals(team_matches, team)
            
            self.team_stats[team] = {
                'gf_l5': np.mean(gf[-5:]) if len(gf) >= 5 else 0,
                'ga_l5': np.mean(ga[-5:]) if len(ga) >= 5 else 0,
                'gf_l10': np.mean(gf) if len(gf) > 0 else 0,
                'ga_l10': np.mean(ga) if len(ga) > 0 else 0,
            }
    
    # ==================== STEP 2: TRAIN MODELS ====================
    
    def train_models(self, features_df, targets=['over_2_5', 'btts'], use_deep_learning=False):
        """
        Train models for multiple targets
        
        Args:
            features_df: DataFrame with engineered features
            targets: List of target variables to predict
            use_deep_learning: Whether to use XGBoost (better performance)
        """
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        # Feature columns
        feature_cols = [col for col in features_df.columns if col not in [
            'match_idx', 'date', 'home_team', 'away_team',
            'total_goals', 'over_2_5', 'over_1_5', 'over_3_5', 'btts'
        ]]
        
        self.feature_names = feature_cols
        
        X = features_df[feature_cols].fillna(0)
        
        results = {}
        
        for target in targets:
            print(f"\n{'='*60}")
            print(f"Training: {target.upper()}")
            print(f"{'='*60}")
            
            y = features_df[target]
            
            # Temporal split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            print(f"Train: {len(X_train)} | Test: {len(X_test)}")
            
            if use_deep_learning:
                # XGBoost with hyperparameter tuning
                print("Using XGBoost with GridSearch...")
                
                param_grid = {
                    'n_estimators': [150, 200],
                    'max_depth': [10, 15],
                    'learning_rate': [0.05, 0.1],
                    'subsample': [0.8, 1.0]
                }
                
                xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
                
                grid_search = GridSearchCV(
                    xgb_model, param_grid,
                    cv=3, scoring='accuracy',
                    verbose=1, n_jobs=-1
                )
                
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                
                print(f"Best params: {grid_search.best_params_}")
            
            else:
                # Random Forest (faster)
                print("Using Random Forest...")
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"\nAccuracy: {accuracy*100:.2f}%")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            # Store model
            self.models[target] = model
            
            results[target] = {
                'accuracy': accuracy,
                'model': model,
                'test_predictions': y_pred,
                'test_probabilities': y_pred_proba
            }
        
        return results
    
    # ==================== STEP 3: SAVE MODELS ====================
    
    def save_models(self):
        """Save trained models to disk"""
        print("\n" + "="*60)
        print("SAVING MODELS")
        print("="*60)
        
        model_data = {
            'league': self.league_name,
            'models': self.models,
            'feature_names': self.feature_names,
            'team_stats': self.team_stats,
            'trained_date': datetime.now().isoformat()
        }
        
        filename = f'models/{self.league_name}_predictor.pkl'
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Saved to: {filename}")
        print(f"  Models: {list(self.models.keys())}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Teams: {len(self.team_stats)}")
    
    @classmethod
    def load_models(cls, league_name):
        """Load saved models from disk"""
        filename = f'models/{league_name}_predictor.pkl'
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"No saved model found for {league_name}")
        
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        predictor = cls(league_name)
        predictor.models = model_data['models']
        predictor.feature_names = model_data['feature_names']
        predictor.team_stats = model_data['team_stats']
        
        print(f"✓ Loaded {league_name} predictor")
        print(f"  Trained: {model_data['trained_date']}")
        
        return predictor
    
    # ==================== STEP 4: PREDICT FUTURE MATCHES ====================
    
    def predict_match(self, home_team, away_team, home_odds=None, draw_odds=None, away_odds=None):
        """
        Predict outcome for a future match
        
        Args:
            home_team: Home team name
            away_team: Away team name
            home_odds: Betting odds for home win (optional)
            draw_odds: Betting odds for draw (optional)
            away_odds: Betting odds for away win (optional)
        
        Returns:
            dict: Predictions for all targets
        """
        print("\n" + "="*60)
        print(f"PREDICTING: {home_team} vs {away_team}")
        print("="*60)
        
        # Check if teams exist in our data
        if home_team not in self.team_stats:
            print(f"⚠️ {home_team} not found in training data")
            print(f"Available teams: {list(self.team_stats.keys())[:10]}...")
            return None
        
        if away_team not in self.team_stats:
            print(f"⚠️ {away_team} not found in training data")
            return None
        
        # Build feature vector
        home_stats = self.team_stats[home_team]
        away_stats = self.team_stats[away_team]
        
        features = {
            'home_gf_l3': home_stats['gf_l5'],  # Approximate
            'home_ga_l3': home_stats['ga_l5'],
            'home_gf_l5': home_stats['gf_l5'],
            'home_ga_l5': home_stats['ga_l5'],
            'home_gf_l10': home_stats['gf_l10'],
            'home_ga_l10': home_stats['ga_l10'],
            
            'away_gf_l3': away_stats['gf_l5'],
            'away_ga_l3': away_stats['ga_l5'],
            'away_gf_l5': away_stats['gf_l5'],
            'away_ga_l5': away_stats['ga_l5'],
            'away_gf_l10': away_stats['gf_l10'],
            'away_ga_l10': away_stats['ga_l10'],
            
            'home_home_gf': home_stats['gf_l5'],  # Approximate
            'home_home_ga': home_stats['ga_l5'],
            'away_away_gf': away_stats['gf_l5'],
            'away_away_ga': away_stats['ga_l5'],
        }
        
        # Add odds if provided
        if home_odds and draw_odds and away_odds:
            total_prob = (1/home_odds) + (1/draw_odds) + (1/away_odds)
            features.update({
                'home_odds': home_odds,
                'draw_odds': draw_odds,
                'away_odds': away_odds,
                'home_prob': (1/home_odds) / total_prob,
                'draw_prob': (1/draw_odds) / total_prob,
                'away_prob': (1/away_odds) / total_prob,
                'combined_attacking': (1/home_odds) + (1/away_odds),
                'odds_diff': home_odds - away_odds,
            })
        else:
            features.update({
                'home_odds': 0, 'draw_odds': 0, 'away_odds': 0,
                'home_prob': 0, 'draw_prob': 0, 'away_prob': 0,
                'combined_attacking': 0, 'odds_diff': 0,
            })
        
        # Interaction features
        features['expected_goals'] = features['home_gf_l5'] + features['away_gf_l5']
        features['home_attack_vs_away_defense'] = features['home_gf_l5'] / max(features['away_ga_l5'], 0.5)
        features['away_attack_vs_home_defense'] = features['away_gf_l5'] / max(features['home_ga_l5'], 0.5)
        
        # Create feature vector in correct order
        X = pd.DataFrame([features])[self.feature_names].fillna(0)
        
        # Make predictions
        predictions = {}
        
        for target, model in self.models.items():
            pred = model.predict(X)[0]
            pred_proba = model.predict_proba(X)[0]
            
            predictions[target] = {
                'prediction': 'Yes' if pred == 1 else 'No',
                'probability': pred_proba[1] * 100,  # Probability of "Yes"
                'confidence': max(pred_proba) * 100
            }
        
        # Display predictions
        print(f"\nPredictions:")
        for target, pred in predictions.items():
            print(f"  {target.upper():15s}: {pred['prediction']:3s} ({pred['probability']:.1f}%)")
        
        if home_odds:
            print(f"\nProvided Odds:")
            print(f"  Home: {home_odds:.2f} | Draw: {draw_odds:.2f} | Away: {away_odds:.2f}")
        
        return predictions
    
    # ==================== COMPLETE WORKFLOW ====================
    
    def train_complete_system(self, csv_path=None, use_deep_learning=False):
        """
        Complete training workflow
        """
        print("\n" + "="*60)
        print(f"TRAINING COMPLETE SYSTEM: {self.league_name.upper()}")
        print("="*60)
        
        # Step 1: Prepare data
        df = self.prepare_league_data(csv_path)
        
        # Step 2: Engineer features
        features_df = self.engineer_features(df, lookback_windows=[3, 5, 10])
        
        # Step 3: Train models
        results = self.train_models(
            features_df,
            targets=['over_2_5', 'over_1_5', 'btts'],
            use_deep_learning=use_deep_learning
        )
        
        # Step 4: Save models
        self.save_models()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        
        for target, data in results.items():
            print(f"{target.upper()}: {data['accuracy']*100:.2f}%")
        
        return results


# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    
    # Example 1: Train Portuguese league model
    print("TRAINING English Premier League MODEL")
    print("="*60)
    
    epl = FootballPredictor('epl')
    epl.train_complete_system(
        csv_path='data/odds/epl_with_odds.csv',
        use_deep_learning=False  # Set True for better accuracy (slower)
    )
    
    # Example 2: Make a prediction
    print("\n\nEXAMPLE PREDICTION")
    print("="*60)
    
    epl.predict_match(
        home_team='Liverpool',
        away_team='Wolves',
        home_odds=1.85,
        draw_odds=3.50,
        away_odds=4.20
    )
    
    # Example 3: Train EPL model (if you have data)
    # epl = FootballPredictor('epl')
    # epl.train_complete_system(csv_path='data/odds/epl_with_odds.csv')