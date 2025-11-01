"""
Complete Football Prediction System
- League-specific models (for simulation)
- Universal multi-league model (for production)
- Hyperparameter tuning
- Model saving/loading
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class CompletePredictionSystem:
    """
    Manages both league-specific and universal models
    """
    
    def __init__(self):
        self.league_models = {}     # One model per league
        self.universal_model = None  # One model for all leagues
        self.models_dir = 'models/complete_system'
        os.makedirs(self.models_dir, exist_ok=True)
    
    # ==================== PART 1: LEAGUE-SPECIFIC MODELS ====================
    
    def train_league_model(self, league_name, csv_path, tune_hyperparameters=False):
        """
        Train a model for ONE specific league
        Used for simulation/analysis
        
        Args:
            league_name: 'portugal', 'epl', 'la_liga', etc.
            csv_path: Path to odds CSV
            tune_hyperparameters: If True, runs GridSearch (slow but better)
        """
        print("\n" + "="*60)
        print(f"TRAINING LEAGUE-SPECIFIC MODEL: {league_name.upper()}")
        print("="*60)
        
        # Load data
        df = self._load_data(csv_path)
        
        # Engineer features
        features_df = self._engineer_features(df, league_name)
        
        # Prepare for training
        X, y, feature_names = self._prepare_training_data(features_df, target='over_2_5')
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Train: {len(X_train)} | Test: {len(X_test)}")
        
        # Train model
        if tune_hyperparameters:
            print("\n🔧 Tuning hyperparameters (this may take 5-10 minutes)...")
            model = self._tune_hyperparameters(X_train, y_train)
        else:
            print("\n⚡ Using default hyperparameters...")
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
        
        print(f"\n✓ Accuracy: {accuracy*100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        self.league_models[league_name] = {
            'model': model,
            'feature_names': feature_names,
            'accuracy': accuracy,
            'trained_date': datetime.now().isoformat(),
            'num_matches': len(df)
        }
        
        self._save_league_model(league_name)
        
        return accuracy
    
    def _tune_hyperparameters(self, X_train, y_train):
        """
        Hyperparameter tuning using GridSearchCV
        Tests different combinations to find the best
        """
        param_grid = {
            'n_estimators': [200, 300, 400],           # Number of trees
            'max_depth': [15, 20, 25],                 # Tree depth
            'min_samples_split': [2, 3, 5],           # Min samples to split
            'max_features': ['sqrt', 'log2'],         # Features per split
            'min_samples_leaf': [1, 2]                # Min samples in leaf
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            rf, param_grid,
            cv=5,                    # 5-fold cross-validation
            scoring='accuracy',
            verbose=1,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\n✓ Best hyperparameters found:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"Best CV score: {grid_search.best_score_*100:.2f}%")
        
        return grid_search.best_estimator_
    
    # ==================== PART 2: UNIVERSAL MULTI-LEAGUE MODEL ====================
    
    def train_universal_model(self, league_csvs, tune_hyperparameters=False):
        """
        Train ONE model on ALL leagues combined
        Used for production predictions across any league
        
        Args:
            league_csvs: Dict like {'portugal': 'path.csv', 'epl': 'path.csv'}
            tune_hyperparameters: If True, optimizes settings
        """
        print("\n" + "="*60)
        print("TRAINING UNIVERSAL MULTI-LEAGUE MODEL")
        print("="*60)
        
        all_features = []
        
        # Load and combine all leagues
        for league_name, csv_path in league_csvs.items():
            print(f"\nLoading {league_name}...")
            df = self._load_data(csv_path)
            features_df = self._engineer_features(df, league_name)
            
            # Add league identifier
            features_df['league'] = league_name
            
            all_features.append(features_df)
        
        # Combine all data
        combined_df = pd.concat(all_features, ignore_index=True)
        
        print(f"\n✓ Combined dataset: {len(combined_df)} matches across {len(league_csvs)} leagues")
        
        # One-hot encode league
        combined_df = pd.get_dummies(combined_df, columns=['league'], prefix='league')
        
        # Prepare training data
        X, y, feature_names = self._prepare_training_data(combined_df, target='over_2_5')
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Train: {len(X_train)} | Test: {len(X_test)}")
        
        # Train
        if tune_hyperparameters:
            print("\n🔧 Tuning hyperparameters...")
            model = self._tune_hyperparameters(X_train, y_train)
        else:
            model = RandomForestClassifier(
                n_estimators=400,      # More trees for multi-league
                max_depth=25,
                min_samples_split=3,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✓ Universal Model Accuracy: {accuracy*100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Test on each league separately
        print("\n" + "-"*60)
        print("Per-League Performance:")
        print("-"*60)
        
        for league_name in league_csvs.keys():
            league_col = f'league_{league_name}'
            if league_col in X_test.columns:
                league_mask = X_test[league_col] == 1
                if league_mask.sum() > 0:
                    league_accuracy = accuracy_score(
                        y_test[league_mask],
                        y_pred[league_mask]
                    )
                    print(f"{league_name.upper():<15}: {league_accuracy*100:.2f}% ({league_mask.sum()} matches)")
        
        # Save universal model
        self.universal_model = {
            'model': model,
            'feature_names': feature_names,
            'accuracy': accuracy,
            'leagues': list(league_csvs.keys()),
            'trained_date': datetime.now().isoformat(),
            'num_matches': len(combined_df)
        }
        
        self._save_universal_model()
        
        return accuracy
    
    # ==================== HELPER METHODS ====================
    
    def _load_data(self, csv_path):
        """Load match data"""
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        df['total_goals'] = df['FTHG'] + df['FTAG']
        df['over_2_5'] = (df['total_goals'] > 2.5).astype(int)
        df['btts'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)
        return df.sort_values('Date').reset_index(drop=True)
    
    def _engineer_features(self, df, league_name):
        """
        Simplified feature engineering
        (Using the odds-independent features from previous script)
        """
        # For brevity, using key features only
        # In production, use full feature engineering from odds_independent_predictor.py
        
        features_list = []
        
        for idx, match in df.iterrows():
            if idx < 50:  # Skip first 50 matches (need history)
                continue
            
            match_date = match['Date']
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            
            # Get recent matches
            home_recent = df[
                ((df['HomeTeam'] == home_team) | (df['AwayTeam'] == away_team)) &
                (df['Date'] < match_date)
            ].tail(10)
            
            away_recent = df[
                ((df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team)) &
                (df['Date'] < match_date)
            ].tail(10)
            
            if len(home_recent) < 5 or len(away_recent) < 5:
                continue
            
            # Extract goals
            home_gf, home_ga = [], []
            for _, m in home_recent.iterrows():
                if m['HomeTeam'] == home_team:
                    home_gf.append(m['FTHG'])
                    home_ga.append(m['FTAG'])
                else:
                    home_gf.append(m['FTAG'])
                    home_ga.append(m['FTHG'])
            
            away_gf, away_ga = [], []
            for _, m in away_recent.iterrows():
                if m['AwayTeam'] == away_team:
                    away_gf.append(m['FTAG'])
                    away_ga.append(m['FTHG'])
                else:
                    away_gf.append(m['FTHG'])
                    away_ga.append(m['FTAG'])
            
            features = {
                'home_gf_l5': np.mean(home_gf[-5:]),
                'home_ga_l5': np.mean(home_ga[-5:]),
                'home_gf_l10': np.mean(home_gf),
                'home_ga_l10': np.mean(home_ga),
                'away_gf_l5': np.mean(away_gf[-5:]),
                'away_ga_l5': np.mean(away_ga[-5:]),
                'away_gf_l10': np.mean(away_gf),
                'away_ga_l10': np.mean(away_ga),
                'expected_goals': np.mean(home_gf[-5:]) + np.mean(away_gf[-5:]),
                'over_2_5': match['over_2_5'],
                'btts': match['btts']
            }
            
            # Add odds if available
            if pd.notna(match.get('home_odds_avg')):
                features['combined_attacking_odds'] = (1/match['home_odds_avg']) + (1/match['away_odds_avg'])
            else:
                features['combined_attacking_odds'] = 0
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _prepare_training_data(self, features_df, target='over_2_5'):
        """Prepare X, y for training"""
        feature_cols = [col for col in features_df.columns 
                       if col not in ['over_2_5', 'btts']]
        
        X = features_df[feature_cols].fillna(0)
        y = features_df[target]
        
        return X, y, feature_cols
    
    # ==================== SAVE/LOAD ====================
    
    def _save_league_model(self, league_name):
        """Save league-specific model"""
        filepath = os.path.join(self.models_dir, f'{league_name}_model.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(self.league_models[league_name], f)
        print(f"✓ Saved to {filepath}")
    
    def _save_universal_model(self):
        """Save universal model"""
        filepath = os.path.join(self.models_dir, 'universal_model.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(self.universal_model, f)
        print(f"✓ Saved universal model to {filepath}")
    
    def load_league_model(self, league_name):
        """Load league-specific model"""
        filepath = os.path.join(self.models_dir, f'{league_name}_model.pkl')
        with open(filepath, 'rb') as f:
            self.league_models[league_name] = pickle.load(f)
        print(f"✓ Loaded {league_name} model")
    
    def load_universal_model(self):
        """Load universal model"""
        filepath = os.path.join(self.models_dir, 'universal_model.pkl')
        with open(filepath, 'rb') as f:
            self.universal_model = pickle.load(f)
        print(f"✓ Loaded universal model")
    
    # ==================== PREDICT ====================
    
    def predict_with_league_model(self, league_name, match_features):
        """Predict using league-specific model"""
        if league_name not in self.league_models:
            raise ValueError(f"No model trained for {league_name}")
        
        model_data = self.league_models[league_name]
        model = model_data['model']
        
        # Prepare features
        X = pd.DataFrame([match_features])[model_data['feature_names']].fillna(0)
        
        pred = model.predict(X)[0]
        pred_proba = model.predict_proba(X)[0]
        
        return {
            'prediction': 'Over 2.5' if pred == 1 else 'Under 2.5',
            'probability': pred_proba[1] * 100,
            'confidence': max(pred_proba) * 100,
            'model_used': f'{league_name}_specific'
        }
    
    def predict_with_universal_model(self, match_features, league_name):
        """Predict using universal model"""
        if not self.universal_model:
            raise ValueError("No universal model loaded")
        
        model = self.universal_model['model']
        
        # Add league one-hot encoding
        for league in self.universal_model['leagues']:
            match_features[f'league_{league}'] = 1 if league == league_name else 0
        
        # Prepare features
        X = pd.DataFrame([match_features])[self.universal_model['feature_names']].fillna(0)
        
        pred = model.predict(X)[0]
        pred_proba = model.predict_proba(X)[0]
        
        return {
            'prediction': 'Over 2.5' if pred == 1 else 'Under 2.5',
            'probability': pred_proba[1] * 100,
            'confidence': max(pred_proba) * 100,
            'model_used': 'universal'
        }
    
    # ==================== SUMMARY ====================
    
    def print_summary(self):
        """Print system summary"""
        print("\n" + "="*60)
        print("PREDICTION SYSTEM SUMMARY")
        print("="*60)
        
        print("\nLeague-Specific Models:")
        for league, data in self.league_models.items():
            print(f"  {league.upper():<15}: {data['accuracy']*100:.2f}% ({data['num_matches']} matches)")
        
        if self.universal_model:
            print(f"\nUniversal Model:")
            print(f"  Accuracy: {self.universal_model['accuracy']*100:.2f}%")
            print(f"  Leagues: {', '.join(self.universal_model['leagues'])}")
            print(f"  Total matches: {self.universal_model['num_matches']}")


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    system = CompletePredictionSystem()
    
    print("="*60)
    print("TRAINING COMPLETE PREDICTION SYSTEM")
    print("="*60)
    
    # Step 1: Train league-specific models
    print("\n[1/2] Training league-specific models...")
    
    system.train_league_model(
        'portugal',
        'data/odds/portugal_with_odds.csv',
        tune_hyperparameters=False  # Set True for best accuracy (slower)
    )
    
    # Add more leagues as you download them
    # system.train_league_model('epl', 'data/odds/epl_with_odds.csv')
    # system.train_league_model('la_liga', 'data/odds/la_liga_with_odds.csv')
    
    # Step 2: Train universal model
    print("\n[2/2] Training universal model...")
    
    system.train_universal_model({
        'portugal': 'data/odds/portugal_with_odds.csv',
        # 'epl': 'data/odds/epl_with_odds.csv',  # Add when ready
    }, tune_hyperparameters=False)
    
    # Print summary
    system.print_summary()
    
    print("\n✓ System ready for frontend integration!")