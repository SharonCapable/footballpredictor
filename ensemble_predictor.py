"""
Ensemble Model - Combine Random Forest + XGBoost for Better Accuracy
Strategy: Average predictions from multiple models (wisdom of crowds)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EnsembleGoalPredictor:
    """Ensemble of multiple models for improved accuracy"""
    
    LEAGUES = ['england', 'spain', 'germany', 'italy', 'france']
    
    FORBIDDEN_FEATURES = [
        'home_score', 'away_score', 'outcome', 'result', 'FTR', 'FTHG', 'FTAG',
        'HTHG', 'HTAG', 'HTR', 'ht_home_score', 'ht_away_score',
        'B365H', 'B365D', 'B365A', 'B365CH', 'B365CD', 'B365CA',
        'BWH', 'BWD', 'BWA', 'BWCH', 'BWCD', 'BWCA',
        'IWH', 'IWD', 'IWA', 'IWCH', 'IWCD', 'IWCA',
        'PSH', 'PSD', 'PSA', 'PSCH', 'PSCD', 'PSCA',
        'WHH', 'WHD', 'WHA', 'WHCH', 'WHCD', 'WHCA',
        'VCH', 'VCD', 'VCA', 'VCCH', 'VCCD', 'VCCA',
        'MaxH', 'MaxD', 'MaxA', 'MaxCH', 'MaxCD', 'MaxCA',
        'AvgH', 'AvgD', 'AvgA', 'AvgCH', 'AvgCD', 'AvgCA',
        'B365AHH', 'B365AHA', 'MaxAHH', 'MaxAHA', 'AvgAHH', 'AvgAHA',
        'B365CAHH', 'B365CAHA', 'MaxCAHH', 'MaxCAHA', 'AvgCAHH', 'AvgCAHA',
        'B365>2.5', 'B365<2.5', 'Max>2.5', 'Max<2.5', 'Avg>2.5', 'Avg<2.5',
        'B365C>2.5', 'B365C<2.5', 'MaxC>2.5', 'MaxC<2.5', 'AvgC>2.5', 'AvgC<2.5',
        'P>2.5', 'P<2.5', 'PC>2.5', 'PC<2.5',
        'PAHH', 'PAHA', 'PCAHH', 'PCAHA',
        'AHh', 'AHCh', 'BFH', 'BFD', 'BFA',
        'BFEH', 'BFED', 'BFEA', 'BFCH', 'BFCD', 'BFCA',
        'BFECH', 'BFECD', 'BFECA', 'BFE>2.5', 'BFE<2.5',
        'BFEC>2.5', 'BFEC<2.5', 'BFEAHH', 'BFEAHA',
        'BFECAHH', 'BFECAHA', '1XBH', '1XBD', '1XBA',
        '1XBCH', '1XBCD', '1XBCA',
        'date', 'League', 'home_team', 'away_team', 'Season', 'Div', 'Time', 'Referee'
    ]
    
    def __init__(self, processed_dir='data/processed', model_dir='models/ensemble'):
        self.processed_dir = Path(processed_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.home_models = []
        self.away_models = []
        self.model_names = []
        self.feature_cols = None
        
    def merge_elo_with_advanced(self, league):
        """Merge ELO + advanced features"""
        advanced_file = self.processed_dir / f"{league}_advanced.csv"
        elo_file = self.processed_dir / f"{league}_with_elo.csv"
        
        if not advanced_file.exists() or not elo_file.exists():
            return None
        
        df_advanced = pd.read_csv(advanced_file)
        df_elo = pd.read_csv(elo_file)
        
        df_advanced['date'] = pd.to_datetime(df_advanced['date'])
        df_elo['date'] = pd.to_datetime(df_elo['date'])
        
        df_elo_subset = df_elo[['date', 'home_team', 'away_team', 
                                 'home_elo_before', 'away_elo_before', 
                                 'elo_diff', 'home_win_prob']]
        
        df_merged = pd.merge(df_advanced, df_elo_subset,
                            on=['date', 'home_team', 'away_team'], how='inner')
        
        return df_merged
    
    def load_data(self):
        """Load all league data"""
        print("\n" + "="*60)
        print("📥 LOADING DATA")
        print("="*60)
        
        all_data = []
        for league in self.LEAGUES:
            df = self.merge_elo_with_advanced(league)
            if df is not None:
                print(f"✓ {league}: {len(df)} matches")
                all_data.append(df)
        
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values('date').reset_index(drop=True)
        
        print(f"\n✓ Total: {len(combined)} matches")
        return combined
    
    def prepare_features(self, df):
        """Extract clean features"""
        all_cols = df.columns.tolist()
        clean_cols = [col for col in all_cols if col not in self.FORBIDDEN_FEATURES]
        
        feature_cols = []
        for col in clean_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_cols.append(col)
        
        return feature_cols
    
    def train_ensemble(self, test_size=0.2, validation_size=0.1):
        """
        Train ensemble of models with proper train/val/test split
        
        Args:
            test_size: Fraction for final testing
            validation_size: Fraction for hyperparameter validation
        """
        print("\n" + "="*60)
        print("🎯 TRAINING ENSEMBLE MODELS")
        print("="*60)
        
        # Load data
        df = self.load_data()
        if df is None or 'home_score' not in df.columns:
            print("❌ Data loading failed!")
            return None
        
        # Prepare features
        self.feature_cols = self.prepare_features(df)
        print(f"\n📊 Using {len(self.feature_cols)} features")
        
        # Extract X and y
        X = df[self.feature_cols].fillna(0).values
        y_home = df['home_score'].values
        y_away = df['away_score'].values
        
        # 3-way split: Train / Validation / Test
        total = len(X)
        test_idx = int(total * (1 - test_size))
        val_idx = int(test_idx * (1 - validation_size))
        
        X_train = X[:val_idx]
        X_val = X[val_idx:test_idx]
        X_test = X[test_idx:]
        
        y_home_train, y_home_val, y_home_test = y_home[:val_idx], y_home[val_idx:test_idx], y_home[test_idx:]
        y_away_train, y_away_val, y_away_test = y_away[:val_idx], y_away[val_idx:test_idx], y_away[test_idx:]
        
        print(f"\n⏰ Time-based 3-way split:")
        print(f"  Train:      {len(X_train):5} matches (60%)")
        print(f"  Validation: {len(X_val):5} matches (20%)")
        print(f"  Test:       {len(X_test):5} matches (20%)")
        
        # Initialize models
        models_config = [
            ('Random Forest', 
             RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=10,
                                  min_samples_leaf=5, random_state=42, n_jobs=-1)),
            
            ('XGBoost',
             xgb.XGBRegressor(objective='reg:squarederror', n_estimators=300, max_depth=6,
                             learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                             random_state=42, n_jobs=-1)),
            
            ('Gradient Boosting',
             GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                                      subsample=0.8, random_state=42)),
        ]
        
        # Try to add LightGBM if available
        try:
            models_config.append(
                ('LightGBM',
                 lgb.LGBMRegressor(objective='regression', n_estimators=300, max_depth=6,
                                  learning_rate=0.1, random_state=42, verbose=-1))
            )
        except:
            print("\n⚠️  LightGBM not available, skipping...")
        
        print(f"\n🤖 Training {len(models_config)} models...")
        
        # Train all models
        val_predictions_home = []
        val_predictions_away = []
        
        for name, model in models_config:
            print(f"\n{'='*60}")
            print(f"Training: {name}")
            print(f"{'='*60}")
            
            # Train home model
            print("  🏠 Home goals...")
            home_model = model
            home_model.fit(X_train, y_home_train)
            
            # Train away model (clone)
            print("  🚗 Away goals...")
            if name == 'Random Forest':
                away_model = RandomForestRegressor(n_estimators=200, max_depth=15,
                                                  min_samples_split=10, min_samples_leaf=5,
                                                  random_state=42, n_jobs=-1)
            elif name == 'XGBoost':
                away_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=300,
                                             max_depth=6, learning_rate=0.1, subsample=0.8,
                                             colsample_bytree=0.8, random_state=42, n_jobs=-1)
            elif name == 'Gradient Boosting':
                away_model = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                      learning_rate=0.1, subsample=0.8,
                                                      random_state=42)
            else:  # LightGBM
                away_model = lgb.LGBMRegressor(objective='regression', n_estimators=300,
                                              max_depth=6, learning_rate=0.1,
                                              random_state=42, verbose=-1)
            
            away_model.fit(X_train, y_away_train)
            
            # Store models
            self.home_models.append(home_model)
            self.away_models.append(away_model)
            self.model_names.append(name)
            
            # Validate
            home_pred_val = home_model.predict(X_val)
            away_pred_val = away_model.predict(X_val)
            
            val_predictions_home.append(home_pred_val)
            val_predictions_away.append(away_pred_val)
            
            # Individual model performance on validation
            home_mae = mean_absolute_error(y_home_val, home_pred_val)
            away_mae = mean_absolute_error(y_away_val, away_pred_val)
            print(f"  ✓ Validation MAE: Home={home_mae:.3f}, Away={away_mae:.3f}")
        
        # Ensemble predictions (average)
        print("\n" + "="*60)
        print("📊 ENSEMBLE EVALUATION")
        print("="*60)
        
        # Validation set
        ensemble_home_val = np.mean(val_predictions_home, axis=0)
        ensemble_away_val = np.mean(val_predictions_away, axis=0)
        
        val_mae_home = mean_absolute_error(y_home_val, ensemble_home_val)
        val_mae_away = mean_absolute_error(y_away_val, ensemble_away_val)
        
        print(f"\n✅ VALIDATION SET:")
        print(f"  Home MAE: {val_mae_home:.3f}")
        print(f"  Away MAE: {val_mae_away:.3f}")
        print(f"  Overall:  {(val_mae_home + val_mae_away)/2:.3f}")
        
        # Test set
        test_predictions_home = []
        test_predictions_away = []
        
        for home_model, away_model in zip(self.home_models, self.away_models):
            test_predictions_home.append(home_model.predict(X_test))
            test_predictions_away.append(away_model.predict(X_test))
        
        ensemble_home_test = np.mean(test_predictions_home, axis=0)
        ensemble_away_test = np.mean(test_predictions_away, axis=0)
        
        # Round and clip
        ensemble_home_test = np.round(np.maximum(ensemble_home_test, 0))
        ensemble_away_test = np.round(np.maximum(ensemble_away_test, 0))
        
        test_mae_home = mean_absolute_error(y_home_test, ensemble_home_test)
        test_mae_away = mean_absolute_error(y_away_test, ensemble_away_test)
        
        print(f"\n🎯 TEST SET (Final Honest Evaluation):")
        print(f"  Home MAE: {test_mae_home:.3f} goals")
        print(f"  Away MAE: {test_mae_away:.3f} goals")
        print(f"  Overall:  {(test_mae_home + test_mae_away)/2:.3f} goals")
        
        # Outcome accuracy
        y_pred = np.column_stack([ensemble_home_test, ensemble_away_test])
        y_test = np.column_stack([y_home_test, y_away_test])
        
        y_test_outcome = self.goals_to_outcome(y_test)
        y_pred_outcome = self.goals_to_outcome(y_pred)
        
        outcome_accuracy = accuracy_score(y_test_outcome, y_pred_outcome)
        
        print(f"\n🏆 Match Outcome Accuracy: {outcome_accuracy*100:.2f}%")
        
        print("\n📋 Outcome Breakdown:")
        print(classification_report(y_test_outcome, y_pred_outcome,
                                   target_names=['Away Win', 'Draw', 'Home Win'],
                                   zero_division=0))
        
        # Sample predictions
        print("\n📋 Sample Predictions:")
        print("-" * 70)
        print(f"{'Actual':<15} {'Predicted':<15} {'Outcome':<10} {'Error':<10}")
        print("-" * 70)
        
        for i in range(min(15, len(y_test))):
            actual = f"{int(y_test[i, 0])}-{int(y_test[i, 1])}"
            predicted = f"{int(y_pred[i, 0])}-{int(y_pred[i, 1])}"
            outcome = "✓" if y_test_outcome[i] == y_pred_outcome[i] else "✗"
            error = abs(y_test[i, 0] - y_pred[i, 0]) + abs(y_test[i, 1] - y_pred[i, 1])
            print(f"{actual:<15} {predicted:<15} {outcome:<10} {error:.1f}")
        
        # Compare individual vs ensemble
        print("\n" + "="*60)
        print("📊 INDIVIDUAL vs ENSEMBLE COMPARISON")
        print("="*60)
        
        for i, name in enumerate(self.model_names):
            home_pred = np.round(np.maximum(test_predictions_home[i], 0))
            away_pred = np.round(np.maximum(test_predictions_away[i], 0))
            pred = np.column_stack([home_pred, away_pred])
            pred_outcome = self.goals_to_outcome(pred)
            acc = accuracy_score(y_test_outcome, pred_outcome)
            print(f"{name:20s} {acc*100:.2f}%")
        
        print(f"{'ENSEMBLE (Average)':20s} {outcome_accuracy*100:.2f}% ⭐")
        
        # Save ensemble
        joblib.dump({
            'home_models': self.home_models,
            'away_models': self.away_models,
            'model_names': self.model_names,
            'features': self.feature_cols,
            'test_accuracy': outcome_accuracy,
            'test_mae': (test_mae_home + test_mae_away) / 2
        }, self.model_dir / 'ensemble_models.pkl')
        
        print(f"\n✓ Ensemble saved to: {self.model_dir}")
        
        return outcome_accuracy
    
    def goals_to_outcome(self, goals):
        """Convert goals to outcomes"""
        outcomes = []
        for home, away in goals:
            if home > away:
                outcomes.append(2)
            elif home < away:
                outcomes.append(0)
            else:
                outcomes.append(1)
        return np.array(outcomes)


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("🚀 ENSEMBLE MODEL TRAINING")
    print("="*60)
    print("\nEnsembling combines multiple models:")
    print("  • Random Forest")
    print("  • XGBoost")
    print("  • Gradient Boosting")
    print("  • LightGBM (if available)")
    print("\nWhy? Each model makes different mistakes.")
    print("Averaging reduces errors (wisdom of crowds!)")
    
    print("\n" + "="*60)
    input("Press Enter to start training...")
    
    predictor = EnsembleGoalPredictor()
    accuracy = predictor.train_ensemble(test_size=0.2, validation_size=0.1)
    
    print("\n" + "="*60)
    print("✅ ENSEMBLE TRAINING COMPLETE!")
    print("="*60)
    
    if accuracy and accuracy > 0.58:
        print("\n🎉 BREAKTHROUGH! Over 58% accuracy!")
        print("   This is competitive with amateur betting models!")
    elif accuracy and accuracy > 0.57:
        print("\n✨ GOOD! Ensemble helped slightly!")
    else:
        print("\n👍 Ensemble didn't help much, but that's okay!")
    
    print("\nReady for inference system? 🚀")


if __name__ == "__main__":
    main()