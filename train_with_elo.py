"""
Train Goal Prediction Model with ELO Ratings
Combines advanced features + ELO for maximum accuracy
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
import xgboost as xgb
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EloEnhancedPredictor:
    """Goal predictor using advanced features + ELO ratings"""
    
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
    
    def __init__(self, processed_dir='data/processed', model_dir='models/elo_enhanced'):
        self.processed_dir = Path(processed_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.home_model = None
        self.away_model = None
        self.feature_cols = None
        
    def merge_elo_with_advanced(self, league):
        """
        Merge ELO features with advanced features
        
        Strategy:
        1. Load advanced features (rolling stats, H2H, etc.)
        2. Load ELO features
        3. Merge on date, home_team, away_team
        """
        print(f"  📥 Loading {league}...")
        
        # Load advanced features
        advanced_file = self.processed_dir / f"{league}_advanced.csv"
        elo_file = self.processed_dir / f"{league}_with_elo.csv"
        
        if not advanced_file.exists() or not elo_file.exists():
            print(f"    ⚠️  Missing files for {league}")
            return None
        
        df_advanced = pd.read_csv(advanced_file)
        df_elo = pd.read_csv(elo_file)
        
        # Convert dates
        df_advanced['date'] = pd.to_datetime(df_advanced['date'])
        df_elo['date'] = pd.to_datetime(df_elo['date'])
        
        # Merge on match identifiers
        df_elo_subset = df_elo[['date', 'home_team', 'away_team', 
                                 'home_elo_before', 'away_elo_before', 
                                 'elo_diff', 'home_win_prob']]
        
        df_merged = pd.merge(
            df_advanced,
            df_elo_subset,
            on=['date', 'home_team', 'away_team'],
            how='inner'
        )
        
        print(f"    ✓ Merged: {len(df_merged)} matches (advanced={len(df_advanced)}, elo={len(df_elo)})")
        
        return df_merged
    
    def load_enhanced_data(self):
        """Load all leagues with ELO + advanced features"""
        print("\n" + "="*60)
        print("📥 LOADING ENHANCED DATA (Advanced Features + ELO)")
        print("="*60)
        
        all_data = []
        
        for league in self.LEAGUES:
            df = self.merge_elo_with_advanced(league)
            if df is not None:
                all_data.append(df)
        
        if not all_data:
            print("\n❌ No data loaded! Check your files.")
            return None
        
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values('date').reset_index(drop=True)
        
        print(f"\n✓ Total: {len(combined)} matches across {len(all_data)} leagues")
        
        return combined
    
    def prepare_features(self, df):
        """Extract clean features including ELO"""
        all_cols = df.columns.tolist()
        clean_cols = [col for col in all_cols if col not in self.FORBIDDEN_FEATURES]
        
        feature_cols = []
        for col in clean_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_cols.append(col)
        
        # Ensure ELO features are included
        elo_features = ['home_elo_before', 'away_elo_before', 'elo_diff', 'home_win_prob']
        elo_included = [f for f in elo_features if f in feature_cols]
        
        print(f"\n📊 Feature Summary:")
        print(f"  Total features: {len(feature_cols)}")
        print(f"  ELO features:   {len(elo_included)} {elo_included}")
        
        return feature_cols
    
    def train_models(self, model_type='xgboost', test_size=0.2):
        """
        Train home and away goal prediction models
        
        Args:
            model_type: 'xgboost' or 'random_forest'
            test_size: Fraction for testing
        """
        print("\n" + "="*60)
        print(f"⚡ TRAINING {model_type.upper()} MODELS WITH ELO")
        print("="*60)
        
        # Load enhanced data
        df = self.load_enhanced_data()
        if df is None:
            return None
        
        # Check for score columns
        if 'home_score' not in df.columns or 'away_score' not in df.columns:
            print("\n❌ Score columns missing!")
            return None
        
        # Prepare features
        self.feature_cols = self.prepare_features(df)
        
        if len(self.feature_cols) == 0:
            print("\n❌ No valid features!")
            return None
        
        # Extract X and y
        X = df[self.feature_cols].fillna(0).values
        y_home = df['home_score'].values
        y_away = df['away_score'].values
        
        # Time-based split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_home_train, y_home_test = y_home[:split_idx], y_home[split_idx:]
        y_away_train, y_away_test = y_away[:split_idx], y_away[split_idx:]
        
        print(f"\n⏰ Time-based split:")
        print(f"  Train: {len(X_train)} matches")
        print(f"  Test:  {len(X_test)} matches")
        
        # Initialize models
        if model_type == 'xgboost':
            print("\n🏠 Training Home Goals (XGBoost)...")
            self.home_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
            print("🚗 Training Away Goals (XGBoost)...")
            self.away_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        else:
            print("\n🏠 Training Home Goals (Random Forest)...")
            self.home_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            
            print("🚗 Training Away Goals (Random Forest)...")
            self.away_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        
        # Train
        self.home_model.fit(X_train, y_home_train)
        self.away_model.fit(X_train, y_away_train)
        
        # Evaluate
        print("\n" + "="*60)
        print("📊 EVALUATION RESULTS")
        print("="*60)
        
        y_home_pred = np.round(np.maximum(self.home_model.predict(X_test), 0))
        y_away_pred = np.round(np.maximum(self.away_model.predict(X_test), 0))
        
        # Goal metrics
        home_mae = mean_absolute_error(y_home_test, y_home_pred)
        away_mae = mean_absolute_error(y_away_test, y_away_pred)
        overall_mae = (home_mae + away_mae) / 2
        
        print(f"\n🎯 Goal Prediction Accuracy:")
        print(f"  Home Goals MAE: {home_mae:.3f} goals")
        print(f"  Away Goals MAE: {away_mae:.3f} goals")
        print(f"  Overall MAE:    {overall_mae:.3f} goals")
        
        # Outcome metrics
        y_pred = np.column_stack([y_home_pred, y_away_pred])
        y_test = np.column_stack([y_home_test, y_away_test])
        
        y_test_outcome = self.goals_to_outcome(y_test)
        y_pred_outcome = self.goals_to_outcome(y_pred)
        
        outcome_accuracy = accuracy_score(y_test_outcome, y_pred_outcome)
        
        print(f"\n🏆 Match Outcome Accuracy:")
        print(f"  W/D/L Accuracy: {outcome_accuracy*100:.2f}%")
        
        # Detailed breakdown
        print("\n📋 Outcome Breakdown:")
        print(classification_report(
            y_test_outcome, y_pred_outcome,
            target_names=['Away Win', 'Draw', 'Home Win'],
            zero_division=0
        ))
        
        # Sample predictions
        print("\n📋 Sample Predictions (First 15 Test Matches):")
        print("-" * 70)
        print(f"{'Actual':<15} {'Predicted':<15} {'Outcome':<10} {'Error':<10}")
        print("-" * 70)
        
        for i in range(min(15, len(y_test))):
            actual = f"{int(y_test[i, 0])}-{int(y_test[i, 1])}"
            predicted = f"{int(y_pred[i, 0])}-{int(y_pred[i, 1])}"
            outcome = "✓" if y_test_outcome[i] == y_pred_outcome[i] else "✗"
            error = abs(y_test[i, 0] - y_pred[i, 0]) + abs(y_test[i, 1] - y_pred[i, 1])
            print(f"{actual:<15} {predicted:<15} {outcome:<10} {error:.1f}")
        
        # Feature importance
        print("\n" + "="*60)
        print("🎯 TOP 20 IMPORTANT FEATURES")
        print("="*60)
        
        if hasattr(self.home_model, 'feature_importances_'):
            importance = self.home_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print(feature_importance.head(20).to_string(index=False))
            
            # Highlight ELO features
            elo_importance = feature_importance[
                feature_importance['feature'].str.contains('elo|home_win_prob', case=False)
            ]
            
            if len(elo_importance) > 0:
                print("\n⚡ ELO Feature Rankings:")
                for idx, row in elo_importance.iterrows():
                    rank = feature_importance.index.get_loc(idx) + 1
                    print(f"  #{rank:2d} {row['feature']:20s} {row['importance']:.4f}")
        
        # Save models
        joblib.dump({
            'home_model': self.home_model,
            'away_model': self.away_model,
            'features': self.feature_cols,
            'home_mae': home_mae,
            'away_mae': away_mae,
            'outcome_accuracy': outcome_accuracy,
            'model_type': model_type
        }, self.model_dir / f'{model_type}_with_elo.pkl')
        
        print(f"\n✓ Models saved to: {self.model_dir}")
        
        return outcome_accuracy
    
    def goals_to_outcome(self, goals):
        """Convert goals to match outcome"""
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
    print("🚀 TRAINING WITH ELO RATINGS")
    print("="*60)
    print("\nThis combines:")
    print("  ✓ Advanced features (rolling stats, H2H, momentum)")
    print("  ✓ ELO ratings (team strength)")
    print("  ✓ No data leakage")
    
    print("\n📋 Model Options:")
    print("  1. XGBoost (recommended)")
    print("  2. Random Forest")
    print("  3. Both (compare)")
    
    choice = input("\nEnter choice (1-3) or press Enter for option 1: ").strip()
    
    predictor = EloEnhancedPredictor()
    
    if choice == '2':
        accuracy = predictor.train_models('random_forest')
    elif choice == '3':
        print("\n" + "="*60)
        print("TRAINING BOTH MODELS")
        print("="*60)
        
        xgb_acc = predictor.train_models('xgboost')
        rf_acc = predictor.train_models('random_forest')
        
        print("\n" + "="*60)
        print("📊 MODEL COMPARISON")
        print("="*60)
        print(f"XGBoost:        {xgb_acc*100:.2f}%")
        print(f"Random Forest:  {rf_acc*100:.2f}%")
        print(f"\n🏆 Winner: {'XGBoost' if xgb_acc > rf_acc else 'Random Forest'}")
    else:
        accuracy = predictor.train_models('xgboost')
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Build inference system for upcoming matches")
    print("  2. Deploy as web app")
    print("  3. Test on real predictions!")


if __name__ == "__main__":
    main()