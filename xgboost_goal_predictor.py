"""
XGBoost Goal Predictor - Enhanced version
Uses XGBoost (often 5-10% better than Random Forest)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import xgboost as xgb
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class XGBoostGoalPredictor:
    """XGBoost-based goal predictor"""
    
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
    
    def __init__(self, processed_dir='data/processed', model_dir='models/xgboost_predictor'):
        self.processed_dir = Path(processed_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.home_model = None
        self.away_model = None
        self.feature_cols = None
        
    def load_clean_data(self):
        """Load all league data"""
        print("\n" + "="*60)
        print("📥 LOADING DATA")
        print("="*60)
        
        all_data = []
        
        for league in self.LEAGUES:
            file_path = self.processed_dir / f"{league}_advanced.csv"
            if not file_path.exists():
                continue
            
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            print(f"✓ Loaded {league}: {len(df)} matches")
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
    
    def train_xgboost_models(self, test_size=0.2):
        """
        Train separate XGBoost models for home and away goals
        """
        print("\n" + "="*60)
        print("⚡ TRAINING XGBOOST GOAL PREDICTORS")
        print("="*60)
        
        # Load data
        df = self.load_clean_data()
        
        if 'home_score' not in df.columns or 'away_score' not in df.columns:
            print("❌ Error: Score columns missing!")
            return None
        
        # Prepare features
        self.feature_cols = self.prepare_features(df)
        print(f"\n📊 Using {len(self.feature_cols)} features")
        
        # Extract X and y
        X = df[self.feature_cols].fillna(0).values
        y_home = df['home_score'].values
        y_away = df['away_score'].values
        
        # Time-based split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_home_train, y_home_test = y_home[:split_idx], y_home[split_idx:]
        y_away_train, y_away_test = y_away[:split_idx], y_away[split_idx:]
        
        print(f"⏰ Time-based split: Train={len(X_train)}, Test={len(X_test)}")
        
        # Train Home Goals Model
        print("\n🏠 Training Home Goals Model...")
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
        self.home_model.fit(X_train, y_home_train, verbose=False)
        
        # Train Away Goals Model
        print("🚗 Training Away Goals Model...")
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
        self.away_model.fit(X_train, y_away_train, verbose=False)
        
        # Evaluate
        print("\n" + "="*60)
        print("📊 EVALUATION")
        print("="*60)
        
        # Predict
        y_home_pred = np.round(np.maximum(self.home_model.predict(X_test), 0))
        y_away_pred = np.round(np.maximum(self.away_model.predict(X_test), 0))
        
        # Metrics
        home_mae = mean_absolute_error(y_home_test, y_home_pred)
        away_mae = mean_absolute_error(y_away_test, y_away_pred)
        overall_mae = (home_mae + away_mae) / 2
        
        print(f"\n🎯 Goal Prediction Accuracy:")
        print(f"  Home Goals MAE: {home_mae:.3f} goals")
        print(f"  Away Goals MAE: {away_mae:.3f} goals")
        print(f"  Overall MAE:    {overall_mae:.3f} goals")
        
        # Derive match outcomes
        y_pred = np.column_stack([y_home_pred, y_away_pred])
        y_test = np.column_stack([y_home_test, y_away_test])
        
        y_test_outcome = self.goals_to_outcome(y_test)
        y_pred_outcome = self.goals_to_outcome(y_pred)
        
        outcome_accuracy = accuracy_score(y_test_outcome, y_pred_outcome)
        
        print(f"\n🏆 Match Outcome Accuracy:")
        print(f"  W/D/L Accuracy: {outcome_accuracy*100:.2f}%")
        
        # Breakdown by outcome
        from sklearn.metrics import classification_report
        print("\n📋 Outcome Breakdown:")
        print(classification_report(
            y_test_outcome, y_pred_outcome,
            target_names=['Away Win', 'Draw', 'Home Win'],
            zero_division=0
        ))
        
        # Sample predictions
        print("\n📋 Sample Predictions:")
        print("-" * 60)
        print(f"{'Actual':<15} {'Predicted':<15} {'Outcome':<10}")
        print("-" * 60)
        
        for i in range(min(15, len(y_test))):
            actual = f"{int(y_test[i, 0])}-{int(y_test[i, 1])}"
            predicted = f"{int(y_pred[i, 0])}-{int(y_pred[i, 1])}"
            outcome = "✓" if y_test_outcome[i] == y_pred_outcome[i] else "✗"
            print(f"{actual:<15} {predicted:<15} {outcome:<10}")
        
        # Feature importance
        print("\n" + "="*60)
        print("🎯 TOP 15 IMPORTANT FEATURES")
        print("="*60)
        
        # Get feature importance from home model
        importance = self.home_model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(feature_importance.head(15).to_string(index=False))
        
        # Save models
        joblib.dump({
            'home_model': self.home_model,
            'away_model': self.away_model,
            'features': self.feature_cols,
            'home_mae': home_mae,
            'away_mae': away_mae,
            'outcome_accuracy': outcome_accuracy
        }, self.model_dir / 'xgboost_models.pkl')
        
        print(f"\n✓ Models saved to: {self.model_dir}")
        
        return outcome_accuracy
    
    def goals_to_outcome(self, goals):
        """Convert goals to match outcome"""
        outcomes = []
        for home, away in goals:
            if home > away:
                outcomes.append(2)  # Home win
            elif home < away:
                outcomes.append(0)  # Away win
            else:
                outcomes.append(1)  # Draw
        return np.array(outcomes)
    
    def predict_match(self, features_dict):
        """Predict goals for a new match"""
        if self.home_model is None or self.away_model is None:
            raise ValueError("Models not trained!")
        
        X = np.array([features_dict.get(f, 0) for f in self.feature_cols]).reshape(1, -1)
        
        home_goals = int(np.round(np.maximum(self.home_model.predict(X)[0], 0)))
        away_goals = int(np.round(np.maximum(self.away_model.predict(X)[0], 0)))
        
        if home_goals > away_goals:
            outcome = "Home Win"
        elif home_goals < away_goals:
            outcome = "Away Win"
        else:
            outcome = "Draw"
        
        return {
            'home_goals': home_goals,
            'away_goals': away_goals,
            'score': f"{home_goals}-{away_goals}",
            'outcome': outcome,
            'total_goals': home_goals + away_goals,
            'over_2_5': home_goals + away_goals > 2.5,
            'both_score': home_goals > 0 and away_goals > 0
        }


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("⚡ XGBOOST GOAL PREDICTOR")
    print("="*60)
    print("\nXGBoost is often 5-10% better than Random Forest!")
    print("\nTraining separate models for:")
    print("  • Home Goals")
    print("  • Away Goals")
    
    input("\nPress Enter to start training...")
    
    predictor = XGBoostGoalPredictor()
    accuracy = predictor.train_xgboost_models(test_size=0.2)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    
    if accuracy:
        if accuracy > 0.60:
            print("\n🔥 EXCELLENT! Over 60% accuracy!")
        elif accuracy > 0.58:
            print("\n✨ GREAT! Near 60% accuracy!")
        else:
            print("\n👍 Good start! Can improve further.")
    
    print("\nNext steps:")
    print("  1. Compare with Random Forest results")
    print("  2. Build inference system")
    print("  3. Deploy as web app")


if __name__ == "__main__":
    main()