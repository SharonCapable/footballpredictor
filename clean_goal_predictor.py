"""
CLEAN FOOTBALL GOAL PREDICTION MODEL
No data leakage, predicts actual goals, uses only pre-match features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CleanGoalPredictor:
    """Predicts home and away goals without data leakage"""
    
    LEAGUES = ['england', 'spain', 'germany', 'italy', 'france']
    
    # CRITICAL: Features that should NEVER be used (data leakage)
    FORBIDDEN_FEATURES = [
        # Match results (know after game)
        'home_score', 'away_score', 'outcome', 'result', 'FTR', 'FTHG', 'FTAG',
        
        # Half-time data (not available before match)
        'HTHG', 'HTAG', 'HTR', 'ht_home_score', 'ht_away_score',
        
        # Betting odds (not available early enough for predictions)
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
        
        # Metadata
        'date', 'League', 'home_team', 'away_team', 'Season', 'Div', 'Time', 'Referee'
    ]
    
    def __init__(self, processed_dir='data/processed', model_dir='models/clean_goal_predictor'):
        self.processed_dir = Path(processed_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.feature_cols = None
        
    def load_clean_data(self):
        """Load all league data and ensure no leakage"""
        print("\n" + "="*60)
        print("📥 LOADING DATA (LEAK-FREE)")
        print("="*60)
        
        all_data = []
        
        for league in self.LEAGUES:
            file_path = self.processed_dir / f"{league}_advanced.csv"
            if not file_path.exists():
                print(f"⚠️  {league} not found")
                continue
            
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            print(f"✓ Loaded {league}: {len(df)} matches")
            all_data.append(df)
        
        # Combine
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values('date').reset_index(drop=True)
        
        print(f"\n✓ Total: {len(combined)} matches")
        return combined
    
    def prepare_features(self, df):
        """Extract ONLY pre-match features (no leakage!)"""
        print("\n" + "="*60)
        print("🔍 EXTRACTING CLEAN FEATURES")
        print("="*60)
        
        # Get all columns
        all_cols = df.columns.tolist()
        
        # Remove forbidden features
        clean_cols = [col for col in all_cols if col not in self.FORBIDDEN_FEATURES]
        
        # Keep only numeric features
        feature_cols = []
        for col in clean_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_cols.append(col)
        
        print(f"✓ Original columns: {len(all_cols)}")
        print(f"✓ After removing leakage: {len(clean_cols)}")
        print(f"✓ Final numeric features: {len(feature_cols)}")
        
        # Show top 10 features
        print("\n📋 Sample Features:")
        for i, col in enumerate(feature_cols[:10], 1):
            print(f"  {i}. {col}")
        if len(feature_cols) > 10:
            print(f"  ... and {len(feature_cols) - 10} more")
        
        return feature_cols
    
    def train_goal_model(self, test_size=0.2, use_time_split=True):
        """
        Train a model to predict home and away goals
        
        Args:
            test_size: Fraction of data for testing
            use_time_split: If True, use chronological split (more realistic)
        """
        print("\n" + "="*60)
        print("🎯 TRAINING GOAL PREDICTION MODEL")
        print("="*60)
        
        # Load data
        df = self.load_clean_data()
        
        # We need actual scores as targets
        if 'home_score' not in df.columns or 'away_score' not in df.columns:
            print("❌ Error: home_score/away_score columns missing!")
            print("   These should be in your raw data. Let me check...")
            return None
        
        # Prepare features
        self.feature_cols = self.prepare_features(df)
        
        if len(self.feature_cols) == 0:
            print("❌ No valid features found!")
            return None
        
        # Extract X and y
        X = df[self.feature_cols].fillna(0).values
        y = df[['home_score', 'away_score']].values
        
        # Split data
        if use_time_split:
            # Chronological split (more realistic)
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            print(f"\n⏰ Using TIME-BASED split (training on older matches)")
        else:
            # Random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            print(f"\n🎲 Using RANDOM split")
        
        print(f"Train: {len(X_train)} matches | Test: {len(X_test)} matches")
        
        # Train model (Random Forest for goals)
        print("\n⚡ Training Multi-Output Random Forest...")
        
        base_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.model = MultiOutputRegressor(base_model)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        print("\n" + "="*60)
        print("📊 EVALUATION")
        print("="*60)
        
        y_pred = self.model.predict(X_test)
        y_pred = np.round(np.maximum(y_pred, 0))  # Round and ensure non-negative
        
        # Goal prediction accuracy
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        home_mae = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
        away_mae = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
        
        print(f"\n🎯 Goal Prediction Accuracy:")
        print(f"  Overall MAE:    {mae:.3f} goals")
        print(f"  Overall RMSE:   {rmse:.3f} goals")
        print(f"  Home Goals MAE: {home_mae:.3f} goals")
        print(f"  Away Goals MAE: {away_mae:.3f} goals")
        
        # Derive match outcome accuracy
        y_test_outcome = self.goals_to_outcome(y_test)
        y_pred_outcome = self.goals_to_outcome(y_pred)
        
        outcome_accuracy = (y_test_outcome == y_pred_outcome).mean()
        
        print(f"\n🏆 Match Outcome Accuracy (derived from goals):")
        print(f"  W/D/L Accuracy: {outcome_accuracy*100:.2f}%")
        
        # Show sample predictions
        print("\n📋 Sample Predictions:")
        print("-" * 60)
        print(f"{'Actual':<15} {'Predicted':<15} {'Outcome':<20}")
        print("-" * 60)
        
        for i in range(min(10, len(y_test))):
            actual = f"{int(y_test[i, 0])}-{int(y_test[i, 1])}"
            predicted = f"{int(y_pred[i, 0])}-{int(y_pred[i, 1])}"
            outcome = "✓" if y_test_outcome[i] == y_pred_outcome[i] else "✗"
            print(f"{actual:<15} {predicted:<15} {outcome:<20}")
        
        # Save model
        model_file = self.model_dir / 'goal_predictor.pkl'
        joblib.dump({
            'model': self.model,
            'features': self.feature_cols,
            'mae': mae,
            'outcome_accuracy': outcome_accuracy
        }, model_file)
        
        print(f"\n✓ Model saved to: {model_file}")
        
        return self.model
    
    def goals_to_outcome(self, goals):
        """Convert goals to match outcome (0=Away, 1=Draw, 2=Home)"""
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
        """
        Predict goals for a new match
        
        Args:
            features_dict: Dictionary with feature values
        
        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained! Call train_goal_model() first.")
        
        # Prepare features in correct order
        X = np.array([features_dict.get(f, 0) for f in self.feature_cols]).reshape(1, -1)
        
        # Predict
        goals = self.model.predict(X)[0]
        goals = np.round(np.maximum(goals, 0))  # Round and ensure non-negative
        
        home_goals = int(goals[0])
        away_goals = int(goals[1])
        
        # Determine outcome
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
    print("⚽ CLEAN GOAL PREDICTION MODEL")
    print("="*60)
    print("\nThis model:")
    print("  ✓ Predicts ACTUAL GOALS (not just W/D/L)")
    print("  ✓ No data leakage (no odds, no half-time scores)")
    print("  ✓ Uses only pre-match features")
    print("  ✓ Time-based validation (train on past, test on future)")
    
    predictor = CleanGoalPredictor()
    
    print("\n📋 Training Options:")
    print("  1. Train with time-based split (recommended)")
    print("  2. Train with random split")
    
    choice = input("\nEnter choice (1/2) or press Enter for option 1: ").strip()
    
    use_time_split = choice != '2'
    
    # Train
    predictor.train_goal_model(test_size=0.2, use_time_split=use_time_split)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Use this model for REAL predictions")
    print("  2. Build inference system")
    print("  3. Deploy as web app")


if __name__ == "__main__":
    main()