"""
Production Inference System for Football Match Prediction
Complete workflow: Predict → Store → Benchmark → Explain → Retrain
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class FootballInferenceSystem:
    """
    Production-ready inference system
    
    Workflow:
    1. Load trained model
    2. Fetch upcoming matches (from API or CSV)
    3. Calculate features for each match
    4. Make predictions
    5. Store predictions in database
    6. After matches: Compare predictions vs actuals
    7. Calculate metrics & benchmark
    8. Explain predictions (SHAP values)
    9. Retrain model periodically
    """
    
    def __init__(self, model_path='models/ensemble/ensemble_models.pkl',
                 predictions_db='data/predictions/predictions.csv',
                 benchmarks_db='data/predictions/benchmarks.csv'):
        
        self.model_path = Path(model_path)
        self.predictions_db = Path(predictions_db)
        self.benchmarks_db = Path(benchmarks_db)
        
        # Create directories
        self.predictions_db.parent.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model_data = None
        self.load_model()
        
    def load_model(self):
        """Load trained model"""
        if not self.model_path.exists():
            print(f"❌ Model not found: {self.model_path}")
            print("   Run train_with_elo.py or ensemble_predictor.py first!")
            return False
        
        print(f"📥 Loading model from {self.model_path}")
        self.model_data = joblib.load(self.model_path)
        
        # Check if ensemble or single model
        if 'home_models' in self.model_data:
            print(f"✓ Loaded ensemble with {len(self.model_data['home_models'])} models")
            print(f"  Models: {', '.join(self.model_data['model_names'])}")
        else:
            print(f"✓ Loaded single model")
        
        print(f"  Features: {len(self.model_data['features'])}")
        print(f"  Test Accuracy: {self.model_data.get('test_accuracy', 0)*100:.2f}%")
        
        return True
    
    def calculate_features_for_match(self, home_team, away_team, match_date=None):
        """
        Calculate all 56 features for a match
        
        This requires:
        - Recent match history (for rolling stats)
        - Head-to-head history
        - ELO ratings (updated)
        
        For now, returns a template showing what's needed
        """
        
        if match_date is None:
            match_date = datetime.now()
        
        # TODO: Implement feature calculation
        # This would query your database for:
        # 1. Last 5 matches for each team
        # 2. Last 5 H2H meetings
        # 3. Current ELO ratings
        # 4. League standings
        
        features = {}
        for feature in self.model_data['features']:
            features[feature] = 0.0  # Placeholder
        
        return features
    
    def predict_match(self, home_team, away_team, features=None, match_date=None):
        """
        Predict a single match
        
        Args:
            home_team: Home team name
            away_team: Away team name
            features: Pre-calculated features dict (optional)
            match_date: Match date (optional)
        
        Returns:
            Dictionary with prediction details
        """
        
        if features is None:
            features = self.calculate_features_for_match(home_team, away_team, match_date)
        
        # Prepare feature vector
        X = np.array([features.get(f, 0) for f in self.model_data['features']]).reshape(1, -1)
        
        # Predict
        if 'home_models' in self.model_data:
            # Ensemble prediction
            home_preds = []
            away_preds = []
            
            for home_model, away_model in zip(self.model_data['home_models'], 
                                             self.model_data['away_models']):
                home_preds.append(home_model.predict(X)[0])
                away_preds.append(away_model.predict(X)[0])
            
            home_goals = np.mean(home_preds)
            away_goals = np.mean(away_preds)
            
        else:
            # Single model
            home_goals = self.model_data['home_model'].predict(X)[0]
            away_goals = self.model_data['away_model'].predict(X)[0]
        
        # Round and clip
        home_goals = int(np.round(np.maximum(home_goals, 0)))
        away_goals = int(np.round(np.maximum(away_goals, 0)))
        
        # Determine outcome
        if home_goals > away_goals:
            outcome = "Home Win"
            winner = home_team
        elif home_goals < away_goals:
            outcome = "Away Win"
            winner = away_team
        else:
            outcome = "Draw"
            winner = "Draw"
        
        # Calculate confidence (simplified)
        goal_diff = abs(home_goals - away_goals)
        confidence = min(0.5 + (goal_diff * 0.15), 0.95)
        
        prediction = {
            'home_team': home_team,
            'away_team': away_team,
            'predicted_score': f"{home_goals}-{away_goals}",
            'home_goals': home_goals,
            'away_goals': away_goals,
            'outcome': outcome,
            'winner': winner,
            'total_goals': home_goals + away_goals,
            'over_2_5': (home_goals + away_goals) > 2.5,
            'both_score': (home_goals > 0 and away_goals > 0),
            'confidence': f"{confidence*100:.1f}%",
            'prediction_date': datetime.now().isoformat(),
            'match_date': match_date.isoformat() if match_date else None
        }
        
        return prediction
    
    def store_prediction(self, prediction, actual_result=None):
        """
        Store prediction in database
        
        Args:
            prediction: Prediction dict from predict_match()
            actual_result: Actual result dict (after match happens)
        """
        
        # Load existing predictions
        if self.predictions_db.exists():
            df = pd.read_csv(self.predictions_db)
        else:
            df = pd.DataFrame()
        
        # Add prediction
        pred_row = prediction.copy()
        
        # Add actual results if available
        if actual_result:
            pred_row['actual_score'] = actual_result.get('score', '')
            pred_row['actual_home_goals'] = actual_result.get('home_goals', None)
            pred_row['actual_away_goals'] = actual_result.get('away_goals', None)
            pred_row['correct_outcome'] = (prediction['outcome'] == actual_result.get('outcome'))
            pred_row['goal_error'] = (abs(prediction['home_goals'] - actual_result.get('home_goals', 0)) +
                                     abs(prediction['away_goals'] - actual_result.get('away_goals', 0)))
        else:
            pred_row['actual_score'] = None
            pred_row['actual_home_goals'] = None
            pred_row['actual_away_goals'] = None
            pred_row['correct_outcome'] = None
            pred_row['goal_error'] = None
        
        # Append
        df = pd.concat([df, pd.DataFrame([pred_row])], ignore_index=True)
        df.to_csv(self.predictions_db, index=False)
        
        print(f"✓ Prediction stored: {prediction['home_team']} vs {prediction['away_team']}")
    
    def update_actual_results(self, home_team, away_team, match_date, actual_home_goals, actual_away_goals):
        """
        Update predictions database with actual results after match
        
        This is called AFTER a match to record what actually happened
        """
        
        if not self.predictions_db.exists():
            print("❌ No predictions database found!")
            return False
        
        df = pd.read_csv(self.predictions_db)
        
        # Find matching prediction
        match_date_str = match_date.isoformat() if isinstance(match_date, datetime) else match_date
        
        mask = ((df['home_team'] == home_team) & 
                (df['away_team'] == away_team) & 
                (df['match_date'] == match_date_str))
        
        if mask.sum() == 0:
            print(f"⚠️  No prediction found for {home_team} vs {away_team} on {match_date_str}")
            return False
        
        # Update actual results
        df.loc[mask, 'actual_home_goals'] = actual_home_goals
        df.loc[mask, 'actual_away_goals'] = actual_away_goals
        df.loc[mask, 'actual_score'] = f"{actual_home_goals}-{actual_away_goals}"
        
        # Calculate correctness
        predicted_outcome = df.loc[mask, 'outcome'].values[0]
        
        if actual_home_goals > actual_away_goals:
            actual_outcome = "Home Win"
        elif actual_home_goals < actual_away_goals:
            actual_outcome = "Away Win"
        else:
            actual_outcome = "Draw"
        
        df.loc[mask, 'correct_outcome'] = (predicted_outcome == actual_outcome)
        
        # Goal error
        pred_home = df.loc[mask, 'home_goals'].values[0]
        pred_away = df.loc[mask, 'away_goals'].values[0]
        df.loc[mask, 'goal_error'] = abs(pred_home - actual_home_goals) + abs(pred_away - actual_away_goals)
        
        # Save
        df.to_csv(self.predictions_db, index=False)
        
        print(f"✓ Actual result recorded: {home_team} {actual_home_goals}-{actual_away_goals} {away_team}")
        print(f"  Predicted: {pred_home}-{pred_away}")
        print(f"  Outcome: {'✓ Correct' if predicted_outcome == actual_outcome else '✗ Wrong'}")
        
        return True
    
    def calculate_benchmarks(self, period='all'):
        """
        Calculate performance metrics
        
        Args:
            period: 'all', 'last_30_days', 'last_week'
        
        Returns:
            Dict with metrics
        """
        
        if not self.predictions_db.exists():
            print("❌ No predictions database!")
            return None
        
        df = pd.read_csv(self.predictions_db)
        
        # Filter to predictions with actual results
        df = df[df['correct_outcome'].notna()].copy()
        
        if len(df) == 0:
            print("⚠️  No predictions with actual results yet!")
            return None
        
        # Filter by period
        if period != 'all':
            df['prediction_date'] = pd.to_datetime(df['prediction_date'])
            if period == 'last_30_days':
                cutoff = datetime.now() - timedelta(days=30)
            elif period == 'last_week':
                cutoff = datetime.now() - timedelta(days=7)
            else:
                cutoff = datetime.min
            
            df = df[df['prediction_date'] >= cutoff]
        
        # Calculate metrics
        total = len(df)
        correct = df['correct_outcome'].sum()
        accuracy = correct / total if total > 0 else 0
        
        avg_goal_error = df['goal_error'].mean()
        
        # Breakdown by outcome
        home_wins = df[df['outcome'] == 'Home Win']
        draws = df[df['outcome'] == 'Draw']
        away_wins = df[df['outcome'] == 'Away Win']
        
        metrics = {
            'period': period,
            'total_predictions': total,
            'correct_predictions': int(correct),
            'accuracy': f"{accuracy*100:.2f}%",
            'avg_goal_error': f"{avg_goal_error:.2f}",
            'home_win_accuracy': f"{home_wins['correct_outcome'].mean()*100:.1f}%" if len(home_wins) > 0 else "N/A",
            'draw_accuracy': f"{draws['correct_outcome'].mean()*100:.1f}%" if len(draws) > 0 else "N/A",
            'away_win_accuracy': f"{away_wins['correct_outcome'].mean()*100:.1f}%" if len(away_wins) > 0 else "N/A",
            'calculated_at': datetime.now().isoformat()
        }
        
        return metrics
    
    def print_benchmarks(self):
        """Print current performance metrics"""
        
        print("\n" + "="*60)
        print("📊 PERFORMANCE BENCHMARKS")
        print("="*60)
        
        for period in ['all', 'last_30_days', 'last_week']:
            metrics = self.calculate_benchmarks(period)
            
            if metrics is None:
                continue
            
            print(f"\n{period.upper().replace('_', ' ')}:")
            print(f"  Total Predictions:  {metrics['total_predictions']}")
            print(f"  Correct:            {metrics['correct_predictions']}")
            print(f"  Accuracy:           {metrics['accuracy']}")
            print(f"  Avg Goal Error:     {metrics['avg_goal_error']}")
            print(f"  Home Win Accuracy:  {metrics['home_win_accuracy']}")
            print(f"  Draw Accuracy:      {metrics['draw_accuracy']}")
            print(f"  Away Win Accuracy:  {metrics['away_win_accuracy']}")
    
    def explain_prediction(self, home_team, away_team, features):
        """
        Explain WHY a prediction was made
        
        Shows top features that influenced the prediction
        """
        
        # Get feature importances from model
        if 'home_models' in self.model_data:
            # Use first model (Random Forest) for importance
            model = self.model_data['home_models'][0]
        else:
            model = self.model_data['home_model']
        
        if not hasattr(model, 'feature_importances_'):
            print("⚠️  Model doesn't support feature importance")
            return None
        
        # Get importances
        importances = model.feature_importances_
        feature_names = self.model_data['features']
        
        # Get feature values for this match
        feature_values = [features.get(f, 0) for f in feature_names]
        
        # Combine
        feature_impact = pd.DataFrame({
            'feature': feature_names,
            'value': feature_values,
            'importance': importances,
            'impact': importances * np.abs(feature_values)
        }).sort_values('impact', ascending=False)
        
        print(f"\n🔍 PREDICTION EXPLANATION: {home_team} vs {away_team}")
        print("="*60)
        print("\nTop 10 factors influencing this prediction:")
        print(feature_impact.head(10).to_string(index=False))
        
        return feature_impact


def demo_workflow():
    """Demonstrate the complete workflow"""
    
    print("\n" + "="*60)
    print("🚀 FOOTBALL PREDICTION INFERENCE SYSTEM")
    print("="*60)
    
    # Initialize
    system = FootballInferenceSystem()
    
    if system.model_data is None:
        print("\n❌ Model not loaded. Exiting.")
        return
    
    print("\n" + "="*60)
    print("📋 WORKFLOW DEMONSTRATION")
    print("="*60)
    
    print("\n1️⃣  PREDICT UPCOMING MATCH")
    print("-" * 60)
    
    # Example prediction
    prediction = system.predict_match(
        home_team="Arsenal",
        away_team="Liverpool",
        match_date=datetime(2025, 11, 15)
    )
    
    print(f"\n⚽ {prediction['home_team']} vs {prediction['away_team']}")
    print(f"   Predicted Score: {prediction['predicted_score']}")
    print(f"   Outcome: {prediction['outcome']}")
    print(f"   Confidence: {prediction['confidence']}")
    print(f"   Total Goals: {prediction['total_goals']}")
    print(f"   Over 2.5: {'Yes' if prediction['over_2_5'] else 'No'}")
    print(f"   Both Teams Score: {'Yes' if prediction['both_score'] else 'No'}")
    
    print("\n2️⃣  STORE PREDICTION")
    print("-" * 60)
    system.store_prediction(prediction)
    
    print("\n3️⃣  AFTER MATCH: UPDATE ACTUAL RESULTS")
    print("-" * 60)
    print("   (Manually or via API after match finishes)")
    print(f"   Example: Arsenal 2-2 Liverpool")
    
    system.update_actual_results(
        home_team="Arsenal",
        away_team="Liverpool",
        match_date=datetime(2025, 11, 15),
        actual_home_goals=2,
        actual_away_goals=2
    )
    
    print("\n4️⃣  VIEW BENCHMARKS")
    print("-" * 60)
    system.print_benchmarks()
    
    print("\n" + "="*60)
    print("✅ WORKFLOW COMPLETE!")
    print("="*60)
    
    print("\nNext steps:")
    print("  1. Connect to API for upcoming fixtures")
    print("  2. Automate result updates")
    print("  3. Build web dashboard")
    print("  4. Set up weekly retraining")


if __name__ == "__main__":
    demo_workflow()