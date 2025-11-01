"""
Complete Multi-League Football Prediction System v2
Trains league-specific and universal models with advanced features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class FootballPredictionSystem:
    """Complete prediction system with multiple model types"""
    
    LEAGUES = ['england', 'spain', 'germany', 'italy', 'france']
    
    def __init__(self, processed_dir='data/processed', model_dir='models/complete_system_v2'):
        self.processed_dir = Path(processed_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.league_models = {}
        self.universal_model = None
        self.best_universal_model = None
        self.model_comparison = []
        
    def load_league_data(self, league, use_advanced=True):
        """Load processed data for a specific league"""
        # Try advanced features first
        if use_advanced:
            file_path = self.processed_dir / f"{league}_advanced.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                print(f"✓ Loaded {league} (advanced): {len(df)} matches")
                return df
        
        # Fall back to basic processed
        file_path = self.processed_dir / f"{league}_processed.csv"
        if not file_path.exists():
            print(f"⚠️  {league} data not found")
            return None
        
        df = pd.read_csv(file_path)
        print(f"✓ Loaded {league}: {len(df)} matches")
        return df
    
    def prepare_features(self, df):
        """Prepare feature matrix X and target y"""
        # Identify metadata columns to exclude
        exclude_cols = [
            'outcome', 'League', 'date', 'home_team', 'away_team', 
            'home_score', 'away_score', 'result', 'Season',
            'Div', 'Time', 'Referee', 'FTR', 'HTR',  # Match metadata
            'League_Name', 'B365H', 'B365D', 'B365A',  # Betting odds
            'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA',  # More odds
            'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA',  # More odds
            'VCH', 'VCD', 'VCA', 'MaxH', 'MaxD', 'MaxA',  # More odds
            'AvgH', 'AvgD', 'AvgA'  # Average odds
        ]
        
        # Get all column names
        all_cols = df.columns.tolist()
        
        # Select only numeric columns (exclude metadata and strings)
        feature_cols = []
        for col in all_cols:
            if col not in exclude_cols:
                # Check if column is numeric
                if pd.api.types.is_numeric_dtype(df[col]):
                    feature_cols.append(col)
                else:
                    # Try converting to numeric
                    try:
                        numeric_values = pd.to_numeric(df[col], errors='coerce')
                        # Only include if at least 50% of values are valid numbers
                        if numeric_values.notna().sum() / len(df) > 0.5:
                            df[col] = numeric_values
                            feature_cols.append(col)
                    except:
                        pass
        
        print(f"  📊 Selected {len(feature_cols)} numeric features")
        
        # Handle missing values
        X = df[feature_cols].fillna(0)  # Fill NaN with 0
        y = df['outcome'].values
        
        return X.values, y, feature_cols
    
    def train_league_model(self, league, model_type='random_forest'):
        """Train a model for a specific league"""
        print(f"\n{'='*60}")
        print(f"🏆 TRAINING: {league.upper()}")
        print(f"{'='*60}")
        
        # Load data
        df = self.load_league_data(league)
        if df is None:
            return None
        
        # Prepare features
        X, y, feature_cols = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Train: {len(X_train)} | Test: {len(X_test)}")
        
        # Initialize model
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            model = LogisticRegression(max_iter=1000, random_state=42)
        
        # Train
        print(f"⚡ Training {model_type}...")
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✓ Accuracy: {accuracy*100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 5 Important Features:")
            print(feature_importance.head().to_string(index=False))
        
        # Save model
        model_file = self.model_dir / f"{league}_{model_type}.pkl"
        joblib.dump({
            'model': model,
            'features': feature_cols,
            'accuracy': accuracy,
            'league': league
        }, model_file)
        
        print(f"✓ Saved to {model_file}")
        
        # Store results
        self.league_models[league] = {
            'model': model,
            'accuracy': accuracy,
            'features': feature_cols
        }
        
        return model, accuracy
    
    def train_all_league_models(self, model_type='random_forest'):
        """Train models for all leagues"""
        print("\n" + "="*60)
        print(f"🌍 TRAINING ALL LEAGUE-SPECIFIC MODELS ({model_type})")
        print("="*60)
        
        results = []
        
        for league in self.LEAGUES:
            model, accuracy = self.train_league_model(league, model_type)
            if model is not None:
                results.append({
                    'League': league.title(),
                    'Accuracy': f"{accuracy*100:.2f}%",
                    'Model': model_type
                })
        
        # Summary
        print("\n" + "="*60)
        print("📊 LEAGUE MODEL SUMMARY")
        print("="*60)
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        
        return results
    
    def train_universal_model(self, model_type='random_forest'):
        """Train a single model on all leagues combined"""
        print("\n" + "="*60)
        print(f"🌍 TRAINING UNIVERSAL MODEL ({model_type})")
        print("="*60)
        
        # Load all league data
        all_data = []
        for league in self.LEAGUES:
            df = self.load_league_data(league)
            if df is not None:
                all_data.append(df)
        
        if not all_data:
            print("❌ No data available for training")
            return None
        
        # Combine datasets
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"✓ Combined dataset: {len(combined_df)} matches across {len(all_data)} leagues")
        
        # Prepare features
        X, y, feature_cols = self.prepare_features(combined_df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Train: {len(X_train)} | Test: {len(X_test)}")
        
        # Initialize model
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            model = LogisticRegression(max_iter=1000, random_state=42)
        
        # Train
        print(f"⚡ Training {model_type}...")
        model.fit(X_train, y_train)
        
        # Evaluate overall
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✓ Overall Accuracy: {accuracy*100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Evaluate per league
        print("\n" + "-"*60)
        print("Per-League Performance:")
        print("-"*60)
        
        test_df = combined_df.iloc[X_test.index] if hasattr(X_test, 'index') else combined_df.sample(len(X_test))
        
        for league in self.LEAGUES:
            league_mask = test_df['League'] == league
            if league_mask.sum() > 0:
                league_indices = test_df[league_mask].index
                y_league_true = y_test[league_mask]
                y_league_pred = y_pred[league_mask]
                league_acc = accuracy_score(y_league_true, y_league_pred)
                print(f"{league.upper():15}: {league_acc*100:.2f}% ({league_mask.sum()} matches)")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n" + "="*60)
            print("🎯 FEATURE IMPORTANCE")
            print("="*60)
            print(feature_importance.to_string(index=False))
        
        # Save model
        model_file = self.model_dir / f"universal_{model_type}.pkl"
        joblib.dump({
            'model': model,
            'features': feature_cols,
            'accuracy': accuracy,
            'leagues': self.LEAGUES
        }, model_file)
        
        print(f"\n✓ Saved to {model_file}")
        
        self.universal_model = {
            'model': model,
            'accuracy': accuracy,
            'features': feature_cols,
            'type': model_type
        }
        
        return model, accuracy
    
    def compare_models(self):
        """Compare different model types"""
        print("\n" + "="*60)
        print("🔬 MODEL COMPARISON")
        print("="*60)
        
        model_types = ['logistic_regression', 'random_forest', 'gradient_boosting']
        results = []
        
        for model_type in model_types:
            print(f"\n{'='*60}")
            print(f"Testing: {model_type.replace('_', ' ').title()}")
            print(f"{'='*60}")
            
            model, accuracy = self.train_universal_model(model_type)
            
            if model is not None:
                results.append({
                    'Model': model_type.replace('_', ' ').title(),
                    'Accuracy': f"{accuracy*100:.2f}%",
                    'Score': accuracy
                })
        
        # Display comparison
        print("\n" + "="*60)
        print("📊 FINAL COMPARISON")
        print("="*60)
        results_df = pd.DataFrame(results).sort_values('Score', ascending=False)
        print(results_df[['Model', 'Accuracy']].to_string(index=False))
        
        # Save best model
        best_model = results_df.iloc[0]
        print(f"\n🏆 Best Model: {best_model['Model']} ({best_model['Accuracy']})")
        
        return results_df
    
    def predict_match(self, home_team, away_team, league, features):
        """Predict outcome for a new match"""
        # Load appropriate model
        if league in self.league_models:
            model_info = self.league_models[league]
        elif self.universal_model:
            model_info = self.universal_model
        else:
            raise ValueError("No trained model available")
        
        model = model_info['model']
        feature_names = model_info['features']
        
        # Prepare features (in correct order)
        X = np.array([features[f] for f in feature_names]).reshape(1, -1)
        
        # Predict
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        outcome_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
        
        return {
            'prediction': outcome_map[prediction],
            'probabilities': {
                'Away Win': f"{probabilities[0]*100:.1f}%",
                'Draw': f"{probabilities[1]*100:.1f}%",
                'Home Win': f"{probabilities[2]*100:.1f}%"
            },
            'confidence': f"{probabilities[prediction]*100:.1f}%"
        }


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("🚀 FOOTBALL PREDICTION SYSTEM v2.0")
    print("="*60)
    
    system = FootballPredictionSystem()
    
    print("\n📋 Training Options:")
    print("  1. Train league-specific models only")
    print("  2. Train universal model only")
    print("  3. Train both and compare")
    print("  4. Compare all model types (Logistic, RF, GB)")
    
    choice = input("\nEnter choice (1-4) or press Enter for option 4: ").strip()
    
    if choice == '1':
        system.train_all_league_models('random_forest')
    elif choice == '2':
        system.train_universal_model('random_forest')
    elif choice == '3':
        system.train_all_league_models('random_forest')
        system.train_universal_model('random_forest')
    else:
        # Default: Full comparison
        system.compare_models()
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print("\nModels saved in: models/complete_system_v2/")
    print("\nNext steps:")
    print("  1. Build inference system for future predictions")
    print("  2. Try different hyperparameters")
    print("  3. Add more advanced features (xG, player stats)")


if __name__ == "__main__":
    main()