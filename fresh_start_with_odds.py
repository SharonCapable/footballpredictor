"""
Fresh Start: Build Prediction Model Directly from Odds CSV
This bypasses all database/merge issues
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class OddsBasedPredictor:
    def __init__(self):
        self.df = None
        self.model = None
    
    def load_and_prepare_data(self, league='portugal'):
        """
        Load odds CSV and engineer features directly from it
        
        Args:
            league: 'portugal' or 'epl'
        """
        print("="*60)
        print(f"LOADING {league.upper()} DATA")
        print("="*60)
        
        if league == 'epl':
            df = pd.read_csv('data/odds/epl_with_odds.csv')
        else:
            df = pd.read_csv('data/odds/portugal_with_odds.csv')
        
        print(f"\n✓ Loaded {len(df)} matches")
        
        # Convert date
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        
        # Create target variables
        df['total_goals'] = df['FTHG'] + df['FTAG']
        df['over_2_5'] = (df['total_goals'] > 2.5).astype(int)
        df['btts'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)
        
        # Outcome
        def get_outcome(row):
            if row['FTHG'] > row['FTAG']:
                return 'H'
            elif row['FTHG'] < row['FTAG']:
                return 'A'
            else:
                return 'D'
        
        df['outcome'] = df.apply(get_outcome, axis=1)
        
        self.df = df.sort_values('Date').reset_index(drop=True)
        
        print(f"Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
        print(f"\nOver 2.5 distribution:")
        print(self.df['over_2_5'].value_counts())
        
        return self.df
    
    def engineer_features(self):
        """
        Engineer features from historical data + odds
        """
        print("\n" + "="*60)
        print("ENGINEERING FEATURES")
        print("="*60)
        
        features_list = []
        
        for idx, match in self.df.iterrows():
            if idx % 100 == 0:
                print(f"  Processing {idx}/{len(self.df)}...")
            
            match_date = match['Date']
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            
            # Get previous matches (last 10)
            home_prev = self.df[
                ((self.df['HomeTeam'] == home_team) | (self.df['AwayTeam'] == home_team)) &
                (self.df['Date'] < match_date)
            ].tail(10)
            
            away_prev = self.df[
                ((self.df['HomeTeam'] == away_team) | (self.df['AwayTeam'] == away_team)) &
                (self.df['Date'] < match_date)
            ].tail(10)
            
            # Skip if insufficient history
            if len(home_prev) < 3 or len(away_prev) < 3:
                continue
            
            # Calculate home team goals
            home_goals_for = []
            home_goals_against = []
            
            for _, prev in home_prev.iterrows():
                if prev['HomeTeam'] == home_team:
                    home_goals_for.append(prev['FTHG'])
                    home_goals_against.append(prev['FTAG'])
                else:
                    home_goals_for.append(prev['FTAG'])
                    home_goals_against.append(prev['FTHG'])
            
            # Calculate away team goals
            away_goals_for = []
            away_goals_against = []
            
            for _, prev in away_prev.iterrows():
                if prev['AwayTeam'] == away_team:
                    away_goals_for.append(prev['FTAG'])
                    away_goals_against.append(prev['FTHG'])
                else:
                    away_goals_for.append(prev['FTHG'])
                    away_goals_against.append(prev['FTAG'])
            
            features = {
                'match_idx': idx,
                'date': match_date,
                'home_team': home_team,
                'away_team': away_team,
                
                # Targets
                'total_goals': match['total_goals'],
                'over_2_5': match['over_2_5'],
                'btts': match['btts'],
                'outcome': match['outcome'],
                
                # Historical features
                'home_avg_gf_l10': np.mean(home_goals_for),
                'home_avg_ga_l10': np.mean(home_goals_against),
                'home_avg_gf_l5': np.mean(home_goals_for[-5:]),
                'home_avg_ga_l5': np.mean(home_goals_against[-5:]),
                
                'away_avg_gf_l10': np.mean(away_goals_for),
                'away_avg_ga_l10': np.mean(away_goals_against),
                'away_avg_gf_l5': np.mean(away_goals_for[-5:]),
                'away_avg_ga_l5': np.mean(away_goals_against[-5:]),
                
                # Odds features (THE IMPORTANT ONES!)
                'home_odds': match['home_odds_avg'],
                'draw_odds': match['draw_odds_avg'],
                'away_odds': match['away_odds_avg'],
                'home_prob': match['home_prob_norm'],
                'draw_prob': match['draw_prob_norm'],
                'away_prob': match['away_prob_norm'],
                
                # Derived features
                'expected_goals': np.mean(home_goals_for[-5:]) + np.mean(away_goals_for[-5:]),
                'combined_attacking': (1/match['home_odds_avg']) + (1/match['away_odds_avg']),
                'odds_competitiveness': abs(match['home_odds_avg'] - match['away_odds_avg']),
            }
            
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        print(f"\n✓ Created features for {len(features_df)} matches")
        print(f"✓ Total features: {len(features_df.columns) - 7}")  # Excluding IDs and targets
        
        return features_df
    
    def train_model(self, features_df, target='over_2_5'):
        """
        Train prediction model
        """
        print("\n" + "="*60)
        print(f"TRAINING MODEL: {target.upper()}")
        print("="*60)
        
        # Feature columns
        feature_cols = [col for col in features_df.columns if col not in [
            'match_idx', 'date', 'home_team', 'away_team',
            'total_goals', 'over_2_5', 'btts', 'outcome'
        ]]
        
        print(f"\nFeatures used: {len(feature_cols)}")
        print(f"Sample: {feature_cols[:5]}")
        
        X = features_df[feature_cols]
        y = features_df[target]
        
        # Train-test split (temporal)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
        
        # Train
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Accuracy: {accuracy*100:.2f}%")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{target} - Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'results_{target}.png')
        print(f"\n✓ Saved confusion matrix to results_{target}.png")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        return accuracy, feature_importance
    
    def run_complete_analysis(self, league='portugal'):
        """
        Run complete analysis pipeline
        """
        # Load data
        self.load_and_prepare_data(league=league)
        
        # Engineer features
        features_df = self.engineer_features()
        
        # Save features
        features_df.to_csv(f'data/processed/{league}_features_with_odds.csv', index=False)
        print(f"\n✓ Saved features to data/processed/{league}_features_with_odds.csv")
        
        # Train models for different targets
        results = {}
        
        for target in ['over_2_5', 'btts']:
            print("\n" + "="*60)
            accuracy, importance = self.train_model(features_df, target=target)
            results[target] = {'accuracy': accuracy, 'importance': importance}
        
        # Summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        
        for target, data in results.items():
            print(f"\n{target.upper()}:")
            print(f"  Accuracy: {data['accuracy']*100:.2f}%")
            print(f"  Top feature: {data['importance'].iloc[0]['feature']}")
        
        return results


if __name__ == "__main__":
    predictor = OddsBasedPredictor()
    
    # Run for Portuguese league (since that's what you actually have)
    results = predictor.run_complete_analysis(league='portugal')