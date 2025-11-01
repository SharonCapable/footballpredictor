import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data with odds
df = pd.read_csv('data/processed/features_with_odds.csv')

print(f"Loaded {len(df)} matches with odds")

# Features (exclude IDs and targets)
exclude_cols = ['match_id', 'fixture_date', 'home_team_id', 'away_team_id',
                'home_goals', 'away_goals', 'total_goals', 'over_2_5', 'btts']

feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"Total features: {len(feature_cols)}")
print(f"Odds features: {[col for col in feature_cols if 'odds' in col.lower() or 'prob' in col.lower()]}")

X = df[feature_cols].fillna(0)
y = df['over_2_5']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
rf = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*60)
print("MODEL WITH BETTING ODDS")
print("="*60)
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Improvement: {(accuracy - 0.4571)*100:.2f} percentage points")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Under 2.5', 'Over 2.5']))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Check if odds features are important
odds_features = feature_importance[feature_importance['feature'].str.contains('odds|prob', case=False)]
print(f"\nOdds-based features in top 10: {len(odds_features.head(10))}")