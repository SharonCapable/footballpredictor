import pandas as pd

# Load new features
df = pd.read_csv('data/processed/improved_features.csv')

# Prepare data
feature_cols = [col for col in df.columns if col not in [
    'match_id', 'fixture_date', 'home_team_id', 'away_team_id',
    'home_goals', 'away_goals', 'total_goals', 'over_2_5', 'btts'
]]

X = df[feature_cols]
y = df['over_2_5']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import accuracy_score
y_pred = rf.predict(X_test)
print(f"NEW FEATURES Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")