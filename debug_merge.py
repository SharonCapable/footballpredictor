import pandas as pd
import os, sys
sys.path.append('..')
from dotenv import load_dotenv
from database_manager import FootballDatabaseManager

load_dotenv()

# 1. Check your database team names
db = FootballDatabaseManager(
    host=os.getenv('DB_HOST'),
    port=int(os.getenv('DB_PORT')),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)

with db.get_cursor() as cursor:
    cursor.execute("""
        SELECT DISTINCT t.team_id, t.team_name, l.league_name
        FROM teams t
        JOIN matches m ON t.team_id = m.home_team_id OR t.team_id = m.away_team_id
        JOIN leagues l ON m.league_id = l.league_id
        ORDER BY l.league_name, t.team_name
    """)
    
    db_teams = cursor.fetchall()

db.close()

print("="*60)
print("YOUR DATABASE TEAMS")
print("="*60)

for row in db_teams:
    print(f"{row['league_name']:20s} | {row['team_name']}")

# 2. Check odds file team names
print("\n" + "="*60)
print("ODDS FILE TEAMS (EPL)")
print("="*60)

epl_odds = pd.read_csv('data/odds/epl_with_odds.csv')
epl_teams = sorted(set(epl_odds['HomeTeam'].unique()) | set(epl_odds['AwayTeam'].unique()))

for team in epl_teams:
    print(f"  {team}")

print("\n" + "="*60)
print("ODDS FILE TEAMS (Portugal)")
print("="*60)

portugal_odds = pd.read_csv('data/odds/portugal_with_odds.csv')
portugal_teams = sorted(set(portugal_odds['HomeTeam'].unique()) | set(portugal_odds['AwayTeam'].unique()))

for team in portugal_teams:
    print(f"  {team}")

# 3. Check date formats
print("\n" + "="*60)
print("DATE COMPARISON")
print("="*60)

features = pd.read_csv('data/processed/improved_features.csv')
print(f"\nFeatures date range:")
print(f"  Min: {features['fixture_date'].min()}")
print(f"  Max: {features['fixture_date'].max()}")
print(f"  Sample: {features['fixture_date'].head(3).tolist()}")

epl_odds['Date'] = pd.to_datetime(epl_odds['Date'], format='%d/%m/%Y', errors='coerce')
print(f"\nOdds date range:")
print(f"  Min: {epl_odds['Date'].min()}")
print(f"  Max: {epl_odds['Date'].max()}")
print(f"  Sample: {epl_odds['Date'].head(3).tolist()}")

# 4. Sample comparison
print("\n" + "="*60)
print("SAMPLE MATCH COMPARISON")
print("="*60)

print("\nFrom your features (first 5):")
print(features[['fixture_date', 'home_team_id', 'away_team_id']].head())

print("\nFrom odds file (first 5):")
print(epl_odds[['Date', 'HomeTeam', 'AwayTeam']].head())