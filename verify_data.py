"""
Data Verification Script
Checks what data has been collected and stored in the database
"""

from database_manager import FootballDatabaseManager
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize database
db = FootballDatabaseManager(
    host=os.getenv('DB_HOST'),
    port=int(os.getenv('DB_PORT')),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)

print("="*60)
print("DATA VERIFICATION REPORT")
print("="*60)

# Get stats
stats = db.get_database_stats()

print("\n1. DATABASE OVERVIEW")
print("-"*60)
for key, value in stats.items():
    print(f"   {key.replace('_', ' ').title():<30}: {value:>10}")

# Check data by league
print("\n2. MATCHES BY LEAGUE")
print("-"*60)

with db.get_cursor() as cursor:
    cursor.execute("""
        SELECT l.league_name, COUNT(m.match_id) as match_count
        FROM matches m
        JOIN leagues l ON m.league_id = l.league_id
        GROUP BY l.league_name
        ORDER BY match_count DESC
    """)
    
    results = cursor.fetchall()
    if results:
        for row in results:
            print(f"   {row['league_name']:<30}: {row['match_count']:>5} matches")
    else:
        print("   No matches found")

# Check data by season
print("\n3. MATCHES BY SEASON")
print("-"*60)

with db.get_cursor() as cursor:
    cursor.execute("""
        SELECT season, COUNT(*) as match_count
        FROM matches
        GROUP BY season
        ORDER BY season DESC
    """)
    
    results = cursor.fetchall()
    if results:
        for row in results:
            print(f"   Season {row['season']:<24}: {row['match_count']:>5} matches")
    else:
        print("   No matches found")

# Check outcome distribution
print("\n4. OUTCOME DISTRIBUTION")
print("-"*60)

with db.get_cursor() as cursor:
    cursor.execute("""
        SELECT 
            outcome,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
        FROM matches
        WHERE outcome IS NOT NULL
        GROUP BY outcome
        ORDER BY outcome
    """)
    
    results = cursor.fetchall()
    if results:
        outcome_labels = {'H': 'Home Wins', 'D': 'Draws', 'A': 'Away Wins'}
        for row in results:
            label = outcome_labels.get(row['outcome'], row['outcome'])
            print(f"   {label:<30}: {row['count']:>5} ({row['percentage']:>5}%)")
    else:
        print("   No outcome data found")

# Check sample matches
print("\n5. SAMPLE MATCHES")
print("-"*60)

with db.get_cursor() as cursor:
    cursor.execute("""
        SELECT 
            m.fixture_date::date as date,
            ht.team_name as home_team,
            at.team_name as away_team,
            m.home_goals,
            m.away_goals,
            m.outcome
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.team_id
        JOIN teams at ON m.away_team_id = at.team_id
        ORDER BY m.fixture_date DESC
        LIMIT 5
    """)
    
    results = cursor.fetchall()
    if results:
        for row in results:
            outcome_symbol = {'H': '✓', 'D': '=', 'A': '✗'}.get(row['outcome'], '?')
            print(f"   {row['date']} | {row['home_team']:<20} {row['home_goals']}-{row['away_goals']} {row['away_team']:<20} [{outcome_symbol}]")
    else:
        print("   No matches found")

# Check teams
print("\n6. TEAMS IN DATABASE")
print("-"*60)

with db.get_cursor() as cursor:
    cursor.execute("""
        SELECT 
            t.team_name,
            COUNT(DISTINCT m.match_id) as matches_played
        FROM teams t
        LEFT JOIN matches m ON t.team_id = m.home_team_id OR t.team_id = m.away_team_id
        GROUP BY t.team_name
        ORDER BY matches_played DESC
        LIMIT 10
    """)
    
    results = cursor.fetchall()
    if results:
        for row in results:
            print(f"   {row['team_name']:<30}: {row['matches_played']:>5} matches")
    else:
        print("   No teams found")

# Check date range
print("\n7. DATA DATE RANGE")
print("-"*60)

with db.get_cursor() as cursor:
    cursor.execute("""
        SELECT 
            MIN(fixture_date::date) as earliest,
            MAX(fixture_date::date) as latest,
            MAX(fixture_date::date) - MIN(fixture_date::date) as span
        FROM matches
    """)
    
    result = cursor.fetchone()
    if result and result['earliest']:
        print(f"   Earliest Match                : {result['earliest']}")
        print(f"   Latest Match                  : {result['latest']}")
        print(f"   Total Span                    : {result['span']} days")
    else:
        print("   No date information available")

print("\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60)

db.close()