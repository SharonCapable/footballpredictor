from hybrid_data_pipeline import HybridDataPipeline
import os
from dotenv import load_dotenv

load_dotenv()

print("="*60)
print("FOOTBALL PREDICTION - FIRST DATA COLLECTION")
print("="*60)

# Initialize pipeline
pipeline = HybridDataPipeline(
    api_football_key=os.getenv('API_FOOTBALL_KEY'),
    weather_api_key=os.getenv('WEATHER_API_KEY'),
    db_config={
        'host': os.getenv('DB_HOST'),
        'port': int(os.getenv('DB_PORT')),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD')
    }
)

# Specify your CSV files (adjust paths as needed)
kaggle_files = {
    'Premier League': 'data/kaggle/epl_2122.csv',
    'Primeira Liga': 'data/kaggle/portugal_2122.csv'
}

print("\nChecking files...")
for league, path in kaggle_files.items():
    if os.path.exists(path):
        print(f"  ✓ {league}: {path}")
    else:
        print(f"  ✗ MISSING: {path}")
        print(f"    Please download from football-data.co.uk")

ready = input("\nReady to start? (yes/no): ")

if ready.lower() == 'yes':
    print("\nStarting collection...")
    pipeline.run_hybrid_pipeline(kaggle_files)
    pipeline.close()
    print("\n✅ COMPLETE! Run 'python verify_data.py' to check results")
else:
    print("Cancelled. Download CSV files first.")