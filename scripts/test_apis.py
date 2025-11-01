from data_collector import ComprehensiveFootballCollector
import os
from dotenv import load_dotenv

load_dotenv()

print("Testing API connections...\n")

collector = ComprehensiveFootballCollector(
    api_football_key=os.getenv('API_FOOTBALL_KEY'),
    weather_api_key=os.getenv('WEATHER_API_KEY')
)

# Test 1: API-Football
print("1. Testing API-Football...")
try:
    fixtures = collector.get_fixtures_detailed(league_id=39, season=2024)
    if fixtures:
        print(f"   ✓ SUCCESS! Found {len(fixtures)} Premier League 2024 fixtures")
        print(f"   Sample: {fixtures[0]['teams']['home']['name']} vs {fixtures[0]['teams']['away']['name']}")
    else:
        print("   ✗ No data returned")
except Exception as e:
    print(f"   ✗ ERROR: {e}")

# Test 2: Weather API
print("\n2. Testing OpenWeatherMap...")
try:
    # Test with London coordinates (Arsenal's Emirates Stadium)
    weather = collector.get_historical_weather(51.5549, -0.1084, None)
    if weather:
        print(f"   ✓ SUCCESS! Weather data retrieved")
        print(f"   Temperature: {weather.get('temperature')}°C")
        print(f"   Condition: {weather.get('weather_condition')}")
    else:
        print("   ✗ No weather data returned")
except Exception as e:
    print(f"   ✗ ERROR: {e}")

print("\n✅ API testing complete!")