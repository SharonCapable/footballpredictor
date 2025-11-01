import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('API_FOOTBALL_KEY')

print(f"Testing API-Sports.io...")
print(f"API Key (first 20 chars): {api_key[:20]}...")
print()

url = "https://v3.football.api-sports.io/leagues"

headers = {
    "x-apisports-key": api_key
}

response = requests.get(url, headers=headers)

print(f"Status code: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    
    if data.get('results', 0) > 0:
        print(f"✓ SUCCESS! Found {data['results']} leagues")
        print(f"\nSample leagues:")
        for league in data['response'][:3]:
            print(f"  - {league['league']['name']} ({league['country']['name']})")
    else:
        print(f"✗ Error: {data.get('errors', 'Unknown error')}")
        print(f"Response: {data}")
else:
    print(f"✗ HTTP Error {response.status_code}")
    print(f"Response: {response.text}")