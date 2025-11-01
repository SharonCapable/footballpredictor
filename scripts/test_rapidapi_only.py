import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = "9db140d831msha53acd3de874957p1ea01ajsn2affbced4862"

print(f"Your API key: {api_key[:20]}... (first 20 chars)")
print(f"Key length: {len(api_key)} characters")
print()

# MUST use RapidAPI URL with RapidAPI key
url = "https://api-football-v1.p.rapidapi.com/v3/leagues"

headers = {
    "X-RapidAPI-Key": api_key,
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
}

print("Testing API-Football connection...")
response = requests.get(url, headers=headers)

print(f"Status code: {response.status_code}")
print()

if response.status_code == 200:
    data = response.json()
    print("✓ SUCCESS! API is working!")
    print(f"Response keys: {list(data.keys())}")
    
    if 'response' in data and len(data['response']) > 0:
        print(f"✓ Found {len(data['response'])} leagues")
        print(f"\nSample leagues:")
        for league in data['response'][:3]:
            print(f"  - {league['league']['name']} ({league['country']['name']})")
    else:
        print(f"Response: {data}")
        
elif response.status_code == 403:
    print("✗ 403 Forbidden - Subscription issue")
elif response.status_code == 429:
    print("✗ 429 Rate limit exceeded")
else:
    print(f"✗ Error {response.status_code}: {response.text}")