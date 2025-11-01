"""
Multi-League Football Data Scraper
Downloads historical match data from Football-Data.co.uk for multiple leagues
"""

import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
import time

class MultiLeagueScraper:
    """Scrapes football data from multiple top European leagues"""
    
    BASE_URL = "https://www.football-data.co.uk"
    
    LEAGUES = {
        'england': {'code': 'E0', 'name': 'Premier League'},
        'spain': {'code': 'SP1', 'name': 'La Liga'},
        'germany': {'code': 'D1', 'name': 'Bundesliga'},
        'italy': {'code': 'I1', 'name': 'Serie A'},
        'france': {'code': 'F1', 'name': 'Ligue 1'}
    }
    
    def __init__(self, output_dir='data/raw'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_league_season(self, league_code, season):
        """
        Download a single season for a league
        
        Args:
            league_code: League code (e.g., 'E0', 'SP1')
            season: Season in format 'YYYY' (e.g., '2324' for 2023/24)
        
        Returns:
            DataFrame or None if download fails
        """
        # URL format: https://www.football-data.co.uk/mmz4281/2324/E0.csv
        url = f"{self.BASE_URL}/mmz4281/{season}/{league_code}.csv"
        
        try:
            print(f"  📥 Downloading {league_code} season {season}...", end='')
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Read CSV from response
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            # Add season column
            df['Season'] = f"20{season[:2]}/20{season[2:]}"
            
            print(f" ✓ ({len(df)} matches)")
            return df
            
        except requests.exceptions.RequestException as e:
            print(f" ✗ Failed")
            return None
    
    def scrape_league(self, league_key, seasons=None):
        """
        Scrape all seasons for a specific league
        
        Args:
            league_key: Key from LEAGUES dict (e.g., 'england')
            seasons: List of seasons to download (default: last 5 seasons)
        
        Returns:
            Combined DataFrame for all seasons
        """
        if league_key not in self.LEAGUES:
            raise ValueError(f"Unknown league: {league_key}. Choose from {list(self.LEAGUES.keys())}")
        
        league_info = self.LEAGUES[league_key]
        league_code = league_info['code']
        league_name = league_info['name']
        
        print(f"\n{'='*60}")
        print(f"🏆 SCRAPING: {league_name.upper()} ({league_key})")
        print(f"{'='*60}")
        
        # Default to last 5 seasons if not specified
        if seasons is None:
            current_year = datetime.now().year
            current_month = datetime.now().month
            
            # If before August, season hasn't started yet
            if current_month < 8:
                current_year -= 1
            
            # Generate last 5 seasons (e.g., 2324, 2223, 2122, 2021, 1920)
            seasons = []
            for i in range(5):
                year = current_year - i
                season_code = f"{str(year-1)[2:]}{str(year)[2:]}"
                seasons.append(season_code)
        
        print(f"📅 Seasons: {', '.join(seasons)}")
        
        # Download each season
        all_data = []
        for season in seasons:
            df = self.download_league_season(league_code, season)
            if df is not None:
                all_data.append(df)
            time.sleep(0.5)  # Be nice to the server
        
        if not all_data:
            print(f"❌ No data downloaded for {league_name}")
            return None
        
        # Combine all seasons
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Add league identifier
        combined_df['League'] = league_key
        combined_df['League_Name'] = league_name
        
        # Save to CSV
        output_file = self.output_dir / f"{league_key}_matches.csv"
        combined_df.to_csv(output_file, index=False)
        
        print(f"\n✓ Total matches: {len(combined_df)}")
        print(f"✓ Saved to: {output_file}")
        
        return combined_df
    
    def scrape_all_leagues(self, seasons=None):
        """
        Scrape all available leagues
        
        Args:
            seasons: List of seasons to download (default: last 5 seasons)
        
        Returns:
            Dictionary with league data {league_key: DataFrame}
        """
        print("\n" + "="*60)
        print("🌍 SCRAPING ALL LEAGUES")
        print("="*60)
        
        results = {}
        summary = []
        
        for league_key in self.LEAGUES.keys():
            df = self.scrape_league(league_key, seasons)
            if df is not None:
                results[league_key] = df
                summary.append({
                    'League': self.LEAGUES[league_key]['name'],
                    'Matches': len(df),
                    'Seasons': df['Season'].nunique(),
                    'Date Range': f"{df['Season'].min()} - {df['Season'].max()}"
                })
        
        # Print summary
        print("\n" + "="*60)
        print("📊 SCRAPING SUMMARY")
        print("="*60)
        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(index=False))
        print(f"\n✓ Total matches across all leagues: {sum(s['Matches'] for s in summary)}")
        
        return results
    
    def standardize_columns(self, df):
        """
        Standardize column names for consistency with your existing code
        
        Args:
            df: Raw DataFrame from Football-Data.co.uk
        
        Returns:
            Standardized DataFrame
        """
        # Key column mappings (Football-Data.co.uk -> Your format)
        column_mapping = {
            'Date': 'date',
            'HomeTeam': 'home_team',
            'AwayTeam': 'away_team',
            'FTHG': 'home_score',  # Full Time Home Goals
            'FTAG': 'away_score',  # Full Time Away Goals
            'FTR': 'result',       # Full Time Result (H/D/A)
            'HTHG': 'ht_home_score',
            'HTAG': 'ht_away_score',
            'HTR': 'ht_result',
            'HS': 'home_shots',
            'AS': 'away_shots',
            'HST': 'home_shots_on_target',
            'AST': 'away_shots_on_target',
            'HF': 'home_fouls',
            'AF': 'away_fouls',
            'HC': 'home_corners',
            'AC': 'away_corners',
            'HY': 'home_yellow_cards',
            'AY': 'away_yellow_cards',
            'HR': 'home_red_cards',
            'AR': 'away_red_cards'
        }
        
        # Rename columns that exist
        rename_dict = {old: new for old, new in column_mapping.items() if old in df.columns}
        df_standardized = df.rename(columns=rename_dict)
        
        # Convert result to numeric (0=Away Win, 1=Draw, 2=Home Win)
        if 'result' in df_standardized.columns:
            result_map = {'A': 0, 'D': 1, 'H': 2}
            df_standardized['outcome'] = df_standardized['result'].map(result_map)
        
        return df_standardized


def main():
    """Main execution function"""
    print("\n🚀 MULTI-LEAGUE FOOTBALL DATA SCRAPER")
    print("="*60)
    
    # Initialize scraper
    scraper = MultiLeagueScraper()
    
    # Option 1: Scrape all leagues (last 5 seasons)
    print("\n📋 OPTION 1: Scrape all leagues (recommended)")
    print("   - England, Spain, Germany, Italy, France")
    print("   - Last 5 seasons each (~1900 matches per league)")
    
    print("\n📋 OPTION 2: Scrape specific league")
    print("   - Choose one: england, spain, germany, italy, france")
    
    print("\n📋 OPTION 3: Custom seasons")
    print("   - Specify seasons manually (e.g., ['2324', '2223'])")
    
    choice = input("\nEnter choice (1/2/3) or press Enter for Option 1: ").strip()
    
    if choice == '2':
        league = input("Enter league (england/spain/germany/italy/france): ").strip().lower()
        scraper.scrape_league(league)
        
    elif choice == '3':
        league = input("Enter league: ").strip().lower()
        seasons_input = input("Enter seasons (comma-separated, e.g., 2324,2223): ").strip()
        seasons = [s.strip() for s in seasons_input.split(',')]
        scraper.scrape_league(league, seasons)
        
    else:
        # Default: scrape all leagues
        results = scraper.scrape_all_leagues()
        
        # Optional: Create a combined dataset
        if results:
            print("\n🔄 Creating combined dataset...")
            all_matches = pd.concat(results.values(), ignore_index=True)
            
            # Standardize columns
            all_matches = scraper.standardize_columns(all_matches)
            
            combined_file = Path('data/raw/all_leagues_combined.csv')
            all_matches.to_csv(combined_file, index=False)
            print(f"✓ Combined dataset saved: {combined_file}")
            print(f"✓ Total matches: {len(all_matches)}")
    
    print("\n✅ SCRAPING COMPLETE!")
    print("\nNext steps:")
    print("  1. Run preprocessing: python preprocess_data.py")
    print("  2. Retrain models: python complete_system.py")
    print("  3. Check accuracy improvements!")


if __name__ == "__main__":
    main()