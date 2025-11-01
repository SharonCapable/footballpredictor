"""
Football News Scraper with Sentiment Analysis
Scrapes news from multiple sources and analyzes sentiment
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
import re
from textblob import TextBlob
from urllib.parse import urljoin, quote

class FootballNewsScraper:
    def __init__(self, rate_limit_delay=2):
        """
        Initialize news scraper
        
        Args:
            rate_limit_delay: Seconds to wait between requests
        """
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    # ==================== BBC SPORT ====================
    
    def scrape_bbc_sport_team_news(self, team_name, max_articles=10):
        """
        Scrape team news from BBC Sport
        
        Args:
            team_name: Name of the team (e.g., "Arsenal", "Manchester United")
            max_articles: Maximum number of articles to scrape
        """
        articles = []
        
        try:
            # BBC Sport search URL
            search_query = quote(f"{team_name} football")
            url = f"https://www.bbc.co.uk/search?q={search_query}&filter=sport"
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article containers (structure may change)
            article_containers = soup.find_all('div', class_='ssrcss-1f3bvyz-Stack', limit=max_articles)
            
            for container in article_containers:
                try:
                    title_elem = container.find('a')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    link = urljoin('https://www.bbc.co.uk', title_elem.get('href', ''))
                    
                    # Get date if available
                    date_elem = container.find('time')
                    pub_date = date_elem.get('datetime') if date_elem else None
                    
                    article = {
                        'headline': title,
                        'source_url': link,
                        'source': 'BBC Sport',
                        'published_date': pub_date,
                        'team_name': team_name
                    }
                    
                    articles.append(article)
                    
                except Exception as e:
                    print(f"Error parsing BBC article: {e}")
                    continue
            
            time.sleep(self.rate_limit_delay)
            
        except Exception as e:
            print(f"Error scraping BBC Sport: {e}")
        
        return articles
    
    # ==================== SKY SPORTS ====================
    
    def scrape_sky_sports_team_news(self, team_name, max_articles=10):
        """Scrape team news from Sky Sports"""
        articles = []
        
        try:
            # Sky Sports team pages usually follow pattern
            team_slug = team_name.lower().replace(' ', '-')
            url = f"https://www.skysports.com/{team_slug}-news"
            
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 404:
                # Try search instead
                search_url = f"https://www.skysports.com/search?q={quote(team_name)}"
                response = self.session.get(search_url, timeout=10)
            
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news articles
            article_containers = soup.find_all('div', class_='news-list__item', limit=max_articles)
            
            for container in article_containers:
                try:
                    title_elem = container.find('a', class_='news-list__headline-link')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    link = urljoin('https://www.skysports.com', title_elem.get('href', ''))
                    
                    # Get date
                    date_elem = container.find('span', class_='label__timestamp')
                    pub_date = date_elem.get_text(strip=True) if date_elem else None
                    
                    article = {
                        'headline': title,
                        'source_url': link,
                        'source': 'Sky Sports',
                        'published_date': pub_date,
                        'team_name': team_name
                    }
                    
                    articles.append(article)
                    
                except Exception as e:
                    print(f"Error parsing Sky Sports article: {e}")
                    continue
            
            time.sleep(self.rate_limit_delay)
            
        except Exception as e:
            print(f"Error scraping Sky Sports: {e}")
        
        return articles
    
    # ==================== ESPN ====================
    
    def scrape_espn_team_news(self, team_name, max_articles=10):
        """Scrape team news from ESPN"""
        articles = []
        
        try:
            # ESPN search
            search_url = f"https://www.espn.com/search/_/q/{quote(team_name)}%20football"
            
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links
            article_links = soup.find_all('a', class_='contentItem__title', limit=max_articles)
            
            for link_elem in article_links:
                try:
                    title = link_elem.get_text(strip=True)
                    link = urljoin('https://www.espn.com', link_elem.get('href', ''))
                    
                    article = {
                        'headline': title,
                        'source_url': link,
                        'source': 'ESPN',
                        'published_date': None,  # Would need to visit article page
                        'team_name': team_name
                    }
                    
                    articles.append(article)
                    
                except Exception as e:
                    print(f"Error parsing ESPN article: {e}")
                    continue
            
            time.sleep(self.rate_limit_delay)
            
        except Exception as e:
            print(f"Error scraping ESPN: {e}")
        
        return articles
    
    # ==================== PLAYER-SPECIFIC NEWS ====================
    
    def scrape_player_news(self, player_name, max_articles=5):
        """Scrape news about a specific player"""
        all_articles = []
        
        # Search BBC Sport for player
        try:
            search_query = quote(f"{player_name} football")
            url = f"https://www.bbc.co.uk/search?q={search_query}&filter=sport"
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            article_containers = soup.find_all('div', class_='ssrcss-1f3bvyz-Stack', limit=max_articles)
            
            for container in article_containers:
                try:
                    title_elem = container.find('a')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    
                    # Only include if player name is in title
                    if player_name.lower() in title.lower():
                        link = urljoin('https://www.bbc.co.uk', title_elem.get('href', ''))
                        
                        article = {
                            'headline': title,
                            'source_url': link,
                            'source': 'BBC Sport',
                            'player_name': player_name
                        }
                        
                        all_articles.append(article)
                
                except Exception as e:
                    continue
            
            time.sleep(self.rate_limit_delay)
            
        except Exception as e:
            print(f"Error scraping player news: {e}")
        
        return all_articles
    
    # ==================== ARTICLE CONTENT EXTRACTION ====================
    
    def extract_article_content(self, url):
        """
        Extract full article text from URL
        (Simplified - would need source-specific parsing)
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove scripts, styles
            for script in soup(['script', 'style']):
                script.decompose()
            
            # Try to find article content (generic approach)
            article_body = soup.find('article') or soup.find('div', class_=re.compile('article|content|story'))
            
            if article_body:
                paragraphs = article_body.find_all('p')
                text = ' '.join([p.get_text(strip=True) for p in paragraphs])
                return text
            
            return None
            
        except Exception as e:
            print(f"Error extracting article content: {e}")
            return None
    
    # ==================== SENTIMENT ANALYSIS ====================
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text using TextBlob
        
        Returns:
            dict: sentiment_score (-1 to 1), sentiment_label
        """
        if not text:
            return {'sentiment_score': 0, 'sentiment_label': 'Neutral'}
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        # Classify sentiment
        if polarity > 0.1:
            label = 'Positive'
        elif polarity < -0.1:
            label = 'Negative'
        else:
            label = 'Neutral'
        
        return {
            'sentiment_score': round(polarity, 4),
            'sentiment_label': label
        }
    
    def categorize_news(self, headline):
        """
        Categorize news based on headline keywords
        
        Categories: Transfer, Injury, Performance, Personal, Tactical
        """
        headline_lower = headline.lower()
        
        # Keyword matching
        if any(word in headline_lower for word in ['transfer', 'sign', 'signing', 'deal', 'move', 'join']):
            return 'Transfer'
        elif any(word in headline_lower for word in ['injury', 'injured', 'fitness', 'doubt', 'out', 'ruled out', 'sidelined']):
            return 'Injury'
        elif any(word in headline_lower for word in ['goal', 'assist', 'performance', 'rating', 'star', 'hero']):
            return 'Performance'
        elif any(word in headline_lower for word in ['tactical', 'formation', 'strategy', 'lineup']):
            return 'Tactical'
        else:
            return 'General'
    
    # ==================== COMPREHENSIVE SCRAPING ====================
    
    def scrape_comprehensive_team_news(self, team_name, days_back=7):
        """
        Scrape news from all sources for a team
        
        Args:
            team_name: Team name
            days_back: Only include news from last N days
        """
        print(f"Scraping news for {team_name}...")
        
        all_articles = []
        
        # Scrape from all sources
        bbc_articles = self.scrape_bbc_sport_team_news(team_name, max_articles=5)
        sky_articles = self.scrape_sky_sports_team_news(team_name, max_articles=5)
        espn_articles = self.scrape_espn_team_news(team_name, max_articles=5)
        
        all_articles.extend(bbc_articles)
        all_articles.extend(sky_articles)
        all_articles.extend(espn_articles)
        
        # Process each article
        processed_articles = []
        
        for article in all_articles:
            # Analyze sentiment from headline
            sentiment = self.analyze_sentiment(article['headline'])
            article['sentiment_score'] = sentiment['sentiment_score']
            article['sentiment_label'] = sentiment['sentiment_label']
            
            # Categorize news
            article['news_category'] = self.categorize_news(article['headline'])
            
            # Set scraped date
            article['scraped_date'] = datetime.now()
            
            processed_articles.append(article)
        
        print(f"✓ Scraped {len(processed_articles)} articles for {team_name}")
        
        return processed_articles
    
    def scrape_injury_news(self, team_name):
        """
        Specifically scrape injury-related news
        Filters for injury keywords
        """
        all_news = self.scrape_comprehensive_team_news(team_name)
        
        injury_news = [
            article for article in all_news 
            if article['news_category'] == 'Injury'
        ]
        
        return injury_news
    
    # ==================== BATCH SCRAPING ====================
    
    def scrape_multiple_teams(self, team_names, delay_between_teams=5):
        """
        Scrape news for multiple teams with rate limiting
        
        Args:
            team_names: List of team names
            delay_between_teams: Seconds to wait between teams
        """
        all_results = {}
        
        for team_name in team_names:
            print(f"\nScraping {team_name}...")
            
            articles = self.scrape_comprehensive_team_news(team_name)
            all_results[team_name] = articles
            
            # Rate limiting
            if team_name != team_names[-1]:  # Don't delay after last team
                print(f"Waiting {delay_between_teams} seconds...")
                time.sleep(delay_between_teams)
        
        return all_results
    
    # ==================== DATA EXPORT ====================
    
    def save_to_dataframe(self, articles):
        """Convert articles list to DataFrame"""
        if not articles:
            return pd.DataFrame()
        
        df = pd.DataFrame(articles)
        return df
    
    def save_to_csv(self, articles, filename):
        """Save articles to CSV"""
        df = self.save_to_dataframe(articles)
        df.to_csv(filename, index=False)
        print(f"✓ Saved {len(articles)} articles to {filename}")


# Example usage
if __name__ == "__main__":
    scraper = FootballNewsScraper(rate_limit_delay=2)
    
    # Example 1: Scrape news for one team
    team_name = "Arsenal"
    articles = scraper.scrape_comprehensive_team_news(team_name)
    
    print(f"\nFound {len(articles)} articles")
    for article in articles[:3]:
        print(f"\n{article['source']}: {article['headline']}")
        print(f"Sentiment: {article['sentiment_label']} ({article['sentiment_score']})")
        print(f"Category: {article['news_category']}")
    
    # Example 2: Scrape injury news
    injury_articles = scraper.scrape_injury_news(team_name)
    print(f"\nFound {len(injury_articles)} injury-related articles")
    
    # Example 3: Scrape multiple teams
    teams = ["Arsenal", "Chelsea", "Manchester United"]
    all_news = scraper.scrape_multiple_teams(teams, delay_between_teams=3)
    
    # Save to CSV
    for team, articles in all_news.items():
        if articles:
            scraper.save_to_csv(articles, f'data/news/{team}_news.csv')