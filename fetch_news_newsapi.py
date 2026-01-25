#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetch news articles for selected publicly traded companies from the last 15 days
using NewsAPI.org and store them in designated folders under the outputs directory.

NewsAPI Free Tier Limitations:
- 100 requests per day
- Max 100 results per request
- News from last 30 days only
- No commercial use
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import time

try:
    from newspaper import Article
except ImportError:
    print("ERROR: newspaper4k package not found. Install with: pip install newspaper4k", file=sys.stderr)
    sys.exit(1)

import requests

# NewsAPI endpoint
NEWSAPI_BASE_URL = "https://newsapi.org/v2"

# Company symbols and their search keywords
COMPANIES = {
    "AAPL": "Apple Inc OR iPhone OR iPad OR Mac OR Tim Cook",
    "MSFT": "Microsoft OR Windows OR Azure OR Satya Nadella",
    "NVDA": "NVIDIA OR GeForce OR RTX OR Jensen Huang",
    "AMZN": "Amazon OR AWS OR Alexa OR Jeff Bezos OR Andy Jassy",
    "GOOGL": "Google OR Alphabet OR Android OR Sundar Pichai",
}

def eprint(msg: str) -> None:
    """Print error message to stderr."""
    print(msg, file=sys.stderr)

def safe_mkdir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def get_api_key() -> str:
    """
    Get NewsAPI key from environment variable or credentials file.
    
    Returns:
        API key string
    """
    # Try environment variable first
    api_key = os.environ.get('NEWSAPI_KEY')
    
    if not api_key:
        # Try reading from credentials file
        cred_file = os.path.join(os.path.dirname(__file__), 'cred', 'newsapi_credentials.json')
        if os.path.exists(cred_file):
            try:
                with open(cred_file, 'r') as f:
                    creds = json.load(f)
                    api_key = creds.get('api_key')
            except Exception as e:
                eprint(f"Error reading credentials file: {e}")
    
    if not api_key:
        eprint("ERROR: NewsAPI key not found!")
        eprint("Please set NEWSAPI_KEY environment variable or create cred/newsapi_credentials.json")
        eprint('Example: {"api_key": "your_key_here"}')
        eprint("\nGet your free API key at: https://newsapi.org/register")
        sys.exit(1)
    
    return api_key

def fetch_company_news_newsapi(
    symbol: str,
    search_query: str,
    api_key: str,
    days: int = 15,
    max_results: int = 100
) -> List[Dict]:
    """
    Fetch news articles for a specific company using NewsAPI.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
        search_query: Search query for the company
        api_key: NewsAPI key
        days: Number of days to look back
        max_results: Maximum number of articles to retrieve
        
    Returns:
        List of article dictionaries
    """
    print(f"Fetching news for {symbol}...")
    
    # Calculate date range
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)
    
    # Prepare API request
    url = f"{NEWSAPI_BASE_URL}/everything"
    params = {
        'q': search_query,
        'from': from_date.strftime('%Y-%m-%d'),
        'to': to_date.strftime('%Y-%m-%d'),
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': min(max_results, 100),  # NewsAPI max is 100
        'apiKey': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') != 'ok':
            eprint(f"API Error: {data.get('message', 'Unknown error')}")
            return []
        
        articles = data.get('articles', [])
        
        if not articles:
            print(f"  No articles found for {symbol}")
            return []
        
        print(f"  Found {len(articles)} articles from NewsAPI")
        
        # Enrich articles with full content
        enriched_articles = []
        for i, article in enumerate(articles, 1):
            print(f"  Processing article {i}/{len(articles)}: {article.get('title', 'N/A')[:50]}...")
            
            article_url = article.get('url', '')
            full_text = article.get('content', '')  # NewsAPI provides partial content
            
            # Try to fetch full article content using newspaper4k
            if article_url:
                try:
                    news_article = Article(article_url)
                    news_article.download()
                    news_article.parse()
                    
                    # Use newspaper4k text if it's longer than NewsAPI's content
                    if news_article.text and len(news_article.text) > len(full_text):
                        full_text = news_article.text
                        print(f"    ✓ Extracted {len(full_text)} characters from full article")
                    else:
                        print(f"    → Using NewsAPI content ({len(full_text)} characters)")
                    
                    # Small delay to be polite
                    time.sleep(0.3)
                    
                except Exception as e:
                    print(f"    ⚠ Could not fetch full article, using NewsAPI content")
            
            article_data = {
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'published_date': article.get('publishedAt', ''),
                'url': article_url,
                'source': article.get('source', {}).get('name', ''),
                'author': article.get('author', ''),
                'image_url': article.get('urlToImage', ''),
                'text': full_text,
            }
            
            enriched_articles.append(article_data)
        
        print(f"Successfully processed {len(enriched_articles)} articles for {symbol}")
        return enriched_articles
        
    except requests.exceptions.RequestException as e:
        eprint(f"Error fetching news for {symbol}: {e}")
        return []
    except Exception as e:
        eprint(f"Unexpected error for {symbol}: {e}")
        return []

def save_articles_to_json(articles: List[Dict], output_dir: str, symbol: str) -> None:
    """
    Save articles to a JSON file in the designated company folder.
    
    Args:
        articles: List of article dictionaries
        output_dir: Base output directory
        symbol: Stock ticker symbol
    """
    if not articles:
        eprint(f"No articles to save for {symbol}")
        return
    
    # Create company-specific folder
    company_folder = os.path.join(output_dir, symbol)
    safe_mkdir(company_folder)
    
    # Generate filename with current date
    today = datetime.now().strftime("%Y%m%d")
    filename = f"{symbol}_news_newsapi_{today}.json"
    filepath = os.path.join(company_folder, filename)
    
    # Save to JSON
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'symbol': symbol,
                'source': 'NewsAPI',
                'fetch_date': today,
                'article_count': len(articles),
                'articles': articles
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(articles)} articles to {filepath}")
        
    except Exception as e:
        eprint(f"Error saving articles for {symbol}: {e}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Fetch news articles for selected companies using NewsAPI.org"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for news articles (default: outputs)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=15,
        help="Number of days to look back (default: 15)"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=100,
        help="Maximum number of articles per company (default: 100, max: 100)"
    )
    parser.add_argument(
        "--symbols",
        nargs='+',
        default=list(COMPANIES.keys()),
        help=f"Stock symbols to fetch news for (default: {' '.join(COMPANIES.keys())})"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="NewsAPI key (or set NEWSAPI_KEY environment variable)"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or get_api_key()
    
    # Validate symbols
    invalid_symbols = [s for s in args.symbols if s not in COMPANIES]
    if invalid_symbols:
        eprint(f"Warning: Unknown symbols will be skipped: {', '.join(invalid_symbols)}")
    
    valid_symbols = [s for s in args.symbols if s in COMPANIES]
    if not valid_symbols:
        eprint("Error: No valid symbols provided")
        sys.exit(1)
    
    print("=" * 70)
    print("NewsAPI Article Fetcher")
    print("=" * 70)
    print(f"Companies: {len(valid_symbols)}")
    print(f"Date range: Last {args.days} days")
    print(f"Max articles per company: {args.max_results}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)
    
    # Fetch news for each company
    for symbol in valid_symbols:
        search_query = COMPANIES[symbol]
        articles = fetch_company_news_newsapi(
            symbol=symbol,
            search_query=search_query,
            api_key=api_key,
            days=args.days,
            max_results=args.max_results
        )
        
        # Save articles
        save_articles_to_json(articles, args.output_dir, symbol)
        print("-" * 70)
        
        # Small delay between companies to respect rate limits
        time.sleep(1)
    
    print("=" * 70)
    print("News fetch completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
