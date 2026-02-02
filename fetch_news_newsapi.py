#!/usr/bin/env python3  # Shebang line: tells Unix/Linux systems to use Python 3 interpreter from PATH.
# -*- coding: utf-8 -*-  # PEP 263: Declares source file encoding as UTF-8 for international characters.

"""
Fetch news articles for selected publicly traded companies from the last 15 days
using NewsAPI.org and store them in designated folders under the outputs directory.

This script is the fourth stage in the stock analysis pipeline:
1. pull_latest_stock.py - Fetches historical stock data from FMP API
2. train_models.py - Trains NeuralProphet models on historical data
3. generate_forecasts.py - Uses trained models to predict future prices
4. fetch_news_newsapi.py (THIS) - Retrieves recent news articles for sentiment analysis
5. stock_analysis_agent.py - Analyzes forecasts and news to make recommendations

NewsAPI Free Tier Limitations:
- 100 requests per day (combined across all endpoints)
- Max 100 results per request (pageSize limit)
- News from last 30 days only (older articles unavailable)
- No commercial use (development/evaluation only)
- Rate limiting applies (hence time.sleep() between requests)

Workflow:
1. Load API key from environment variable or credentials file.
2. For each company, construct a search query with relevant keywords.
3. Call NewsAPI /everything endpoint to retrieve article metadata.
4. Use newspaper4k to extract full article text from each URL.
5. Save enriched articles to JSON files in company-specific folders.

Output Structure:
    outputs/
    ├── AAPL/
    │   └── AAPL_news_newsapi_20260124.json
    ├── MSFT/
    │   └── MSFT_news_newsapi_20260124.json
    └── ...

Usage:
    python fetch_news_newsapi.py --output-dir outputs --days 15 --max-results 100

Requirements:
    - NewsAPI API key (free at https://newsapi.org/register)
    - newspaper4k package for full-text extraction
"""  # Module docstring explaining the script's purpose, limitations, and workflow.

# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================

import argparse  # Built-in module for parsing command-line arguments.
import json  # Built-in module for JSON serialization/deserialization.
import os  # Built-in module for operating system interactions (file paths, env vars).
import sys  # Built-in module for system-specific parameters (stderr, exit).
from datetime import datetime, timedelta  # Built-in classes for date calculations.
from typing import Dict, List, Optional  # Type hints for better code documentation.
from pathlib import Path  # Object-oriented filesystem path handling.
import time  # Built-in module for delays between API requests (rate limiting).

# =============================================================================
# THIRD-PARTY LIBRARY IMPORTS
# =============================================================================

# Try to import newspaper4k for full-text article extraction
try:  # Graceful handling of missing optional dependency.
    from newspaper import Article  # newspaper4k's Article class for web scraping.
except ImportError:  # Catch if newspaper4k is not installed.
    print("ERROR: newspaper4k package not found. Install with: pip install newspaper4k", file=sys.stderr)  # User-friendly error.
    sys.exit(1)  # Exit with error code 1.

import requests  # Third-party HTTP library for making API calls (simpler than urllib).

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# NewsAPI endpoint - the base URL for all NewsAPI requests
NEWSAPI_BASE_URL = "https://newsapi.org/v2"  # NewsAPI v2 is the current stable version.

# Company symbols and their search keywords
# Each stock symbol maps to a NewsAPI search query with OR operators
# Including company names, products, and CEO names improves article relevance
COMPANIES = {  # Dictionary mapping stock symbols to NewsAPI search queries.
    "AAPL": "Apple Inc OR iPhone OR iPad OR Mac OR Tim Cook",  # Apple-related keywords.
    "MSFT": "Microsoft OR Windows OR Azure OR Satya Nadella",  # Microsoft-related keywords.
    "NVDA": "NVIDIA OR GeForce OR RTX OR Jensen Huang",  # NVIDIA-related keywords.
    "AMZN": "Amazon OR AWS OR Alexa OR Jeff Bezos OR Andy Jassy",  # Amazon-related keywords.
    "GOOGL": "Google OR Alphabet OR Android OR Sundar Pichai",  # Google/Alphabet-related keywords.
}  # End of COMPANIES dictionary.


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def eprint(msg: str) -> None:  # Helper function to print messages to standard error stream.
    """
    Print error message to stderr (standard error).
    
    Why stderr instead of stdout?
    - stdout is for program output (article data, results).
    - stderr is for diagnostic messages (logs, errors, progress).
    - Allows separation: python fetch_news.py > data.json 2> logs.txt
    
    Args:
        msg: The message string to print to stderr.
        
    Returns:
        None (void function with side effects only).
    """  # Docstring following Google-style format.
    print(msg, file=sys.stderr)  # print() with file= parameter redirects to specified stream.


def safe_mkdir(path: str) -> None:  # Create directory if it doesn't exist.
    """
    Create directory if it doesn't exist (similar to mkdir -p).
    
    This function is used to create company-specific folders for storing
    news articles. The exist_ok=True flag prevents errors if the directory
    already exists.
    
    Args:
        path: Directory path to create.
        
    Returns:
        None (creates directory as side effect).
    """  # Docstring explaining the function's purpose.
    os.makedirs(path, exist_ok=True)  # makedirs creates parent directories; exist_ok prevents errors.


# =============================================================================
# API KEY MANAGEMENT
# =============================================================================

def get_api_key() -> str:  # Retrieve API key from environment or file.
    """
    Get NewsAPI key from environment variable or credentials file.
    
    API Key Resolution Order:
    1. NEWSAPI_KEY environment variable (highest priority)
    2. cred/newsapi_credentials.json file
    3. Exit with error if not found
    
    This hierarchy allows:
    - CI/CD systems to use environment variables
    - Local development to use credentials file
    - Secure handling without hardcoding keys
    
    Credentials File Format:
        {"api_key": "your_newsapi_key_here"}
    
    Returns:
        str: NewsAPI API key string.
        
    Raises:
        SystemExit: If no API key is found (exits the program).
    """  # Docstring explaining the key resolution hierarchy.
    # Try environment variable first - highest priority
    api_key = os.environ.get('NEWSAPI_KEY')  # Returns None if not set.
    
    if not api_key:  # Environment variable not set, try credentials file.
        # Try reading from credentials file in cred/ directory
        cred_file = os.path.join(os.path.dirname(__file__), 'cred', 'newsapi_credentials.json')  # Build path relative to script.
        if os.path.exists(cred_file):  # Check if credentials file exists.
            try:  # Wrap file operations in try-except.
                with open(cred_file, 'r') as f:  # Open file for reading.
                    creds = json.load(f)  # Parse JSON content.
                    api_key = creds.get('api_key')  # Extract api_key field.
            except Exception as e:  # Catch JSON parse errors or file read errors.
                eprint(f"Error reading credentials file: {e}")  # Log error but continue.
    
    if not api_key:  # Neither environment variable nor file provided a key.
        eprint("ERROR: NewsAPI key not found!")  # Error message.
        eprint("Please set NEWSAPI_KEY environment variable or create cred/newsapi_credentials.json")  # Instructions.
        eprint('Example: {"api_key": "your_key_here"}')  # Example JSON format.
        eprint("\nGet your free API key at: https://newsapi.org/register")  # Registration link.
        sys.exit(1)  # Exit with error code 1.
    
    return api_key  # Return the retrieved API key.


# =============================================================================
# NEWS FETCHING FUNCTIONS
# =============================================================================

def fetch_company_news_newsapi(  # Main function to fetch news for a single company.
    symbol: str,
    search_query: str,
    api_key: str,
    days: int = 15,
    max_results: int = 100
) -> List[Dict]:
    """
    Fetch news articles for a specific company using NewsAPI.
    
    This function performs the following steps:
    1. Calculate date range (from N days ago to today).
    2. Make API request to NewsAPI /everything endpoint.
    3. For each article, attempt to extract full text using newspaper4k.
    4. Return enriched articles with full content.
    
    NewsAPI /everything Endpoint:
    - Searches all articles (not just headlines)
    - Supports complex queries with OR/AND operators
    - Returns up to 100 articles per request
    - Sorted by publishedAt (most recent first)
    
    newspaper4k Integration:
    - NewsAPI only returns partial content (first ~200 chars)
    - newspaper4k extracts full article text from the URL
    - Falls back to partial content if extraction fails
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL') for logging.
        search_query: NewsAPI search query string with keywords.
        api_key: Valid NewsAPI API key.
        days: Number of days to look back (default: 15).
        max_results: Maximum articles to retrieve (default: 100, max: 100).
        
    Returns:
        List[Dict]: List of enriched article dictionaries with fields:
            - title: Article headline
            - description: Brief description/snippet
            - published_date: ISO 8601 timestamp
            - url: Full article URL
            - source: Publisher name
            - author: Author name (if available)
            - image_url: Featured image URL (if available)
            - text: Full article text (extracted via newspaper4k)
    """  # Comprehensive docstring explaining the function's workflow.
    print(f"Fetching news for {symbol}...")  # Progress message to stdout.
    
    # Calculate date range for the query
    to_date = datetime.now()  # End date is today.
    from_date = to_date - timedelta(days=days)  # Start date is N days ago.
    
    # Prepare API request parameters
    # NewsAPI /everything endpoint accepts these query parameters
    url = f"{NEWSAPI_BASE_URL}/everything"  # Full URL for the everything endpoint.
    params = {  # Dictionary of query parameters for the API request.
        'q': search_query,  # Search query with OR operators for broader matching.
        'from': from_date.strftime('%Y-%m-%d'),  # Start date in YYYY-MM-DD format.
        'to': to_date.strftime('%Y-%m-%d'),  # End date in YYYY-MM-DD format.
        'language': 'en',  # English language articles only.
        'sortBy': 'publishedAt',  # Sort by publication date (most recent first).
        'pageSize': min(max_results, 100),  # NewsAPI max is 100 per request.
        'apiKey': api_key  # API key for authentication.
    }  # End of params dictionary.
    
    try:  # Wrap API call in try-except for error handling.
        response = requests.get(url, params=params, timeout=30)  # Make GET request with 30s timeout.
        response.raise_for_status()  # Raise HTTPError for 4xx/5xx status codes.
        data = response.json()  # Parse JSON response body.
        
        # Check if API returned success status
        if data.get('status') != 'ok':  # NewsAPI returns status field.
            eprint(f"API Error: {data.get('message', 'Unknown error')}")  # Log error message.
            return []  # Return empty list on API error.
        
        # Extract articles list from response
        articles = data.get('articles', [])  # Get articles array (may be empty).
        
        if not articles:  # No articles found for this query.
            print(f"  No articles found for {symbol}")  # Info message.
            return []  # Return empty list.
        
        print(f"  Found {len(articles)} articles from NewsAPI")  # Log article count.
        
        # Enrich articles with full content using newspaper4k
        # NewsAPI only provides partial content (~200 chars truncated)
        enriched_articles = []  # List to store enriched article dictionaries.
        for i, article in enumerate(articles, 1):  # Iterate with 1-based index.
            print(f"  Processing article {i}/{len(articles)}: {article.get('title', 'N/A')[:50]}...")  # Progress.
            
            article_url = article.get('url', '')  # Get article URL for full-text extraction.
            full_text = article.get('content', '')  # NewsAPI provides partial content.
            
            # Try to fetch full article content using newspaper4k
            if article_url:  # Only attempt if URL is available.
                try:  # Wrap newspaper4k calls in try-except.
                    news_article = Article(article_url)  # Create Article object with URL.
                    news_article.download()  # Download HTML content from URL.
                    news_article.parse()  # Parse HTML to extract article components.
                    
                    # Use newspaper4k text if it's longer than NewsAPI's content
                    if news_article.text and len(news_article.text) > len(full_text):  # Compare text lengths.
                        full_text = news_article.text  # Use the fuller text from newspaper4k.
                        print(f"    ✓ Extracted {len(full_text)} characters from full article")  # Success indicator.
                    else:  # newspaper4k didn't get more content.
                        print(f"    → Using NewsAPI content ({len(full_text)} characters)")  # Using partial.
                    
                    # Small delay to be polite to the target website
                    # Prevents hammering the server with rapid requests
                    time.sleep(0.3)  # 300ms delay between article fetches.
                    
                except Exception as e:  # Catch network errors, parsing errors, etc.
                    print(f"    ⚠ Could not fetch full article, using NewsAPI content")  # Warning message.
            
            # Build normalized article dictionary with consistent field names
            article_data = {  # Dictionary with all article metadata.
                'title': article.get('title', ''),  # Article headline.
                'description': article.get('description', ''),  # Brief snippet/description.
                'published_date': article.get('publishedAt', ''),  # ISO 8601 timestamp.
                'url': article_url,  # Full article URL.
                'source': article.get('source', {}).get('name', ''),  # Publisher name (nested in source object).
                'author': article.get('author', ''),  # Author name (may be None).
                'image_url': article.get('urlToImage', ''),  # Featured image URL.
                'text': full_text,  # Full article text (from newspaper4k or NewsAPI).
            }  # End of article_data dictionary.
            
            enriched_articles.append(article_data)  # Add to list of processed articles.
        
        print(f"Successfully processed {len(enriched_articles)} articles for {symbol}")  # Summary message.
        return enriched_articles  # Return list of enriched article dictionaries.
        
    except requests.exceptions.RequestException as e:  # Catch network/HTTP errors.
        eprint(f"Error fetching news for {symbol}: {e}")  # Log error.
        return []  # Return empty list on error.
    except Exception as e:  # Catch any other unexpected errors.
        eprint(f"Unexpected error for {symbol}: {e}")  # Log error.
        return []  # Return empty list on error.


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_articles_to_json(articles: List[Dict], output_dir: str, symbol: str) -> None:  # Save articles to JSON file.
    """
    Save articles to a JSON file in the designated company folder.
    
    This function creates a company-specific folder and saves all articles
    to a timestamped JSON file for later analysis by stock_analysis_agent.py.
    
    Output Structure:
        {
            "symbol": "AAPL",
            "source": "NewsAPI",
            "fetch_date": "20260124",
            "article_count": 25,
            "articles": [...]
        }
    
    Filename Format: {SYMBOL}_news_newsapi_{YYYYMMDD}.json
    Example: outputs/AAPL/AAPL_news_newsapi_20260124.json
    
    Args:
        articles: List of enriched article dictionaries.
        output_dir: Base output directory (e.g., 'outputs').
        symbol: Stock ticker symbol for folder and filename.
        
    Returns:
        None (saves file as side effect).
    """  # Docstring explaining output format.
    if not articles:  # Check if there are articles to save.
        eprint(f"No articles to save for {symbol}")  # Log warning.
        return  # Exit early if nothing to save.
    
    # Create company-specific folder under output directory
    company_folder = os.path.join(output_dir, symbol)  # Build path like 'outputs/AAPL'.
    safe_mkdir(company_folder)  # Create folder if it doesn't exist.
    
    # Generate filename with current date for versioning
    today = datetime.now().strftime("%Y%m%d")  # Format: YYYYMMDD (e.g., 20260124).
    filename = f"{symbol}_news_newsapi_{today}.json"  # e.g., 'AAPL_news_newsapi_20260124.json'.
    filepath = os.path.join(company_folder, filename)  # Full path to output file.
    
    # Save to JSON with pretty formatting
    try:  # Wrap file operations in try-except.
        with open(filepath, 'w', encoding='utf-8') as f:  # Open file for writing in UTF-8.
            json.dump({  # Write JSON with metadata wrapper.
                'symbol': symbol,  # Stock symbol for reference.
                'source': 'NewsAPI',  # Data source identifier.
                'fetch_date': today,  # When the data was fetched.
                'article_count': len(articles),  # Number of articles included.
                'articles': articles  # List of article dictionaries.
            }, f, indent=2, ensure_ascii=False)  # indent=2 for readability; ensure_ascii=False for unicode.
        
        print(f"Saved {len(articles)} articles to {filepath}")  # Success message.
        
    except Exception as e:  # Catch file write errors.
        eprint(f"Error saving articles for {symbol}: {e}")  # Log error.


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():  # Main function orchestrating the news fetching workflow.
    """
    Main execution function for the NewsAPI article fetcher.
    
    This function orchestrates the entire news fetching pipeline:
    1. Parse command-line arguments for configuration.
    2. Load API key from environment or credentials file.
    3. Validate the requested stock symbols.
    4. For each company: fetch articles and save to JSON.
    5. Print completion summary.
    
    Returns:
        None (implicit). Exits via sys.exit() on error.
    """  # Docstring explaining the main function's workflow.
    
    # Create argument parser for command-line interface
    parser = argparse.ArgumentParser(  # Initialize argument parser.
        description="Fetch news articles for selected companies using NewsAPI.org"  # Description for --help.
    )  # End of ArgumentParser initialization.
    
    # Define --output-dir argument for output directory
    parser.add_argument(  # Add output directory argument.
        "--output-dir",  # Where news JSON files will be saved.
        type=str,  # String type.
        default="outputs",  # Default to 'outputs' subdirectory.
        help="Output directory for news articles (default: outputs)"  # Help text.
    )  # End of --output-dir argument.
    
    # Define --days argument for lookback period
    parser.add_argument(  # Add days argument.
        "--days",  # Number of days to look back for news.
        type=int,  # Integer type.
        default=15,  # 15 days is a reasonable default.
        help="Number of days to look back (default: 15)"  # Help text.
    )  # End of --days argument.
    
    # Define --max-results argument for article limit
    parser.add_argument(  # Add max results argument.
        "--max-results",  # Maximum articles to fetch per company.
        type=int,  # Integer type.
        default=100,  # 100 is NewsAPI's max per request.
        help="Maximum number of articles per company (default: 100, max: 100)"  # Help text.
    )  # End of --max-results argument.
    
    # Define --symbols argument for selecting companies
    parser.add_argument(  # Add symbols argument.
        "--symbols",  # Which stock symbols to fetch news for.
        nargs='+',  # Accept one or more values as a list.
        default=list(COMPANIES.keys()),  # Default to all defined companies.
        help=f"Stock symbols to fetch news for (default: {' '.join(COMPANIES.keys())})"  # Help text.
    )  # End of --symbols argument.
    
    # Define --api-key argument for direct API key input
    parser.add_argument(  # Add API key argument.
        "--api-key",  # Direct API key input (overrides env/file).
        type=str,  # String type.
        help="NewsAPI key (or set NEWSAPI_KEY environment variable)"  # Help text.
    )  # End of --api-key argument.
    
    # Parse command-line arguments
    args = parser.parse_args()  # Returns Namespace object with all argument values.
    
    # Get API key (from CLI arg, env var, or credentials file)
    api_key = args.api_key or get_api_key()  # CLI arg takes priority; get_api_key() checks env/file.
    
    # ==========================================================================
    # SYMBOL VALIDATION
    # ==========================================================================
    
    # Validate symbols - check against defined COMPANIES dictionary
    invalid_symbols = [s for s in args.symbols if s not in COMPANIES]  # Find unknown symbols.
    if invalid_symbols:  # Warn about unknown symbols.
        eprint(f"Warning: Unknown symbols will be skipped: {', '.join(invalid_symbols)}")  # Log warning.
    
    valid_symbols = [s for s in args.symbols if s in COMPANIES]  # Keep only valid symbols.
    if not valid_symbols:  # No valid symbols to process.
        eprint("Error: No valid symbols provided")  # Log error.
        sys.exit(1)  # Exit with error code 1.
    
    # ==========================================================================
    # PRINT STARTUP BANNER
    # ==========================================================================
    
    # Print startup information
    print("=" * 70)  # Separator line.
    print("NewsAPI Article Fetcher")  # Script title.
    print("=" * 70)  # Separator line.
    print(f"Companies: {len(valid_symbols)}")  # Number of companies to process.
    print(f"Date range: Last {args.days} days")  # Lookback period.
    print(f"Max articles per company: {args.max_results}")  # Article limit.
    print(f"Output directory: {args.output_dir}")  # Where files will be saved.
    print("=" * 70)  # Separator line.
    
    # ==========================================================================
    # MAIN FETCH LOOP
    # ==========================================================================
    
    # Fetch news for each company in the valid symbols list
    for symbol in valid_symbols:  # Iterate through symbols.
        search_query = COMPANIES[symbol]  # Get search query for this symbol.
        articles = fetch_company_news_newsapi(  # Fetch articles from NewsAPI.
            symbol=symbol,  # Stock symbol for logging.
            search_query=search_query,  # Search query with keywords.
            api_key=api_key,  # API key for authentication.
            days=args.days,  # Lookback period.
            max_results=args.max_results  # Maximum articles to fetch.
        )  # End of function call.
        
        # Save articles to JSON file
        save_articles_to_json(articles, args.output_dir, symbol)  # Save to company folder.
        print("-" * 70)  # Visual separator between companies.
        
        # Small delay between companies to respect rate limits
        # NewsAPI has rate limits; being polite prevents 429 errors
        time.sleep(1)  # 1 second delay between companies.
    
    # ==========================================================================
    # PRINT COMPLETION BANNER
    # ==========================================================================
    
    # Print completion message
    print("=" * 70)  # Separator line.
    print("News fetch completed!")  # Success message.
    print("=" * 70)  # Separator line.


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

# Standard Python idiom: only execute main() when script is run directly
if __name__ == "__main__":  # True only when script is executed directly (not imported).
    main()  # Call main function to start execution.
