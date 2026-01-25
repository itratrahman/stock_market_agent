# Stock Market Agent

AI-powered stock market analysis and trading agent with news sentiment analysis capabilities.

## Features

- 📈 **Stock Data Fetching**: Pull historical stock data from Financial Modeling Prep API
- 🤖 **Time Series Forecasting**: Train NeuralProphet models for stock prediction
- 📰 **News Article Retrieval**: Fetch and store news articles from NewsAPI.org
- 💾 **Data Storage**: Organized storage of stock data, models, and news articles

## Project Structure

```
stock_market_agent/
├── cred/                           # Credential files (not tracked in git)
│   ├── credentials.json            # FMP API key
│   └── newsapi_credentials.json    # NewsAPI key
├── data/                           # Stock data CSV files (AAPL, AMZN, GOOGL, MSFT, NVDA)
├── models/                         # Trained NeuralProphet models
├── lightning_logs/                 # Training logs organized by stock symbol
├── outputs/                        # Model outputs and news articles
│   ├── AAPL/                      # Apple news articles
│   ├── AMZN/                      # Amazon news articles
│   ├── GOOGL/                     # Google news articles
│   ├── MSFT/                      # Microsoft news articles
│   └── NVDA/                      # NVIDIA news articles
├── tests/                          # Unit tests
├── scripts/                        # Utility scripts
├── pull_latest_stock.py            # Fetch stock data from FMP API
├── train_models.py                 # Train NeuralProphet models
├── fetch_news_newsapi.py           # Fetch news articles using NewsAPI
└── requirements.txt                # Python dependencies
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Credentials

#### Stock Data API (Financial Modeling Prep)
Create `cred/credentials.json` with your FMP API key:

```json
{
  "FMP_API_KEY": "your_fmp_api_key_here"
}
```

Get your API key at: https://site.financialmodelingprep.com/developer/docs

#### News API (NewsAPI.org)
Set environment variable or create `cred/newsapi_credentials.json`:

**Option A: Environment Variable (Recommended)**
```bash
# Windows PowerShell
$env:NEWSAPI_KEY="your_newsapi_key_here"

# Linux/Mac
export NEWSAPI_KEY="your_newsapi_key_here"
```

**Option B: Credentials File**
Create `cred/newsapi_credentials.json`:
```json
{
  "api_key": "your_newsapi_key_here"
}
```

Get your free API key at: https://newsapi.org/register

## Usage

### Fetch Stock Data

Pull historical stock data from Financial Modeling Prep API:

```bash
python pull_latest_stock.py
```

Options:
- `--years N`: Fetch N years of historical data (default: 5)
- `--top-n N`: Number of companies to fetch (default: 5)
- `--outdir PATH`: Output directory for CSV files (default: data)
- `--cred-file PATH`: Path to credentials file (default: cred/credentials.json)
- `--api-key KEY`: Override API key from command line
- `--universe TICKERS`: Comma-separated list of tickers (default: AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA)

Example:
```bash
python pull_latest_stock.py --years 3 --top-n 7
```

The script will fetch stock data and save CSV files in the `data/` directory in NeuralProphet-ready format (ds, y columns).

### Fetch News Articles

Retrieve news articles for stock companies to support sentiment analysis using NewsAPI.org:

```bash
python fetch_news_newsapi.py
```

Options:
- `--output-dir PATH`: Output directory (default: outputs)
- `--days N`: Days to look back (default: 15, max: 30 for free tier)
- `--max-results N`: Max articles per company (default: 100, max: 100)
- `--symbols TICKERS`: Space-separated stock symbols (default: all)
- `--api-key KEY`: NewsAPI key (or use NEWSAPI_KEY env variable)

Example:
```bash
python fetch_news_newsapi.py --symbols AAPL MSFT --days 7 --max-results 50
```

**NewsAPI Features:**
- ✅ Official REST API with 70,000+ sources
- ✅ Rich metadata (author, source, images, partial content)
- ✅ Direct article URLs (no redirect resolution needed)
- ✅ 100 requests/day free tier
- ✅ Reliable and well-documented

**Output Format:**

Articles are saved as JSON files in company-specific folders:
```
outputs/
├── AAPL/
│   └── AAPL_news_newsapi_20260124.json
├── MSFT/
│   └── MSFT_news_newsapi_20260124.json
└── ...
```

Each JSON file contains:
```json
{
  "symbol": "AAPL",
  "source": "NewsAPI",
  "fetch_date": "20260124",
  "article_count": 85,
  "articles": [
    {
      "title": "Article Title",
      "description": "Brief description",
      "published_date": "2026-01-24T10:30:00Z",
      "url": "https://example.com/article",
      "source": "TechCrunch",
      "author": "John Doe",
      "image_url": "https://example.com/image.jpg",
      "text": "Full article body text..."
    }
  ]
}
```

### Train Models

Train NeuralProphet models on the stock data:

```bash
python train_models.py
```

Options:
- `--data-dir PATH`: Directory containing stock CSV files (default: data)
- `--model-dir PATH`: Directory to save trained models (default: models)
- `--epochs N`: Number of training epochs (default: 100)
- `--learning-rate RATE`: Learning rate (default: auto)
- `--n-changepoints N`: Number of potential changepoints (default: 10)
- `--yearly-seasonality`: Enable yearly seasonality (default: True)
- `--verbose`: Show training progress
- `--pattern GLOB`: Glob pattern for CSV files (default: *_daily_*.csv)

Example:
```bash
python train_models.py --epochs 150 --verbose
```

The script will train a separate model for each stock and save them as PyTorch files in the `models/` directory.

### View Training Logs

Training logs are organized by stock symbol in `lightning_logs/` directory:
```
lightning_logs/
├── AAPL/
├── AMZN/
├── GOOGL/
├── MSFT/
└── NVDA/
```

Visualize training metrics with TensorBoard:
```bash
tensorboard --logdir=lightning_logs
```

Then open http://localhost:6006 to view:
- Loss curves over training epochs
- Model performance metrics
- Training comparisons across different stocks
