# Stock Market Agent

AI-powered stock market analysis and trading agent with news sentiment analysis capabilities.

## 1. Features

- 📈 **Stock Data Fetching**: Pull historical stock data from Financial Modeling Prep API
- 🤖 **Time Series Forecasting**: Train NeuralProphet models for stock prediction
- 📊 **Forecast Generation**: Generate 30-day price forecasts using trained models
- 📰 **News Article Retrieval**: Fetch and store news articles from NewsAPI.org
- 🧠 **AI Analysis Agent**: LangGraph-based agentic workflow for investment recommendations
- 💾 **Data Storage**: Organized storage of stock data, models, and news articles

## 2. Project Structure

```
stock_market_agent/
├── cred/                           # Credential files (not tracked in git)
│   ├── credentials.json            # FMP API key
│   └── newsapi_credentials.json    # NewsAPI key
├── data/                           # Stock data CSV files (AAPL, AMZN, GOOGL, MSFT, NVDA)
├── models/                         # Trained NeuralProphet models
├── lightning_logs/                 # Training logs organized by stock symbol
├── outputs/                        # Model outputs, forecasts, and news articles
│   ├── *_forecast_30d_*.csv       # 30-day price forecasts
│   ├── stock_analysis_report_*.txt # AI agent analysis reports
│   ├── AAPL/                      # Apple news articles
│   ├── AMZN/                      # Amazon news articles
│   ├── GOOGL/                     # Google news articles
│   ├── MSFT/                      # Microsoft news articles
│   └── NVDA/                      # NVIDIA news articles
├── tests/                          # Unit tests
├── pull_latest_stock.py            # Fetch stock data from FMP API
├── train_models.py                 # Train NeuralProphet models
├── generate_forecasts.py           # Generate price forecasts
├── fetch_news_newsapi.py           # Fetch news articles using NewsAPI
├── stock_analysis_agent.py         # AI agent for investment analysis
└── requirements.txt                # Python dependencies
```

## 3. Setup

### 3.1 Install Dependencies

```bash
pip install -r requirements.txt
```

### 3.2 Configure API Credentials

#### 3.2.1 Stock Data API (Financial Modeling Prep)
Create `cred/credentials.json` with your FMP API key:

```json
{
  "FMP_API_KEY": "your_fmp_api_key_here"
}
```

Get your API key at: https://site.financialmodelingprep.com/developer/docs

#### 3.2.2 News API (NewsAPI.org)
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

#### 3.2.3 LLM API (Ollama - Local)
The stock analysis agent uses Ollama with Llama 3.2 for AI-powered analysis.

1. Install Ollama: https://ollama.com/download
2. Pull the model:
```bash
ollama pull llama3.2
```

## 4. Usage

### 4.1 Fetch Stock Data

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

### 4.2 Train Models

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

### 4.3 View Training Logs

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

### 4.4 Generate Forecasts

Generate 30-day price forecasts using trained models:

```bash
python generate_forecasts.py
```

Options:
- `--data-dir PATH`: Directory containing stock CSV files (default: data)
- `--model-dir PATH`: Directory containing trained models (default: models)
- `--output-dir PATH`: Directory to save forecast CSV files (default: outputs)
- `--periods N`: Number of days to forecast (default: 30)
- `--pattern GLOB`: Glob pattern for CSV files (default: *_daily_*.csv)

Example:
```bash
python generate_forecasts.py --periods 60
```

**Output Format:**
Forecasts are saved as CSV files in `outputs/`:
```
outputs/
├── AAPL_forecast_30d_20260121.csv
├── MSFT_forecast_30d_20260121.csv
└── ...
```

Each CSV contains:
- `date` - Forecast date
- `predicted_price` - Predicted closing price
- `lower_bound` - Lower 95% confidence interval
- `upper_bound` - Upper 95% confidence interval

### 4.5 Fetch News Articles

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

### 4.6 Run Stock Analysis Agent

Run the AI-powered analysis agent to get investment recommendations:

```bash
python stock_analysis_agent.py
```

**Prerequisites:**
1. Ollama running with llama3.2 model installed
2. Forecast CSV files in `outputs/` directory
3. News JSON files in `outputs/{SYMBOL}/` directories

**Agentic Flow:**

```
┌─────────────────────┐
│  Analyze Forecast   │  Node 1: Read CSV, create summary (<300 chars)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Decision: Quality? │  Is forecast promising?
└──────┬──────┬───────┘
       │      │
  YES  │      │ NO
       ▼      ▼
┌──────────┐ ┌────────────┐
│Summarize │ │ Skip News  │  Node 2: Process or skip news
│  News    │ │            │
└────┬─────┘ └─────┬──────┘
     │             │
     └──────┬──────┘
            ▼
┌─────────────────────┐
│  Investment Decision│  Node 3: INVEST / AVOID / NEUTRAL
└─────────┬───────────┘
          │
          ▼
        [END]
```

**Output:**
The agent produces a tabulated report saved to:
```
outputs/stock_analysis_report_YYYYMMDD_HHMMSS.txt
```

**Report Contents:**
- Summary table of all stocks with decisions
- Detailed analysis for each stock:
  - Forecast summary (<300 characters)
  - News summary (<1000 characters)
  - Investment decision (INVEST/AVOID/NEUTRAL)
  - Decision reasoning (<200 characters)

**Example Output:**
```
================================================================================
STOCK ANALYSIS AGENT REPORT
Generated: 2026-01-21 14:30:00
================================================================================

SUMMARY TABLE
--------------------------------------------------------------------------------
Symbol   Decision   Promising
--------------------------------------------------------------------------------
AAPL     INVEST     YES
MSFT     NEUTRAL    YES
NVDA     INVEST     YES
AMZN     AVOID      NO
GOOGL    NEUTRAL    YES
--------------------------------------------------------------------------------

DETAILED ANALYSIS
================================================================================

STOCK: AAPL
----------------------------------------

[FORECAST SUMMARY] (285 chars)
Apple stock predicted to rise 2.3% over 30 days, from $249.56 to $255.30...

[NEWS SUMMARY] (890 chars)
Apple reported record iPhone sales in India with 14M units shipped...

[INVESTMENT DECISION]
Recommendation: INVEST

[DECISION SUMMARY] (175 chars)
Strong forecast with positive news sentiment. Record sales in key markets...
================================================================================
```

## 5. Complete Workflow

Run the full pipeline:

```bash
# 1. Fetch latest stock data
python pull_latest_stock.py

# 2. Train forecasting models
python train_models.py

# 3. Generate price forecasts
python generate_forecasts.py

# 4. Fetch recent news articles
python fetch_news_newsapi.py

# 5. Run AI analysis agent
python stock_analysis_agent.py
```

## 6. Technologies Used

- **NeuralProphet**: Time series forecasting with neural networks
- **LangGraph**: Agentic workflow orchestration
- **LangChain**: LLM integration framework
- **Ollama + Llama 3.2**: Open-source LLM for analysis
- **NewsAPI**: News article retrieval
- **Pandas**: Data manipulation and analysis

## 7. License

MIT License
