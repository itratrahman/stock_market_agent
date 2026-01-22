# Stock Market Agent

AI-powered stock market analysis and trading agent.

## Project Structure

```
stock_market_agent/
├── cred/                    # Credential files (not tracked in git)
│   └── credentials.json     # API keys and secrets
├── data/                    # Stock data CSV files (AAPL, AMZN, GOOGL, MSFT, NVDA)
├── models/                  # Trained NeuralProphet models
├── lightning_logs/          # Training logs organized by stock symbol
├── tests/                   # Unit tests
├── scripts/                 # Utility scripts
├── outputs/                 # Model outputs and results
├── pull_latest_stock.py     # Fetch stock data from FMP API
├── train_models.py          # Train NeuralProphet models
└── requirements.txt         # Python dependencies
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure API credentials:
   - Copy `cred/credentials.json.example` to `cred/credentials.json`
   - Add your Financial Modeling Prep API key to the credentials file

```json
{
  "FMP_API_KEY": "your_api_key_here"
}
```

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
