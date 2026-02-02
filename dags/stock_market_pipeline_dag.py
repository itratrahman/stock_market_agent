#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stock Market Pipeline DAG for Apache Airflow.

This DAG orchestrates the 5-stage stock market analysis pipeline:
1. pull_latest_stock.py   - Fetch historical stock data from FMP API
2. train_models.py        - Train NeuralProphet forecasting models
3. generate_forecasts.py  - Generate 30-day price predictions
4. fetch_news_newsapi.py  - Fetch news articles from NewsAPI
5. stock_analysis_agent.py - Run AI agent for investment recommendations

Each task runs sequentially. If any script fails (non-zero exit code),
the pipeline safely exits and subsequent tasks are skipped.

Requirements:
- Apache Airflow 2.x installed
- All pipeline scripts in the parent directory
- Proper credentials configured in cred/ directory
"""

# =============================================================================
# IMPORTS
# =============================================================================

import json  # Parse JSON configuration files.
from datetime import datetime, timedelta  # Date/time handling for DAG scheduling.
from pathlib import Path  # Cross-platform path handling.

from airflow import DAG  # Core DAG class for defining workflows.
from airflow.operators.bash import BashOperator  # Execute bash/shell commands.
from airflow.operators.python import PythonOperator  # Execute Python callables.


# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to the project directory (parent of dags folder).
PROJECT_DIR = Path(__file__).resolve().parent.parent  # Resolves to stock_market_agent/.

# Path to the Airflow configuration file containing Python executable path.
AIRFLOW_CONFIG_FILE = PROJECT_DIR / "cred" / "airflow_config.json"  # JSON config file.


def get_python_executable() -> str:
    """
    Read the absolute Python executable path from the Airflow config JSON file.
    
    Expected JSON format:
    {
        "python_executable": "/path/to/python"
    }
    
    Returns:
        str: Absolute path to the Python interpreter.
        
    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If 'python_executable' key is missing from the JSON.
    """
    if not AIRFLOW_CONFIG_FILE.exists():  # Check if config file exists.
        raise FileNotFoundError(
            f"Airflow config file not found: {AIRFLOW_CONFIG_FILE}\n"
            f"Please create it with: {{\"python_executable\": \"/path/to/python\"}}"
        )
    
    with open(AIRFLOW_CONFIG_FILE, "r", encoding="utf-8") as f:  # Open config file.
        config = json.load(f)  # Parse JSON content.
    
    if "python_executable" not in config:  # Validate required key exists.
        raise KeyError(
            f"Missing 'python_executable' key in {AIRFLOW_CONFIG_FILE}\n"
            f"Please add: {{\"python_executable\": \"/path/to/python\"}}"
        )
    
    return config["python_executable"]  # Return the Python executable path.


# Python interpreter path loaded from JSON config file.
PYTHON_EXECUTABLE = get_python_executable()  # Absolute path to Python interpreter.

# Default arguments applied to all tasks in the DAG.
default_args = {
    "owner": "airflow",  # Owner of the DAG for filtering in UI.
    "depends_on_past": False,  # Tasks don't depend on previous run's success.
    "email_on_failure": False,  # Disable email notifications on failure.
    "email_on_retry": False,  # Disable email notifications on retry.
    "retries": 0,  # No retries - fail fast for debugging.
    "retry_delay": timedelta(minutes=5),  # Delay between retries (if enabled).
}


# =============================================================================
# DAG DEFINITION
# =============================================================================

# Define the DAG with scheduling and configuration.
with DAG(
    dag_id="stock_market_pipeline",  # Unique identifier for the DAG.
    default_args=default_args,  # Apply default arguments to all tasks.
    description="Sequential pipeline for stock market analysis with AI agent",
    schedule_interval=None,  # Manual trigger only (no automatic scheduling).
    start_date=datetime(2026, 1, 1),  # DAG start date (historical runs won't execute).
    catchup=False,  # Don't backfill missed runs.
    max_active_runs=1,  # Only one instance of this DAG can run at a time.
    tags=["stock-market", "ml-pipeline", "neuralprophet", "langgraph"],  # Tags for filtering in UI.
) as dag:
    
    # =========================================================================
    # TASK 0: CLEANUP PREVIOUS RUN DATA
    # =========================================================================
    # Cleans up data/, models/, and outputs/ directories before starting.
    # Preserves README.md files. Uses --confirm flag for actual deletion.
    
    cleanup_data = BashOperator(
        task_id="cleanup_previous_data",  # Unique task identifier.
        bash_command=f"cd {PROJECT_DIR} && {PYTHON_EXECUTABLE} cleanup.py --confirm",
        doc_md="""
        ### Cleanup Previous Run Data
        
        Removes all files and directories from data/, models/, and outputs/
        folders to ensure a clean state before running the pipeline.
        
        **Script:** `cleanup.py`
        **Targets:** `data/`, `models/`, `outputs/`
        **Preserves:** README.md files in each directory
        """,
    )
    
    # =========================================================================
    # TASK 1: PULL LATEST STOCK DATA
    # =========================================================================
    # Fetches 5 years of daily OHLCV data from Financial Modeling Prep API.
    # Saves CSV files in NeuralProphet format (ds, y columns) to data/ directory.
    
    pull_stock_data = BashOperator(
        task_id="pull_latest_stock_data",  # Unique task identifier.
        bash_command=f"cd {PROJECT_DIR} && {PYTHON_EXECUTABLE} pull_latest_stock.py",
        doc_md="""
        ### Pull Latest Stock Data
        
        Fetches historical stock data from Financial Modeling Prep API.
        
        **Script:** `pull_latest_stock.py`
        **Output:** `data/*.csv` (AAPL, MSFT, NVDA, AMZN, GOOGL)
        **Format:** NeuralProphet-ready (ds, y columns)
        """,
    )
    
    # =========================================================================
    # TASK 2: TRAIN NEURALPROPHET MODELS
    # =========================================================================
    # Trains one NeuralProphet model per stock symbol.
    # Uses 100 epochs with yearly seasonality and 10 changepoints.
    
    train_models = BashOperator(
        task_id="train_forecasting_models",  # Unique task identifier.
        bash_command=f"cd {PROJECT_DIR} && {PYTHON_EXECUTABLE} train_models.py",
        doc_md="""
        ### Train Forecasting Models
        
        Trains NeuralProphet models on historical stock data.
        
        **Script:** `train_models.py`
        **Input:** `data/*.csv`
        **Output:** `models/*_neuralprophet`
        **Logs:** `lightning_logs/{SYMBOL}/`
        """,
    )
    
    # =========================================================================
    # TASK 3: GENERATE PRICE FORECASTS
    # =========================================================================
    # Loads trained models and generates 30-day price predictions.
    # Includes upper/lower 95% confidence bounds.
    
    generate_forecasts = BashOperator(
        task_id="generate_price_forecasts",  # Unique task identifier.
        bash_command=f"cd {PROJECT_DIR} && {PYTHON_EXECUTABLE} generate_forecasts.py",
        doc_md="""
        ### Generate Price Forecasts
        
        Generates 30-day price predictions using trained models.
        
        **Script:** `generate_forecasts.py`
        **Input:** `models/*_neuralprophet`, `data/*.csv`
        **Output:** `outputs/*_forecast_30d_*.csv`
        **Columns:** date, predicted_price, lower_bound, upper_bound
        """,
    )
    
    # =========================================================================
    # TASK 4: FETCH NEWS ARTICLES
    # =========================================================================
    # Retrieves recent news articles from NewsAPI for each stock symbol.
    # Extracts full article text using newspaper4k.
    
    fetch_news = BashOperator(
        task_id="fetch_news_articles",  # Unique task identifier.
        bash_command=f"cd {PROJECT_DIR} && {PYTHON_EXECUTABLE} fetch_news_newsapi.py",
        doc_md="""
        ### Fetch News Articles
        
        Retrieves news articles from NewsAPI with full-text extraction.
        
        **Script:** `fetch_news_newsapi.py`
        **Output:** `outputs/{SYMBOL}/*_news_newsapi_*.json`
        **Features:** Title, description, full text, author, source, images
        """,
    )
    
    # =========================================================================
    # TASK 5: RUN STOCK ANALYSIS AGENT
    # =========================================================================
    # Executes the LangGraph-based AI agent to analyze forecasts and news.
    # Generates INVEST/AVOID/NEUTRAL recommendations with reasoning.
    
    run_analysis_agent = BashOperator(
        task_id="run_stock_analysis_agent",  # Unique task identifier.
        bash_command=f"cd {PROJECT_DIR} && {PYTHON_EXECUTABLE} stock_analysis_agent.py",
        doc_md="""
        ### Run Stock Analysis Agent
        
        Executes AI agent for investment recommendations.
        
        **Script:** `stock_analysis_agent.py`
        **Input:** `outputs/*_forecast_*.csv`, `outputs/{SYMBOL}/*.json`
        **Output:** `outputs/stock_analysis_report_*.txt`
        **LLM:** Ollama with Llama 3.2 (local inference)
        """,
    )
    
    # =========================================================================
    # TASK DEPENDENCIES (SEQUENTIAL EXECUTION)
    # =========================================================================
    # Define the execution order using bitshift operators.
    # If any task fails, downstream tasks are automatically skipped.
    
    cleanup_data >> pull_stock_data >> train_models >> generate_forecasts >> fetch_news >> run_analysis_agent
