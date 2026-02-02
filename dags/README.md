# Airflow DAGs

This directory contains Apache Airflow DAG definitions for orchestrating the stock market analysis pipeline.

## DAG: stock_market_pipeline

**File:** `stock_market_pipeline_dag.py`

A sequential pipeline that runs all 6 stages of the stock market analysis workflow. If any stage fails, the pipeline safely exits and skips downstream tasks.

### Pipeline Stages

```
┌─────────────────────────┐
│ 0. cleanup.py           │  Clean data/, models/, outputs/ (keeps READMEs)
└───────────┬─────────────┘
            │ (on success)
            ▼
┌─────────────────────────┐
│ 1. pull_latest_stock.py │  Fetch 5yr OHLCV data from FMP API
└───────────┬─────────────┘
            │ (on success)
            ▼
┌─────────────────────────┐
│ 2. train_models.py      │  Train NeuralProphet models per stock
└───────────┬─────────────┘
            │ (on success)
            ▼
┌─────────────────────────┐
│ 3. generate_forecasts.py│  Generate 30-day price predictions
└───────────┬─────────────┘
            │ (on success)
            ▼
┌─────────────────────────┐
│ 4. fetch_news_newsapi.py│  Fetch news articles from NewsAPI
└───────────┬─────────────┘
            │ (on success)
            ▼
┌─────────────────────────┐
│ 5. stock_analysis_agent │  Run AI agent for recommendations
└─────────────────────────┘
```

### Error Handling

- **Fail-Fast:** If any script returns a non-zero exit code, the task fails
- **Skip Downstream:** Failed tasks cause all downstream tasks to be skipped
- **No Retries:** `retries=0` ensures immediate failure notification
- **Single Run:** `max_active_runs=1` prevents concurrent execution
- **Clean State:** Cleanup runs first to ensure fresh data each run

### Setup

1. **Install Airflow:**
   ```bash
   pip install apache-airflow
   ```

2. **Configure Airflow Home:**
   ```bash
   export AIRFLOW_HOME=~/airflow
   airflow db init
   ```

3. **Link DAGs Directory:**
   ```bash
   # Option A: Symlink this dags folder
   ln -s /path/to/stock_market_agent/dags $AIRFLOW_HOME/dags/stock_market
   
   # Option B: Copy DAG file
   cp stock_market_pipeline_dag.py $AIRFLOW_HOME/dags/
   ```

4. **Start Airflow:**
   ```bash
   airflow webserver --port 8080 &
   airflow scheduler &
   ```

5. **Access UI:** Open http://localhost:8080

### Manual Trigger

The DAG is configured for manual trigger only (`schedule_interval=None`).

**Via UI:**
1. Navigate to http://localhost:8080
2. Find `stock_market_pipeline` in the DAG list
3. Toggle the DAG to "On"
4. Click "Trigger DAG" (play button)

**Via CLI:**
```bash
airflow dags trigger stock_market_pipeline
```

### Configuration

The DAG reads the Python executable path from a JSON config file.

**Create `cred/airflow_config.json`:**
```json
{
    "python_executable": "C:/path/to/your/python.exe"
}
```

**Finding your Python path:**
```bash
# Windows PowerShell
(Get-Command python).Source

# Linux/Mac
which python3
```

**Other configurable options** in `stock_market_pipeline_dag.py`:

| Variable | Description | Default |
|----------|-------------|---------|
| `PYTHON_EXECUTABLE` | Python interpreter path | From `cred/airflow_config.json` |
| `PROJECT_DIR` | Path to project root | Auto-detected |
| `schedule_interval` | Cron schedule or None | `None` (manual) |
| `retries` | Number of retry attempts | `0` |

### Scheduling Examples

To enable automatic scheduling, modify `schedule_interval`:

```python
# Daily at midnight
schedule_interval="0 0 * * *"

# Every Monday at 6 AM
schedule_interval="0 6 * * 1"

# Every 4 hours
schedule_interval="0 */4 * * *"
```

### Prerequisites

Before running the DAG, ensure:

1. ✅ FMP API credentials in `cred/credentials.json`
2. ✅ NewsAPI key in env var or `cred/newsapi_credentials.json`
3. ✅ Python executable path in `cred/airflow_config.json`
4. ✅ Ollama running with `llama3.2` model installed
5. ✅ All Python dependencies installed (`pip install -r requirements.txt`)
