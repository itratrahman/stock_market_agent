#!/usr/bin/env python3  # Shebang line: tells Unix/Linux systems to use Python 3 interpreter from PATH.
# -*- coding: utf-8 -*-  # PEP 263: Declares source file encoding as UTF-8 for international characters.

"""
Generate stock price forecasts using trained NeuralProphet models.
Load historical data from 'data/' directory, use models from 'models/' directory,
and save forecasts to 'outputs/' directory.

This script is the third stage in the stock analysis pipeline:
1. pull_latest_stock.py - Fetches historical stock data from FMP API
2. train_models.py - Trains NeuralProphet models on historical data
3. generate_forecasts.py (THIS) - Uses trained models to predict future prices
4. fetch_news_newsapi.py - Retrieves recent news articles
5. stock_analysis_agent.py - Analyzes forecasts and news to make recommendations

Workflow:
1. Discover CSV files in data/ directory matching the pattern.
2. For each CSV, load the corresponding trained model from models/.
3. Generate N-day price forecasts using model.predict().
4. Save forecast results with confidence intervals to outputs/.

Output Format (CSV):
    date,predicted_price,lower_bound,upper_bound
    2026-01-22,245.67,233.39,257.95
    ...

Usage:
    python generate_forecasts.py --data-dir data --model-dir models --output-dir outputs --periods 30

Requirements:
    - Trained models in models/<SYMBOL>_neuralprophet/
    - Historical CSV data in data/<SYMBOL>_daily_5y.csv
"""  # Module docstring explaining the script's purpose, pipeline position, and workflow.

# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================

import argparse  # Built-in module for parsing command-line arguments (--periods, --data-dir, etc.).
import glob  # Built-in module for Unix-style pathname pattern matching (find files like *.csv).
import os  # Built-in module for operating system interactions (file paths, directory operations).
import sys  # Built-in module for system-specific parameters (stderr, exit codes).
from datetime import datetime  # Built-in class for date/time operations (timestamps for filenames).
from pathlib import Path  # Object-oriented filesystem path handling (modern alternative to os.path).
from typing import Optional, Tuple  # Type hints: Optional means value can be None, Tuple for multiple returns.

# =============================================================================
# THIRD-PARTY LIBRARY IMPORTS
# =============================================================================

import pandas as pd  # Data manipulation library; pd is the conventional alias. Used for DataFrame operations.
from neuralprophet import load  # Function to deserialize saved NeuralProphet models from disk.


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def eprint(msg: str) -> None:  # Helper function to print messages to standard error stream.
    """
    Print message to stderr (standard error).
    
    Why stderr instead of stdout?
    - stdout is for program output (forecast data, results).
    - stderr is for diagnostic messages (logs, errors, progress).
    - This separation allows piping output while still seeing logs.
    - Example: python generate_forecasts.py 2> logs.txt
    
    Args:
        msg: The message string to print to stderr.
        
    Returns:
        None (void function with side effects only).
    """  # Docstring following Google-style format.
    print(msg, file=sys.stderr)  # print() with file= parameter redirects to specified stream.


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_stock_data(csv_path: str) -> Optional[pd.DataFrame]:  # Load CSV and return DataFrame or None.
    """
    Load stock data from CSV file and prepare it for forecasting.
    
    This function loads the historical data that will be passed to the trained
    NeuralProphet model. The model needs recent history to make future predictions.
    
    NeuralProphet requires specific column names:
    - 'ds': The date/timestamp column (datestamp). Must be datetime type.
    - 'y': The target variable (closing price).
    
    Args:
        csv_path: Absolute or relative path to the CSV file.
        
    Returns:
        pd.DataFrame: DataFrame with validated columns ready for prediction,
                      or None if loading/validation fails.
                      
    Example:
        >>> df = load_stock_data('data/AAPL_daily_5y.csv')
        >>> print(df.shape)
        (1260, 7)  # ~5 years of daily data
    """  # Docstring explaining the function's role in the forecasting pipeline.
    try:  # Wrap file operations in try-except for graceful error handling.
        df = pd.read_csv(csv_path)  # pandas.read_csv() parses CSV into a DataFrame object.
        
        # Validate required columns - NeuralProphet MUST have 'ds' and 'y' columns
        if 'ds' not in df.columns or 'y' not in df.columns:  # Check if required columns exist.
            eprint(f"[ERROR] CSV missing required columns (ds, y): {csv_path}")  # Log error with file path.
            return None  # Return None to signal failure (Optional return type).
        
        # Ensure ds is datetime - NeuralProphet requires proper datetime objects
        df['ds'] = pd.to_datetime(df['ds'])  # Convert string dates to pandas Timestamp objects.
        
        # Sort by date - time series must be in chronological order for forecasting
        df = df.sort_values('ds').reset_index(drop=True)  # Sort ascending; reset_index removes old index.
        
        eprint(f"[INFO] Loaded {len(df)} rows from {os.path.basename(csv_path)}")  # Log success with row count.
        return df  # Return the cleaned and validated DataFrame.
    
    except Exception as ex:  # Catch any exception (file not found, parse errors, etc.).
        eprint(f"[ERROR] Failed to load {csv_path}: {ex}")  # Log the exception message.
        return None  # Return None to indicate failure.


# =============================================================================
# MODEL LOADING FUNCTIONS
# =============================================================================

def load_model(model_path: str):  # Load trained NeuralProphet model from disk.
    """
    Load a trained NeuralProphet model from disk.
    
    NeuralProphet models are saved as directories containing:
    - model.pt: PyTorch model weights (state dict)
    - config.json: Model configuration and hyperparameters
    
    The neuralprophet.load() function handles deserialization of both
    the model architecture and trained weights.
    
    Args:
        model_path: Path to the model directory (e.g., 'models/AAPL_neuralprophet').
        
    Returns:
        NeuralProphet: Loaded model ready for prediction, or None if loading fails.
        
    Note:
        The return type is not annotated because NeuralProphet type is imported
        only for loading, not for type hints.
    """  # Docstring explaining model serialization format.
    try:  # Wrap in try-except to handle corrupted models or missing files.
        model = load(model_path)  # neuralprophet.load() deserializes the saved model.
        eprint(f"[INFO] Loaded model from {model_path}")  # Log successful load.
        return model  # Return the loaded NeuralProphet model.
    
    except Exception as ex:  # Catch deserialization errors, file not found, etc.
        eprint(f"[ERROR] Failed to load model from {model_path}: {ex}")  # Log error with details.
        return None  # Return None to indicate failure.


# =============================================================================
# FORECAST GENERATION FUNCTIONS
# =============================================================================

def generate_forecast(model, df: pd.DataFrame, periods: int, symbol: str) -> Optional[pd.DataFrame]:  # Generate predictions.
    """
    Generate forecast using a trained NeuralProphet model.
    
    This function creates future predictions by:
    1. Creating a future DataFrame with dates extending beyond historical data.
    2. Running model.predict() to generate point forecasts and uncertainty intervals.
    
    NeuralProphet Prediction Process:
    - make_future_dataframe() extends the time index into the future.
    - n_historic_predictions includes past dates for continuity visualization.
    - predict() runs the neural network forward pass for each date.
    - Output includes 'yhat1' (prediction) and optional confidence bounds.
    
    Args:
        model: Trained NeuralProphet model object.
        df: Historical DataFrame with 'ds' and 'y' columns.
        periods: Number of future periods (days) to forecast.
        symbol: Stock ticker symbol for logging purposes.
        
    Returns:
        pd.DataFrame: Forecast DataFrame with predictions, or None if generation fails.
        Columns include: ds (date), yhat1 (prediction), and optional bounds.
    """  # Docstring explaining the NeuralProphet prediction workflow.
    try:  # Wrap prediction in try-except to handle model errors.
        eprint(f"[INFO] Generating {periods}-day forecast for {symbol}...")  # Log forecast start.
        
        # Make future dataframe - extends time series into the future
        # periods: how many future timesteps to predict
        # n_historic_predictions: include historical predictions for continuity
        future = model.make_future_dataframe(df[['ds', 'y']], periods=periods, n_historic_predictions=len(df))  # Create extended time index.
        
        # Generate forecast by running the trained model on the future dataframe
        forecast = model.predict(future)  # Returns DataFrame with predictions and components.
        
        eprint(f"[OK] Forecast generated for {symbol}")  # Log successful generation.
        return forecast  # Return the forecast DataFrame.
    
    except Exception as ex:  # Catch prediction errors (shape mismatch, etc.).
        eprint(f"[ERROR] Failed to generate forecast for {symbol}: {ex}")  # Log error with details.
        return None  # Return None to indicate failure.


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_forecast(forecast: pd.DataFrame, symbol: str, output_dir: str, periods: int) -> bool:  # Save forecast to CSV.
    """
    Save forecast to a CSV file with a timestamped filename.
    
    This function processes the raw NeuralProphet forecast output and creates
    a clean CSV file suitable for downstream analysis by the stock_analysis_agent.
    
    Output CSV Format:
        date,predicted_price,lower_bound,upper_bound
        2026-01-22,245.67,233.39,257.95
        2026-01-23,247.12,234.76,259.48
        ...
    
    Confidence Intervals:
    - If the model was trained with uncertainty estimation, uses actual bounds.
    - Otherwise, applies a simple ±5% heuristic for the bounds.
    
    Filename Format: {SYMBOL}_forecast_{PERIODS}d_{YYYYMMDD}.csv
    Example: AAPL_forecast_30d_20260121.csv
    
    Args:
        forecast: Raw NeuralProphet forecast DataFrame with predictions.
        symbol: Stock ticker symbol for filename.
        output_dir: Directory to save the forecast CSV.
        periods: Number of forecast periods (used in filename).
        
    Returns:
        bool: True if save was successful, False if any error occurred.
    """  # Docstring explaining output format and confidence interval handling.
    try:  # Wrap in try-except for handling disk errors.
        os.makedirs(output_dir, exist_ok=True)  # Create output directory; exist_ok prevents error if exists.
        
        # Get only future predictions (last 'periods' rows)
        # The forecast includes historical predictions; we only want future ones
        future_forecast = forecast.tail(periods).copy()  # .tail(n) gets last n rows; .copy() avoids warnings.
        
        # Format the forecast for output
        # Extract and rename columns for clarity
        output_df = pd.DataFrame({  # Create new DataFrame with renamed columns.
            'date': future_forecast['ds'].dt.strftime('%Y-%m-%d'),  # Format datetime as string.
            'predicted_price': future_forecast['yhat1'].round(2),  # Round predictions to 2 decimal places.
            'lower_bound': future_forecast.get('yhat1 - 95%', future_forecast['yhat1'] * 0.95).round(2),  # Use actual or estimated lower bound.
            'upper_bound': future_forecast.get('yhat1 + 95%', future_forecast['yhat1'] * 1.05).round(2),  # Use actual or estimated upper bound.
        })  # End of DataFrame construction.
        
        # Generate filename with timestamp for versioning
        timestamp = datetime.now().strftime('%Y%m%d')  # Format: YYYYMMDD (e.g., 20260121).
        output_path = os.path.join(output_dir, f"{symbol}_forecast_{periods}d_{timestamp}.csv")  # Build full path.
        
        # Save to CSV without the DataFrame index
        output_df.to_csv(output_path, index=False)  # index=False prevents writing row numbers.
        
        # Log save details and forecast summary statistics
        eprint(f"[OK] Saved forecast to {output_path}")  # Confirm file location.
        eprint(f"     First predicted date: {output_df['date'].iloc[0]}")  # Show forecast start date.
        eprint(f"     Last predicted date: {output_df['date'].iloc[-1]}")  # Show forecast end date.
        eprint(f"     Price range: ${output_df['predicted_price'].min():.2f} - ${output_df['predicted_price'].max():.2f}")  # Show price range.
        
        return True  # Return True to indicate success.
    
    except Exception as ex:  # Catch disk errors, permission issues, etc.
        eprint(f"[ERROR] Failed to save forecast for {symbol}: {ex}")  # Log error with details.
        return False  # Return False to indicate failure.


# =============================================================================
# FILENAME PARSING FUNCTIONS
# =============================================================================

def extract_symbol_from_filename(filename: str) -> str:  # Parse filename to get stock symbol.
    """
    Extract stock ticker symbol from CSV filename.
    
    This function follows the naming convention established by pull_latest_stock.py:
    - Format: {SYMBOL}_daily_{N}y.csv
    - Examples: 'AAPL_daily_5y.csv' -> 'AAPL'
                'data/NVDA_daily_5y.csv' -> 'NVDA'
    
    The function handles both full paths and just filenames by first
    extracting the basename.
    
    Args:
        filename: Full path or just filename of the CSV file.
        
    Returns:
        str: The stock ticker symbol (uppercase, first segment before underscore).
    """  # Docstring explaining the filename convention.
    basename = os.path.basename(filename)  # Extract filename from path: 'data/AAPL_daily_5y.csv' -> 'AAPL_daily_5y.csv'.
    symbol = basename.split('_')[0]  # Split by underscore, take first part: 'AAPL_daily_5y.csv' -> 'AAPL'.
    return symbol  # Return the extracted stock symbol.


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> int:  # Main function returning exit code (0 = success, non-zero = error).
    """
    Main entry point for the forecast generation script.
    
    This function orchestrates the entire forecast generation pipeline:
    1. Parse command-line arguments for configuration.
    2. Discover CSV files matching the specified pattern.
    3. For each stock: load data, load model, generate forecast, save results.
    4. Report summary of successful/failed generations.
    
    Exit Codes:
        0: All forecasts generated successfully.
        1: At least one forecast failed to generate.
        
    Returns:
        int: Exit code for the process (used by SystemExit).
    """  # Docstring explaining the main function's orchestration role.
    
    # Create argument parser for command-line interface
    parser = argparse.ArgumentParser(  # Initialize argument parser object.
        description="Generate stock price forecasts using trained NeuralProphet models."  # Description for --help.
    )  # End of ArgumentParser initialization.
    
    # Define --data-dir argument for input data directory
    parser.add_argument(  # Add a command-line argument definition.
        "--data-dir",  # Argument name (accessed as args.data_dir after parsing).
        default="data",  # Default value if not specified on command line.
        help="Directory containing stock CSV files"  # Help text shown in --help.
    )  # End of --data-dir argument.
    
    # Define --model-dir argument for trained models directory
    parser.add_argument(  # Add model directory argument.
        "--model-dir",  # Location of trained NeuralProphet models.
        default="models",  # Default to 'models' subdirectory.
        help="Directory containing trained models"  # Describe purpose in help.
    )  # End of --model-dir argument.
    
    # Define --output-dir argument for forecast output directory
    parser.add_argument(  # Add output directory argument.
        "--output-dir",  # Where forecast CSVs will be saved.
        default="outputs",  # Default to 'outputs' subdirectory.
        help="Directory to save forecast CSV files"  # Describe purpose in help.
    )  # End of --output-dir argument.
    
    # Define --periods argument for forecast horizon
    parser.add_argument(  # Add forecast periods argument.
        "--periods",  # Number of future days to predict.
        type=int,  # Enforce integer type (argparse validates this).
        default=30,  # 30 days is a reasonable default for short-term forecasting.
        help="Number of days to forecast (default: 30)"  # Explain what periods means.
    )  # End of --periods argument.
    
    # Define --pattern argument for file matching
    parser.add_argument(  # Add file pattern argument.
        "--pattern",  # Glob pattern to find CSV files.
        default="*_daily_*.csv",  # Matches files like 'AAPL_daily_5y.csv'.
        help="Glob pattern for CSV files to process"  # Explain glob patterns.
    )  # End of --pattern argument.
    
    # Parse command-line arguments
    args = parser.parse_args()  # Returns Namespace object with all argument values.
    
    # ==========================================================================
    # FILE DISCOVERY
    # ==========================================================================
    
    # Find all CSV files in data directory using glob pattern matching
    csv_pattern = os.path.join(args.data_dir, args.pattern)  # Build pattern like 'data/*_daily_*.csv'.
    csv_files = glob.glob(csv_pattern)  # Find all files matching the pattern; returns list of paths.
    
    # Validate that we found at least one CSV file to process
    if not csv_files:  # Empty list is falsy in Python.
        eprint(f"[ERROR] No CSV files found matching pattern: {csv_pattern}")  # Report error with pattern.
        return 1  # Return exit code 1 to indicate failure.
    
    # Log discovery results
    eprint(f"[INFO] Found {len(csv_files)} CSV files to process")  # Log number of files found.
    eprint(f"[INFO] Forecast period: {args.periods} days")  # Log the forecast horizon.
    eprint("")  # Empty line for visual separation.
    
    # ==========================================================================
    # MAIN FORECAST GENERATION LOOP
    # ==========================================================================
    
    # Process each stock - track success and failure counts
    success_count = 0  # Counter for successfully generated forecasts.
    fail_count = 0  # Counter for failed forecast attempts.
    
    # Iterate through each discovered CSV file
    for csv_file in csv_files:  # Loop through list of file paths.
        # Extract stock symbol from filename for logging and model lookup
        symbol = extract_symbol_from_filename(csv_file)  # e.g., 'data/AAPL_daily_5y.csv' -> 'AAPL'.
        
        # Print visual separator for each stock
        eprint(f"\n{'='*50}")  # Visual separator line.
        eprint(f"Processing {symbol}")  # Log current stock being processed.
        eprint(f"{'='*50}")  # Visual separator line.
        
        # Load historical data from CSV file
        df = load_stock_data(csv_file)  # Returns DataFrame or None on failure.
        if df is None:  # Check if loading failed.
            fail_count += 1  # Increment failure counter.
            continue  # Skip to next file; don't attempt forecasting.
        
        # Load trained model for this stock symbol
        model_path = os.path.join(args.model_dir, f"{symbol}_neuralprophet")  # Build model path.
        if not os.path.exists(model_path):  # Check if model directory exists.
            eprint(f"[ERROR] Model not found: {model_path}")  # Log missing model.
            fail_count += 1  # Increment failure counter.
            continue  # Skip to next file.
        
        model = load_model(model_path)  # Load the trained model from disk.
        if model is None:  # Check if loading failed.
            fail_count += 1  # Increment failure counter.
            continue  # Skip to next file.
        
        # Generate forecast using the loaded model and data
        forecast = generate_forecast(model, df, args.periods, symbol)  # Returns forecast DataFrame or None.
        if forecast is None:  # Check if generation failed.
            fail_count += 1  # Increment failure counter.
            continue  # Skip saving.
        
        # Save forecast to CSV file
        if save_forecast(forecast, symbol, args.output_dir, args.periods):  # Returns True on success.
            success_count += 1  # Increment success counter on successful save.
        else:  # Save failed.
            fail_count += 1  # Increment failure counter.
    
    # ==========================================================================
    # PRINT SUMMARY
    # ==========================================================================
    
    # Print summary of forecast generation results
    eprint("\n" + "="*50)  # Print newline and separator line.
    eprint(f"[SUMMARY] Forecast generation complete")  # Summary header.
    eprint(f"  Success: {success_count}/{len(csv_files)}")  # Report successful generations.
    eprint(f"  Failed:  {fail_count}/{len(csv_files)}")  # Report failed generations.
    eprint(f"  Output directory: {args.output_dir}")  # Report where forecasts were saved.
    eprint("="*50)  # Closing separator line.
    
    # Return appropriate exit code based on results
    return 0 if fail_count == 0 else 1  # Ternary expression: return 0 if all succeeded, else 1.


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

# Standard Python idiom: only execute main() when script is run directly
if __name__ == "__main__":  # True only when script is executed directly (not imported).
    raise SystemExit(main())  # Call main() and exit with its return code.
