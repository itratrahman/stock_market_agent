#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate stock price forecasts using trained NeuralProphet models.
Load historical data from 'data/' directory, use models from 'models/' directory,
and save forecasts to 'outputs/' directory.
"""

import argparse
import glob
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from neuralprophet import load


def eprint(msg: str) -> None:
    """Print message to stderr."""
    print(msg, file=sys.stderr)


def load_stock_data(csv_path: str) -> Optional[pd.DataFrame]:
    """Load stock data from CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with stock data or None if loading fails
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        if 'ds' not in df.columns or 'y' not in df.columns:
            eprint(f"[ERROR] CSV missing required columns (ds, y): {csv_path}")
            return None
        
        # Ensure ds is datetime
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Sort by date
        df = df.sort_values('ds').reset_index(drop=True)
        
        eprint(f"[INFO] Loaded {len(df)} rows from {os.path.basename(csv_path)}")
        return df
    
    except Exception as ex:
        eprint(f"[ERROR] Failed to load {csv_path}: {ex}")
        return None


def load_model(model_path: str):
    """Load a trained NeuralProphet model.
    
    Args:
        model_path: Path to model directory
        
    Returns:
        Loaded NeuralProphet model or None if loading fails
    """
    try:
        model = load(model_path)
        eprint(f"[INFO] Loaded model from {model_path}")
        return model
    
    except Exception as ex:
        eprint(f"[ERROR] Failed to load model from {model_path}: {ex}")
        return None


def generate_forecast(model, df: pd.DataFrame, periods: int, symbol: str) -> Optional[pd.DataFrame]:
    """Generate forecast using trained model.
    
    Args:
        model: Trained NeuralProphet model
        df: Historical data DataFrame
        periods: Number of periods to forecast
        symbol: Stock symbol for logging
        
    Returns:
        DataFrame with forecast or None if generation fails
    """
    try:
        eprint(f"[INFO] Generating {periods}-day forecast for {symbol}...")
        
        # Make future dataframe
        future = model.make_future_dataframe(df[['ds', 'y']], periods=periods, n_historic_predictions=len(df))
        
        # Generate forecast
        forecast = model.predict(future)
        
        eprint(f"[OK] Forecast generated for {symbol}")
        return forecast
    
    except Exception as ex:
        eprint(f"[ERROR] Failed to generate forecast for {symbol}: {ex}")
        return None


def save_forecast(forecast: pd.DataFrame, symbol: str, output_dir: str, periods: int) -> bool:
    """Save forecast to CSV file.
    
    Args:
        forecast: Forecast DataFrame
        symbol: Stock symbol for filename
        output_dir: Directory to save forecast
        periods: Number of forecast periods
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Get only future predictions (last 'periods' rows)
        future_forecast = forecast.tail(periods).copy()
        
        # Format the forecast for output
        output_df = pd.DataFrame({
            'date': future_forecast['ds'].dt.strftime('%Y-%m-%d'),
            'predicted_price': future_forecast['yhat1'].round(2),
            'lower_bound': future_forecast.get('yhat1 - 95%', future_forecast['yhat1'] * 0.95).round(2),
            'upper_bound': future_forecast.get('yhat1 + 95%', future_forecast['yhat1'] * 1.05).round(2),
        })
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d')
        output_path = os.path.join(output_dir, f"{symbol}_forecast_{periods}d_{timestamp}.csv")
        
        # Save to CSV
        output_df.to_csv(output_path, index=False)
        
        eprint(f"[OK] Saved forecast to {output_path}")
        eprint(f"     First predicted date: {output_df['date'].iloc[0]}")
        eprint(f"     Last predicted date: {output_df['date'].iloc[-1]}")
        eprint(f"     Price range: ${output_df['predicted_price'].min():.2f} - ${output_df['predicted_price'].max():.2f}")
        
        return True
    
    except Exception as ex:
        eprint(f"[ERROR] Failed to save forecast for {symbol}: {ex}")
        return False


def extract_symbol_from_filename(filename: str) -> str:
    """Extract stock symbol from CSV filename.
    
    Args:
        filename: CSV filename (e.g., 'AAPL_daily_5y.csv')
        
    Returns:
        Stock symbol (e.g., 'AAPL')
    """
    basename = os.path.basename(filename)
    symbol = basename.split('_')[0]
    return symbol


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate stock price forecasts using trained NeuralProphet models."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing stock CSV files"
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to save forecast CSV files"
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=30,
        help="Number of days to forecast (default: 30)"
    )
    parser.add_argument(
        "--pattern",
        default="*_daily_*.csv",
        help="Glob pattern for CSV files to process"
    )
    
    args = parser.parse_args()
    
    # Find all CSV files in data directory
    csv_pattern = os.path.join(args.data_dir, args.pattern)
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        eprint(f"[ERROR] No CSV files found matching pattern: {csv_pattern}")
        return 1
    
    eprint(f"[INFO] Found {len(csv_files)} CSV files to process")
    eprint(f"[INFO] Forecast period: {args.periods} days")
    eprint("")
    
    # Process each stock
    success_count = 0
    fail_count = 0
    
    for csv_file in csv_files:
        symbol = extract_symbol_from_filename(csv_file)
        
        eprint(f"\n{'='*50}")
        eprint(f"Processing {symbol}")
        eprint(f"{'='*50}")
        
        # Load historical data
        df = load_stock_data(csv_file)
        if df is None:
            fail_count += 1
            continue
        
        # Load trained model
        model_path = os.path.join(args.model_dir, f"{symbol}_neuralprophet")
        if not os.path.exists(model_path):
            eprint(f"[ERROR] Model not found: {model_path}")
            fail_count += 1
            continue
        
        model = load_model(model_path)
        if model is None:
            fail_count += 1
            continue
        
        # Generate forecast
        forecast = generate_forecast(model, df, args.periods, symbol)
        if forecast is None:
            fail_count += 1
            continue
        
        # Save forecast
        if save_forecast(forecast, symbol, args.output_dir, args.periods):
            success_count += 1
        else:
            fail_count += 1
    
    # Print summary
    eprint("\n" + "="*50)
    eprint(f"[SUMMARY] Forecast generation complete")
    eprint(f"  Success: {success_count}/{len(csv_files)}")
    eprint(f"  Failed:  {fail_count}/{len(csv_files)}")
    eprint(f"  Output directory: {args.output_dir}")
    eprint("="*50)
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
