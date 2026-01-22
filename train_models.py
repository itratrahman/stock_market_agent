#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train NeuralProphet models on stock price data from CSV files.
Load data from the 'data' directory and save trained models to the 'models' directory.
"""

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from neuralprophet import NeuralProphet


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


def train_model(df: pd.DataFrame, symbol: str, config: dict) -> Optional[NeuralProphet]:
    """Train a NeuralProphet model on stock data.
    
    Args:
        df: DataFrame with columns ds (datetime) and y (price)
        symbol: Stock symbol for logging
        config: Training configuration dictionary
        
    Returns:
        Trained NeuralProphet model or None if training fails
    """
    try:
        eprint(f"[INFO] Training model for {symbol}...")
        
        # Initialize NeuralProphet model
        model = NeuralProphet(
            growth=config.get('growth', 'linear'),
            n_changepoints=config.get('n_changepoints', 10),
            changepoints_range=config.get('changepoints_range', 0.8),
            trend_reg=config.get('trend_reg', 0),
            yearly_seasonality=config.get('yearly_seasonality', True),
            weekly_seasonality=config.get('weekly_seasonality', False),
            daily_seasonality=config.get('daily_seasonality', False),
            seasonality_mode=config.get('seasonality_mode', 'additive'),
            seasonality_reg=config.get('seasonality_reg', 0),
            n_lags=config.get('n_lags', 0),
            learning_rate=config.get('learning_rate', None),
            epochs=config.get('epochs', 100),
            batch_size=config.get('batch_size', None),
            loss_func=config.get('loss_func', 'Huber'),
        )
        
        # Keep only required columns (ds, y) for training
        df_train = df[['ds', 'y']].copy()
        
        # Train the model
        metrics = model.fit(df_train, freq='D')
        
        # Rename the latest version directory to stock name
        lightning_logs_dir = 'lightning_logs'
        if os.path.exists(lightning_logs_dir):
            # Find all version directories
            version_dirs = [d for d in os.listdir(lightning_logs_dir) 
                          if os.path.isdir(os.path.join(lightning_logs_dir, d)) and d.startswith('version_')]
            
            if version_dirs:
                # Sort to get the latest version
                version_dirs.sort(key=lambda x: int(x.split('_')[1]))
                latest_version = version_dirs[-1]
                old_path = os.path.join(lightning_logs_dir, latest_version)
                new_path = os.path.join(lightning_logs_dir, symbol)
                
                # If stock directory exists, add version number to it
                if os.path.exists(new_path):
                    version_num = 0
                    while os.path.exists(os.path.join(lightning_logs_dir, f"{symbol}_v{version_num}")):
                        version_num += 1
                    new_path = os.path.join(lightning_logs_dir, f"{symbol}_v{version_num}")
                
                # Rename the directory
                os.rename(old_path, new_path)
                eprint(f"[INFO] Logs saved to {new_path}")
        
        eprint(f"[OK] Model trained for {symbol}")
        return model
    
    except Exception as ex:
        eprint(f"[ERROR] Failed to train model for {symbol}: {ex}")
        return None


def save_model(model: NeuralProphet, symbol: str, output_dir: str) -> bool:
    """Save trained model to disk.
    
    Args:
        model: Trained NeuralProphet model
        symbol: Stock symbol for filename
        output_dir: Directory to save model
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"{symbol}_neuralprophet")
        
        # NeuralProphet uses torch.save internally, save to directory
        from neuralprophet import save, load
        save(model, model_path)
        
        eprint(f"[OK] Saved model to {model_path}")
        return True
    
    except Exception as ex:
        eprint(f"[ERROR] Failed to save model for {symbol}: {ex}")
        return False


def extract_symbol_from_filename(filename: str) -> str:
    """Extract stock symbol from CSV filename.
    
    Args:
        filename: CSV filename (e.g., 'AAPL_daily_5y.csv')
        
    Returns:
        Stock symbol (e.g., 'AAPL')
    """
    # Remove extension and extract symbol
    basename = os.path.basename(filename)
    symbol = basename.split('_')[0]
    return symbol


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train NeuralProphet models on stock price data."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing stock CSV files"
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (default: auto)"
    )
    parser.add_argument(
        "--n-changepoints",
        type=int,
        default=10,
        help="Number of potential changepoints"
    )
    parser.add_argument(
        "--yearly-seasonality",
        action="store_true",
        default=True,
        help="Enable yearly seasonality"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show training progress"
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
    
    # Training configuration
    config = {
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'n_changepoints': args.n_changepoints,
        'yearly_seasonality': args.yearly_seasonality,
        'verbose': args.verbose,
        'growth': 'linear',
        'seasonality_mode': 'additive',
        'loss_func': 'Huber',
    }
    
    # Train models for each CSV file
    success_count = 0
    fail_count = 0
    
    for csv_file in csv_files:
        symbol = extract_symbol_from_filename(csv_file)
        
        # Load data
        df = load_stock_data(csv_file)
        if df is None:
            fail_count += 1
            continue
        
        # Check minimum data requirements
        if len(df) < 30:
            eprint(f"[WARN] Insufficient data for {symbol} ({len(df)} rows). Skipping.")
            fail_count += 1
            continue
        
        # Train model
        model = train_model(df, symbol, config)
        if model is None:
            fail_count += 1
            continue
        
        # Save model
        if save_model(model, symbol, args.model_dir):
            success_count += 1
        else:
            fail_count += 1
    
    # Print summary
    eprint("\n" + "="*50)
    eprint(f"[SUMMARY] Training complete")
    eprint(f"  Success: {success_count}/{len(csv_files)}")
    eprint(f"  Failed:  {fail_count}/{len(csv_files)}")
    eprint("="*50)
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
