#!/usr/bin/env python3  # Shebang line: tells Unix/Linux systems to use Python 3 interpreter from PATH.
# -*- coding: utf-8 -*-  # PEP 263: Declares source file encoding as UTF-8 for international characters.

"""
Train NeuralProphet models on stock price data from CSV files.
Load data from the 'data' directory and save trained models to the 'models' directory.

NeuralProphet Overview:
- NeuralProphet is a neural network-based time series forecasting library.
- It combines Facebook Prophet's decomposable model with PyTorch's deep learning.
- Key components: trend, seasonality, auto-regression (AR), lagged regressors, future regressors.
- Uses PyTorch Lightning for training (hence the lightning_logs directory).

Workflow:
1. Load historical stock price data from CSV files (format: ds=date, y=closing_price).
2. Initialize NeuralProphet model with configurable hyperparameters.
3. Train the model on the time series data.
4. Save the trained model for later use in forecasting.

Usage:
    python train_models.py --data-dir data --model-dir models --epochs 100

Output:
    - Trained models saved to models/<SYMBOL>_neuralprophet/
    - Training logs saved to lightning_logs/<SYMBOL>/
"""  # Module docstring explaining the script's purpose and workflow.

# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================

import argparse  # Built-in module for parsing command-line arguments (--epochs, --data-dir, etc.).
import glob  # Built-in module for Unix-style pathname pattern matching (find files like *.csv).
import os  # Built-in module for operating system interactions (file paths, directory operations).
import sys  # Built-in module for system-specific parameters (stderr, exit codes).
from pathlib import Path  # Object-oriented filesystem path handling (modern alternative to os.path).
from typing import Optional  # Type hint indicating a value can be of specified type or None.

# =============================================================================
# THIRD-PARTY LIBRARY IMPORTS
# =============================================================================

import pandas as pd  # Data manipulation library; pd is the conventional alias. Used for CSV I/O and DataFrames.
from neuralprophet import NeuralProphet  # Main forecasting class from NeuralProphet library.


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def eprint(msg: str) -> None:  # Helper function to print messages to standard error stream.
    """
    Print message to stderr (standard error).
    
    Why stderr instead of stdout?
    - stdout is for program output (data, results).
    - stderr is for diagnostic messages (logs, errors, progress).
    - This separation allows users to redirect output without mixing logs.
    - Example: python train_models.py > results.txt 2> logs.txt
    
    Args:
        msg: The message string to print to stderr.
        
    Returns:
        None (this is a void function that produces side effects only).
    """  # Docstring following Google-style format for documentation.
    print(msg, file=sys.stderr)  # print() with file= parameter redirects output to specified stream.


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_stock_data(csv_path: str) -> Optional[pd.DataFrame]:  # Load CSV and return DataFrame or None.
    """
    Load stock data from CSV file and prepare it for NeuralProphet.
    
    NeuralProphet requires specific column names:
    - 'ds': The date/timestamp column (datestamp). Must be datetime type.
    - 'y': The target variable to forecast (e.g., closing price).
    
    This function validates the CSV format, converts dates, and sorts chronologically.
    
    Args:
        csv_path: Absolute or relative path to the CSV file to load.
        
    Returns:
        pd.DataFrame: DataFrame with validated columns, or None if loading/validation fails.
        
    Example CSV format:
        ds,y,open,high,low,close,volume
        2020-01-02,300.35,296.24,300.60,295.19,300.35,33911282
    """  # Comprehensive docstring explaining NeuralProphet's data requirements.
    try:  # Wrap file operations in try-except for graceful error handling.
        df = pd.read_csv(csv_path)  # pandas.read_csv() parses CSV into a DataFrame object.
        
        # Validate required columns - NeuralProphet MUST have 'ds' and 'y' columns
        if 'ds' not in df.columns or 'y' not in df.columns:  # Check if required columns exist in DataFrame.
            eprint(f"[ERROR] CSV missing required columns (ds, y): {csv_path}")  # Log error with file path.
            return None  # Return None to signal failure (Optional return type).
        
        # Ensure ds is datetime - NeuralProphet requires proper datetime objects, not strings
        df['ds'] = pd.to_datetime(df['ds'])  # Convert string dates to pandas Timestamp objects.
        
        # Sort by date - time series must be in chronological order for training
        df = df.sort_values('ds').reset_index(drop=True)  # Sort ascending; reset_index removes old index.
        
        eprint(f"[INFO] Loaded {len(df)} rows from {os.path.basename(csv_path)}")  # Log success with row count.
        return df  # Return the cleaned and validated DataFrame.
    
    except Exception as ex:  # Catch any exception (file not found, parse errors, etc.).
        eprint(f"[ERROR] Failed to load {csv_path}: {ex}")  # Log the exception message.
        return None  # Return None to indicate failure.


# =============================================================================
# MODEL TRAINING FUNCTIONS
# =============================================================================

def train_model(df: pd.DataFrame, symbol: str, config: dict) -> Optional[NeuralProphet]:  # Train and return model.
    """
    Train a NeuralProphet model on stock price time series data.
    
    NeuralProphet Model Architecture:
    - Decomposes time series into: trend + seasonality + events + auto-regression.
    - Uses a feed-forward neural network trained with stochastic gradient descent.
    - Supports multiple seasonalities (yearly, weekly, daily) with Fourier terms.
    - Uses PyTorch Lightning for training (automatic GPU support, logging, etc.).
    
    Key Hyperparameters Explained:
    - growth: 'linear' (constant rate) or 'discontinuous' (allows sudden changes).
    - n_changepoints: Number of points where trend can change direction.
    - changepoints_range: Fraction of data where changepoints can occur (0.8 = first 80%).
    - yearly_seasonality: Capture annual patterns (earnings seasons, holidays).
    - n_lags: Auto-regression order (use past N values to predict future).
    - epochs: Number of complete passes through the training data.
    - loss_func: 'Huber' is robust to outliers; alternatives: 'MSE', 'MAE'.
    
    Args:
        df: DataFrame with 'ds' (datetime) and 'y' (target price) columns.
        symbol: Stock ticker symbol (e.g., 'AAPL') for logging and file naming.
        config: Dictionary containing hyperparameters for model initialization.
        
    Returns:
        NeuralProphet: Trained model object, or None if training fails.
    """  # Detailed docstring explaining NeuralProphet's internals and hyperparameters.
    try:  # Wrap training in try-except to handle CUDA errors, memory issues, etc.
        eprint(f"[INFO] Training model for {symbol}...")  # Log training start for progress tracking.
        
        # Initialize NeuralProphet model with configuration from config dict
        # dict.get(key, default) returns default if key is missing - avoids KeyError
        model = NeuralProphet(  # Create NeuralProphet instance with specified hyperparameters.
            growth=config.get('growth', 'linear'),  # Trend growth type: 'linear' assumes steady growth/decline.
            n_changepoints=config.get('n_changepoints', 10),  # Number of potential trend change points to fit.
            changepoints_range=config.get('changepoints_range', 0.8),  # Place changepoints in first 80% of data.
            trend_reg=config.get('trend_reg', 0),  # L2 regularization on trend; 0 = no regularization.
            yearly_seasonality=config.get('yearly_seasonality', True),  # Model yearly patterns (Q1-Q4 cycles).
            weekly_seasonality=config.get('weekly_seasonality', False),  # Disable weekly patterns for stocks.
            daily_seasonality=config.get('daily_seasonality', False),  # Disable intraday patterns (we use daily data).
            seasonality_mode=config.get('seasonality_mode', 'additive'),  # 'additive' or 'multiplicative' seasonality.
            seasonality_reg=config.get('seasonality_reg', 0),  # Regularization for seasonality components.
            n_lags=config.get('n_lags', 0),  # Auto-regression lags; 0 = no AR component.
            learning_rate=config.get('learning_rate', None),  # None = auto learning rate finder.
            epochs=config.get('epochs', 100),  # Number of training epochs (full passes through data).
            batch_size=config.get('batch_size', None),  # None = auto batch size based on data size.
            loss_func=config.get('loss_func', 'Huber'),  # Huber loss: robust to outliers, combines MSE and MAE.
        )  # End of NeuralProphet initialization.
        
        # Keep only required columns (ds, y) for training - NeuralProphet ignores extra columns
        df_train = df[['ds', 'y']].copy()  # .copy() creates independent copy to avoid SettingWithCopyWarning.
        
        # Train the model using fit() method
        # freq='D' specifies daily frequency for the time series
        metrics = model.fit(df_train, freq='D')  # Returns training metrics DataFrame (loss per epoch).
        
        # Rename the latest version directory to stock name for better organization
        # PyTorch Lightning creates 'lightning_logs/version_N/' directories automatically
        lightning_logs_dir = 'lightning_logs'  # Default directory created by PyTorch Lightning.
        if os.path.exists(lightning_logs_dir):  # Check if lightning_logs directory exists.
            # Find all version directories created by PyTorch Lightning
            # List comprehension filters for directories starting with 'version_'
            version_dirs = [d for d in os.listdir(lightning_logs_dir)   # List all items in lightning_logs.
                          if os.path.isdir(os.path.join(lightning_logs_dir, d)) and d.startswith('version_')]  # Filter for version dirs.
            
            if version_dirs:  # Proceed only if version directories were found.
                # Sort to get the latest version by extracting the version number
                # lambda extracts number after underscore: 'version_5' -> 5
                version_dirs.sort(key=lambda x: int(x.split('_')[1]))  # Sort numerically by version number.
                latest_version = version_dirs[-1]  # Get last element (highest version number).
                old_path = os.path.join(lightning_logs_dir, latest_version)  # Build full path to version dir.
                new_path = os.path.join(lightning_logs_dir, symbol)  # Target path with stock symbol name.
                
                # If stock directory already exists, add version number to avoid overwriting
                if os.path.exists(new_path):  # Check if we'd overwrite existing logs.
                    version_num = 0  # Start version counter at 0.
                    # Find next available version number with while loop
                    while os.path.exists(os.path.join(lightning_logs_dir, f"{symbol}_v{version_num}")):  # Check if exists.
                        version_num += 1  # Increment until we find unused version number.
                    new_path = os.path.join(lightning_logs_dir, f"{symbol}_v{version_num}")  # Use versioned name.
                
                # Rename the directory from version_N to SYMBOL (or SYMBOL_vN)
                os.rename(old_path, new_path)  # Atomic rename operation on the filesystem.
                eprint(f"[INFO] Logs saved to {new_path}")  # Log the new location for TensorBoard viewing.
        
        eprint(f"[OK] Model trained for {symbol}")  # Log successful training completion.
        return model  # Return the trained model object for saving.
    
    except Exception as ex:  # Catch training failures (CUDA OOM, data issues, etc.).
        eprint(f"[ERROR] Failed to train model for {symbol}: {ex}")  # Log error with details.
        return None  # Return None to signal training failure.


# =============================================================================
# MODEL PERSISTENCE FUNCTIONS
# =============================================================================

def save_model(model: NeuralProphet, symbol: str, output_dir: str) -> bool:  # Save model, return success boolean.
    """
    Save trained NeuralProphet model to disk for later use.
    
    Model Serialization:
    - NeuralProphet uses torch.save() internally (PyTorch's serialization).
    - Saves model architecture, weights, and configuration.
    - Creates a directory with multiple files (not a single file).
    - Models can be loaded later with neuralprophet.load().
    
    File Structure Created:
        models/
        └── AAPL_neuralprophet/
            ├── model.pt          # PyTorch model weights
            └── config.json       # Model configuration
    
    Args:
        model: Trained NeuralProphet model object to serialize.
        symbol: Stock ticker symbol for naming the model directory.
        output_dir: Parent directory where model subdirectory will be created.
        
    Returns:
        bool: True if save was successful, False if any error occurred.
    """  # Docstring explaining serialization mechanism and file structure.
    try:  # Wrap in try-except for handling disk errors, permissions, etc.
        os.makedirs(output_dir, exist_ok=True)  # Create output directory; exist_ok prevents error if exists.
        model_path = os.path.join(output_dir, f"{symbol}_neuralprophet")  # Build path like 'models/AAPL_neuralprophet'.
        
        # NeuralProphet uses torch.save internally, save to directory
        # Import save/load functions from neuralprophet for model persistence
        from neuralprophet import save, load  # Local import; could be at top but here for clarity.
        save(model, model_path)  # Serialize model to disk; creates directory with model files.
        
        eprint(f"[OK] Saved model to {model_path}")  # Log successful save with path.
        return True  # Return True to indicate success.
    
    except Exception as ex:  # Catch disk errors, permission denied, serialization errors.
        eprint(f"[ERROR] Failed to save model for {symbol}: {ex}")  # Log error with details.
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
                'NVDA_daily_5y.csv' -> 'NVDA'
    
    The function handles both full paths and just filenames.
    
    Args:
        filename: Full path or just filename of the CSV file.
                  Example: 'data/AAPL_daily_5y.csv' or 'AAPL_daily_5y.csv'
        
    Returns:
        str: The stock ticker symbol (first segment before underscore).
    """  # Docstring explaining the filename convention and parsing logic.
    # Remove extension and extract symbol using string splitting
    basename = os.path.basename(filename)  # Extract filename from path: 'data/AAPL_daily_5y.csv' -> 'AAPL_daily_5y.csv'.
    symbol = basename.split('_')[0]  # Split by underscore, take first part: 'AAPL_daily_5y.csv' -> 'AAPL'.
    return symbol  # Return the extracted stock symbol.


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> int:  # Main function returning exit code (0 = success, non-zero = error).
    """
    Main entry point for the NeuralProphet training script.
    
    This function orchestrates the entire training pipeline:
    1. Parse command-line arguments for configuration.
    2. Discover CSV files matching the specified pattern.
    3. Load, validate, and train a model for each stock.
    4. Save trained models to the output directory.
    5. Report summary of successful/failed training runs.
    
    Exit Codes:
        0: All models trained successfully.
        1: At least one model failed to train.
        
    Returns:
        int: Exit code for the process (used by SystemExit).
    """  # Docstring explaining the main function's orchestration role.
    
    # Create argument parser for command-line interface
    # argparse provides automatic --help generation and type validation
    parser = argparse.ArgumentParser(  # Initialize argument parser object.
        description="Train NeuralProphet models on stock price data."  # Description shown in --help output.
    )  # End of ArgumentParser initialization.
    
    # Define --data-dir argument for input directory
    parser.add_argument(  # Add a command-line argument definition.
        "--data-dir",  # Argument name (accessed as args.data_dir after parsing).
        default="data",  # Default value if not specified on command line.
        help="Directory containing stock CSV files"  # Help text shown in --help.
    )  # End of --data-dir argument.
    
    # Define --model-dir argument for output directory
    parser.add_argument(  # Add another argument.
        "--model-dir",  # Models will be saved to this directory.
        default="models",  # Default to 'models' subdirectory.
        help="Directory to save trained models"  # Describe purpose in help.
    )  # End of --model-dir argument.
    
    # Define --epochs argument for training duration
    parser.add_argument(  # Add epochs configuration.
        "--epochs",  # Number of complete passes through training data.
        type=int,  # Enforce integer type (argparse validates this).
        default=100,  # 100 epochs is a reasonable default for NeuralProphet.
        help="Number of training epochs"  # Explain what epochs means.
    )  # End of --epochs argument.
    
    # Define --learning-rate argument for optimizer
    parser.add_argument(  # Add learning rate configuration.
        "--learning-rate",  # Controls step size in gradient descent.
        type=float,  # Learning rate is a floating-point number.
        default=None,  # None = use NeuralProphet's auto learning rate finder.
        help="Learning rate (default: auto)"  # Explain auto behavior.
    )  # End of --learning-rate argument.
    
    # Define --n-changepoints argument for trend flexibility
    parser.add_argument(  # Add changepoints configuration.
        "--n-changepoints",  # Number of points where trend can change.
        type=int,  # Must be an integer.
        default=10,  # 10 changepoints works well for 5 years of daily data.
        help="Number of potential changepoints"  # Explain changepoints concept.
    )  # End of --n-changepoints argument.
    
    # Define --yearly-seasonality flag
    parser.add_argument(  # Add seasonality configuration.
        "--yearly-seasonality",  # Model annual patterns in stock prices.
        action="store_true",  # Boolean flag: presence = True, absence = default.
        default=True,  # Enable yearly seasonality by default.
        help="Enable yearly seasonality"  # Explain seasonality.
    )  # End of --yearly-seasonality argument.
    
    # Define --verbose flag for training progress
    parser.add_argument(  # Add verbosity configuration.
        "--verbose",  # Show detailed training progress.
        action="store_true",  # Boolean flag.
        help="Show training progress"  # Explain what verbose does.
    )  # End of --verbose argument.
    
    # Define --pattern argument for file matching
    parser.add_argument(  # Add file pattern configuration.
        "--pattern",  # Glob pattern to find CSV files.
        default="*_daily_*.csv",  # Matches files like 'AAPL_daily_5y.csv'.
        help="Glob pattern for CSV files to process"  # Explain glob patterns.
    )  # End of --pattern argument.
    
    # Parse command-line arguments and store in args namespace object
    args = parser.parse_args()  # Returns Namespace object with all argument values.
    
    # Find all CSV files in data directory using glob pattern matching
    # os.path.join safely combines directory and pattern across OS
    csv_pattern = os.path.join(args.data_dir, args.pattern)  # Build pattern like 'data/*_daily_*.csv'.
    csv_files = glob.glob(csv_pattern)  # Find all files matching the pattern; returns list of paths.
    
    # Validate that we found at least one CSV file to process
    if not csv_files:  # Empty list is falsy in Python.
        eprint(f"[ERROR] No CSV files found matching pattern: {csv_pattern}")  # Report error with pattern.
        return 1  # Return exit code 1 to indicate failure.
    
    eprint(f"[INFO] Found {len(csv_files)} CSV files to process")  # Log number of files found.
    
    # Training configuration dictionary
    # This dict is passed to train_model() and used to initialize NeuralProphet
    # Using a dict allows easy extension without changing function signatures
    config = {  # Configuration dictionary for model hyperparameters.
        'epochs': args.epochs,  # Number of training epochs from CLI arg.
        'learning_rate': args.learning_rate,  # Learning rate from CLI (None = auto).
        'n_changepoints': args.n_changepoints,  # Trend changepoints from CLI.
        'yearly_seasonality': args.yearly_seasonality,  # Yearly seasonality flag.
        'verbose': args.verbose,  # Verbosity flag for training output.
        'growth': 'linear',  # Use linear trend growth (constant rate of change).
        'seasonality_mode': 'additive',  # Additive: seasonality is added to trend.
        'loss_func': 'Huber',  # Huber loss: robust to outliers in stock prices.
    }  # End of config dictionary.
    
    # ==========================================================================
    # MAIN TRAINING LOOP
    # ==========================================================================
    
    # Train models for each CSV file found in the data directory
    # Track success and failure counts for final summary
    success_count = 0  # Counter for successfully trained models.
    fail_count = 0  # Counter for failed training attempts.
    
    # Iterate through each discovered CSV file
    for csv_file in csv_files:  # Loop through list of file paths.
        # Extract stock symbol from filename for logging and model naming
        symbol = extract_symbol_from_filename(csv_file)  # e.g., 'data/AAPL_daily_5y.csv' -> 'AAPL'.
        
        # Load data from CSV file into pandas DataFrame
        df = load_stock_data(csv_file)  # Returns DataFrame or None on failure.
        if df is None:  # Check if loading failed.
            fail_count += 1  # Increment failure counter.
            continue  # Skip to next file; don't attempt training.
        
        # Check minimum data requirements - NeuralProphet needs sufficient history
        # 30 data points is minimum for reliable trend and seasonality estimation
        if len(df) < 30:  # Validate minimum data requirement.
            eprint(f"[WARN] Insufficient data for {symbol} ({len(df)} rows). Skipping.")  # Log warning.
            fail_count += 1  # Count as failure.
            continue  # Skip to next file.
        
        # Train model using loaded data and configuration
        model = train_model(df, symbol, config)  # Returns trained model or None on failure.
        if model is None:  # Check if training failed.
            fail_count += 1  # Increment failure counter.
            continue  # Skip saving; move to next file.
        
        # Save trained model to disk
        if save_model(model, symbol, args.model_dir):  # Returns True on success, False on failure.
            success_count += 1  # Increment success counter on successful save.
        else:  # Save failed.
            fail_count += 1  # Increment failure counter.
    
    # ==========================================================================
    # PRINT TRAINING SUMMARY
    # ==========================================================================
    
    # Print summary of training results
    # Use visual separators to make summary stand out in logs
    eprint("\n" + "="*50)  # Print newline and separator line (50 equals signs).
    eprint(f"[SUMMARY] Training complete")  # Summary header.
    eprint(f"  Success: {success_count}/{len(csv_files)}")  # Report successful trainings.
    eprint(f"  Failed:  {fail_count}/{len(csv_files)}")  # Report failed trainings.
    eprint("="*50)  # Closing separator line.
    
    # Return appropriate exit code based on results
    # Unix convention: 0 = success, non-zero = error
    return 0 if fail_count == 0 else 1  # Ternary expression: return 0 if all succeeded, else 1.


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

# Standard Python idiom: only execute main() when script is run directly
# This check is False when the module is imported by another script
if __name__ == "__main__":  # True only when script is executed directly (not imported).
    raise SystemExit(main())  # Call main() and exit with its return code.
