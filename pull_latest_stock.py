#!/usr/bin/env python3  # Use the system's Python 3 interpreter.
# -*- coding: utf-8 -*-  # Ensure UTF-8 source encoding.

"""
Fetch last N years of daily stock history (OHLCV) for the top N companies (by market cap)
from a default mega-cap universe using Financial Modeling Prep (FMP), and save one CSV per symbol
in a NeuralProphet-friendly format (columns: ds, y).
"""  # Module docstring describing what this script does.

import argparse  # Parse command-line arguments.
import csv  # Write CSV files without extra dependencies.
import datetime as dt  # Work with dates/times for filtering last N years.
import json  # Parse JSON credential files.
import os  # Read environment variables and create directories.
import sys  # Exit with proper status codes and print to stderr.
from typing import Any, Dict, List, Optional, Tuple  # Type hints for readability and safety.
from urllib.parse import urlencode  # Safely encode URL query parameters.

import requests  # Make HTTP requests to the FMP API.

BASE_URL = "https://financialmodelingprep.com/stable"  # Base URL for FMP "stable" endpoints.  # noqa: E501

DEFAULT_UNIVERSE = [  # Fixed list of tickers (no automatic detection).
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "NVDA",  # NVIDIA
    "AMZN",  # Amazon
    "GOOGL",  # Alphabet Class A
    "META",  # Meta
    "TSLA",  # Tesla
]  # Fixed candidate list (no market-cap ranking).

def eprint(msg: str) -> None:  # Small helper to print errors to stderr.
    print(msg, file=sys.stderr)  # Print message to stderr.

def safe_mkdir(path: str) -> None:  # Create a directory if it doesn’t exist.
    os.makedirs(path, exist_ok=True)  # Make directory recursively; ignore if already exists.

def build_url(path: str, params: Dict[str, Any]) -> str:  # Build a full URL with query params.
    return f"{BASE_URL}{path}?{urlencode(params)}"  # Join base + path and encode parameters.

def http_get_json(session: requests.Session, url: str, timeout: int = 30) -> Any:  # GET JSON with basic safety.
    resp = session.get(url, timeout=timeout)  # Send HTTP GET request with a timeout.
    if resp.status_code != 200:  # Check for non-success HTTP responses.
        raise RuntimeError(f"HTTP {resp.status_code} for URL: {url} | Body: {resp.text[:300]}")  # Raise with context.
    try:  # Attempt to parse JSON response body.
        return resp.json()  # Return parsed JSON (dict/list depending on endpoint).
    except ValueError as ex:  # Catch JSON parsing errors.
        raise RuntimeError(f"Invalid JSON for URL: {url} | Body: {resp.text[:300]}") from ex  # Re-raise with context.

def get_market_cap(session: requests.Session, api_key: str, symbol: str) -> Optional[float]:  # Fetch market cap for a symbol.
    url = build_url("/profile", {"symbol": symbol, "apikey": api_key})  # Build the profile endpoint URL.
    data = http_get_json(session, url)  # Call the API and parse JSON.
    if not isinstance(data, list) or len(data) == 0:  # Profile endpoint typically returns a list.
        return None  # Return None if profile data is missing.
    item = data[0]  # Use the first (and usually only) profile object.
    mc = item.get("marketCap")  # Read the marketCap field from the profile payload.
    if mc is None:  # Handle missing market cap.
        return None  # Return None if not available.
    try:  # Convert market cap to float safely.
        return float(mc)  # Return numeric market cap.
    except (TypeError, ValueError):  # Handle unexpected types (e.g., strings that don’t parse).
        return None  # Return None if parsing fails.

def pick_top_by_market_cap(session: requests.Session, api_key: str, universe: List[str], top_n: int) -> List[Tuple[str, float]]:  # Rank symbols by market cap.
    caps: List[Tuple[str, float]] = []  # Prepare (symbol, market_cap) list for sorting.
    for sym in universe:  # Loop over candidate tickers.
        try:  # Catch per-symbol failures without killing the full run.
            mc = get_market_cap(session, api_key, sym)  # Fetch market cap from FMP profile endpoint.
            if mc is not None:  # Keep only valid market cap values.
                caps.append((sym, mc))  # Store symbol + market cap for later sorting.
            else:  # If market cap not found.
                eprint(f"[WARN] marketCap missing for {sym}; skipping ranking for this symbol.")  # Warn and continue.
        except Exception as ex:  # Handle network/JSON/HTTP errors for this symbol.
            eprint(f"[WARN] Failed to fetch marketCap for {sym}: {ex}")  # Log the failure.
    caps.sort(key=lambda x: x[1], reverse=True)  # Sort descending by market cap.
    return caps[:top_n]  # Return the top N (symbol, market_cap) pairs.

def clamp_years_ago(today: dt.date, years: int) -> dt.date:  # Compute a date "years" back with leap-day safety.
    try:  # Try the simple “same month/day” approach.
        return dt.date(today.year - years, today.month, today.day)  # Construct date years ago.
    except ValueError:  # Handle cases like Feb 29 on a non-leap year.
        return dt.date(today.year - years, today.month, min(today.day, 28))  # Fall back to day 28 for safety.

def fetch_eod_history(session: requests.Session, api_key: str, symbol: str) -> List[Dict[str, Any]]:  # Fetch daily EOD history.
    url = build_url("/historical-price-eod/full", {"symbol": symbol, "apikey": api_key})  # Build EOD endpoint URL.
    data = http_get_json(session, url)  # Call the API and parse JSON.
    if not isinstance(data, dict):  # Expect a dict with a "historical" field.
        raise RuntimeError(f"Unexpected payload type for {symbol}: {type(data)}")  # Raise if format is unexpected.
    hist = data.get("historical", [])  # Pull the historical list (may be empty).
    if not isinstance(hist, list):  # Ensure historical is a list.
        raise RuntimeError(f"Unexpected 'historical' type for {symbol}: {type(hist)}")  # Raise if wrong type.
    return hist  # Return the raw list of daily bars.

def to_date(s: str) -> dt.date:  # Convert YYYY-MM-DD string to a date object.
    return dt.datetime.strptime(s, "%Y-%m-%d").date()  # Parse and return date.

def sanitize_filename_symbol(symbol: str) -> str:  # Make a symbol safe for filenames across OSes.
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in symbol)  # Replace weird chars with "_".

def load_api_key_from_cred(cred_file: str = "cred/credentials.json") -> Optional[str]:  # Load API key from credential file.
    if not os.path.exists(cred_file):  # Check if credential file exists.
        return None  # Return None if file doesn't exist.
    try:  # Attempt to read and parse credential file.
        with open(cred_file, "r", encoding="utf-8") as f:  # Open credential file.
            creds = json.load(f)  # Parse JSON content.
            return creds.get("FMP_API_KEY") or creds.get("api_key")  # Try both key names.
    except (json.JSONDecodeError, IOError) as ex:  # Handle JSON or file errors.
        eprint(f"[WARN] Failed to load credentials from {cred_file}: {ex}")  # Log warning.
        return None  # Return None on error.

def write_neuralprophet_csv(out_path: str, rows: List[Dict[str, Any]]) -> None:  # Save rows to CSV with ds,y columns.
    fieldnames = ["ds", "y", "open", "high", "low", "close", "volume"]  # CSV header (ds,y required by NeuralProphet).
    with open(out_path, "w", newline="", encoding="utf-8") as f:  # Open file for writing in UTF-8.
        writer = csv.DictWriter(f, fieldnames=fieldnames)  # Create a DictWriter with fixed columns.
        writer.writeheader()  # Write header row once.
        for r in rows:  # Iterate over prepared rows.
            writer.writerow(r)  # Write each row as a CSV line.

def main() -> int:  # Main entry point returning process exit code.
    parser = argparse.ArgumentParser(description="Fetch top N stock histories from FMP and save NeuralProphet-ready CSVs.")  # CLI parser.
    parser.add_argument("--api-key", default="", help="FMP API key (overrides credential file).")  # API key input.
    parser.add_argument("--cred-file", default="cred/credentials.json", help="Path to credentials JSON file.")  # Credential file path.
    parser.add_argument("--years", type=int, default=5, help="How many years of daily history to keep.")  # Lookback in years.
    parser.add_argument("--top-n", type=int, default=5, help="How many top companies (by market cap) to fetch.")  # Number of companies.
    parser.add_argument("--outdir", default="data", help="Output directory for per-symbol CSV files.")  # Output directory.
    parser.add_argument("--universe", default=",".join(DEFAULT_UNIVERSE), help="Comma-separated tickers to rank by market cap.")  # Universe override.
    args = parser.parse_args()  # Parse CLI arguments.

    # Load API key from credential file if not provided via CLI
    api_key = args.api_key or load_api_key_from_cred(args.cred_file) or os.getenv("FMP_API_KEY", "")  # Try CLI, then cred file, then env var.
    
    if not api_key:  # Ensure an API key is provided.
        eprint(f"[ERROR] Missing API key. Provide via --api-key, {args.cred_file}, or FMP_API_KEY env var.")  # Explain how to provide key.
        return 2  # Exit code 2 for invalid usage.

    if args.years <= 0:  # Validate years argument.
        eprint("[ERROR] --years must be a positive integer.")  # Print validation error.
        return 2  # Exit code 2 for invalid usage.

    if args.top_n <= 0:  # Validate top-n argument.
        eprint("[ERROR] --top-n must be a positive integer.")  # Print validation error.
        return 2  # Exit code 2 for invalid usage.

    universe = [s.strip().upper() for s in args.universe.split(",") if s.strip()]  # Normalize universe tickers.
    if not universe:  # Ensure universe is not empty.
        eprint("[ERROR] Universe is empty. Provide --universe or rely on the default list.")  # Explain requirement.
        return 2  # Exit code 2 for invalid usage.

    safe_mkdir(args.outdir)  # Create output directory if needed.

    session = requests.Session()  # Reuse HTTP connections for efficiency.

    eprint(f"[INFO] Ranking {len(universe)} symbols by market cap using FMP /profile ...")  # Status message.
    top = [(sym, 0.0) for sym in universe[:args.top_n]]  # Use provided list only (no market-cap ranking).
    if not top:  # Handle case where ranking failed completely.
        eprint("[ERROR] Could not retrieve market caps for any symbols in the universe.")  # Print error.
        return 1  # Exit code 1 for runtime failure.

    eprint("[INFO] Selected top symbols (symbol, marketCap):")  # Header for selected list.
    for sym, mc in top:  # Loop over selected symbols.
        eprint(f"       - {sym}: {mc:,.0f}")  # Print market cap with thousands separators.

    today = dt.date.today()  # Today's date for lookback computation.
    start_date = clamp_years_ago(today, args.years)  # Compute start date years back.
    eprint(f"[INFO] Keeping rows from {start_date.isoformat()} to {today.isoformat()} (inclusive).")  # Print chosen date range.

    for sym, _mc in top:  # Loop over each selected symbol.
        try:  # Catch per-symbol errors and continue.
            eprint(f"[INFO] Fetching EOD history for {sym} ...")  # Status message.
            hist = fetch_eod_history(session, api_key, sym)  # Pull raw historical bars.
            kept: List[Dict[str, Any]] = []  # List for filtered + transformed rows.
            for bar in hist:  # Iterate over raw bars (typically includes date, open, high, low, close, volume, etc.).
                d_str = bar.get("date")  # Extract date string.
                if not d_str:  # Skip if missing date.
                    continue  # Move to next bar.
                d = to_date(d_str)  # Parse date string to date object.
                if d < start_date or d > today:  # Filter outside the desired window.
                    continue  # Skip bars outside the lookback window.
                close = bar.get("close")  # Get close price (used as y for NeuralProphet).
                if close is None:  # Skip if close is missing.
                    continue  # Move to next bar.
                kept.append({  # Append one normalized row with NeuralProphet-friendly columns.
                    "ds": d_str,  # ds = timestamp column (YYYY-MM-DD).
                    "y": float(close),  # y = target series value (float).
                    "open": float(bar.get("open", 0.0) or 0.0),  # Optional feature: open price.
                    "high": float(bar.get("high", 0.0) or 0.0),  # Optional feature: high price.
                    "low": float(bar.get("low", 0.0) or 0.0),  # Optional feature: low price.
                    "close": float(close),  # Keep close explicitly as well (same as y).
                    "volume": float(bar.get("volume", 0.0) or 0.0),  # Optional feature: volume.
                })  # End of row dict.

            kept.sort(key=lambda r: r["ds"])  # Ensure ascending chronological order for training.

            safe_sym = sanitize_filename_symbol(sym)  # Sanitize symbol for filename safety.
            out_path = os.path.join(args.outdir, f"{safe_sym}_daily_{args.years}y.csv")  # Build per-symbol output path.
            write_neuralprophet_csv(out_path, kept)  # Write CSV to disk.
            eprint(f"[OK] Wrote {len(kept)} rows -> {out_path}")  # Confirm saved file.
        except Exception as ex:  # Catch and report per-symbol failures.
            eprint(f"[WARN] Failed for {sym}: {ex}")  # Log warning and continue.

    eprint("[DONE] Finished fetching and writing CSVs.")  # Final status message.
    return 0  # Exit success.

if __name__ == "__main__":  # Standard Python entry point guard.
    raise SystemExit(main())  # Run main() and exit with its return code.
