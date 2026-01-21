# Credentials Directory

This directory stores API keys and other sensitive credentials.

## Files

- `credentials.json` - Contains API keys and secrets (not tracked in git)
- `credentials.json.example` - Template file showing the expected format

## Setup

1. Copy the example file:
   ```bash
   cp credentials.json.example credentials.json
   ```

2. Edit `credentials.json` and add your actual API keys:
   ```json
   {
     "FMP_API_KEY": "your_financial_modeling_prep_api_key"
   }
   ```

## Security

⚠️ **Important**: Never commit `credentials.json` to git. This file is excluded via `.gitignore`.
