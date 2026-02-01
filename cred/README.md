# Credentials Directory

This directory stores API keys and other sensitive credentials.

## Files

- `credentials.json` - Financial Modeling Prep API key
- `newsapi_credentials.json` - NewsAPI key for news fetching

## Setup

### Financial Modeling Prep API (Stock Data)

Create `credentials.json`:
```json
{
  "FMP_API_KEY": "your_financial_modeling_prep_api_key"
}
```

Get your API key at: https://site.financialmodelingprep.com/developer/docs

### NewsAPI (News Articles)

**Option A: Environment Variable (Recommended)**
```bash
# Windows PowerShell
$env:NEWSAPI_KEY="your_newsapi_key"

# Linux/Mac
export NEWSAPI_KEY="your_newsapi_key"
```

**Option B: Credentials File**
Create `newsapi_credentials.json`:
```json
{
  "api_key": "your_newsapi_key"
}
```

Get your free API key at: https://newsapi.org/register

### Ollama LLM (AI Analysis Agent)

The stock analysis agent uses Ollama with Llama 3.2 locally. No API key needed.

1. Install Ollama: https://ollama.com/download
2. Pull the model:
```bash
ollama pull llama3.2
```

## Security

⚠️ **Important**: Never commit credential files to git. These files are excluded via `.gitignore`.
