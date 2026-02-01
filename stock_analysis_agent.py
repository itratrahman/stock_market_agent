#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stock Analysis Agent using LangGraph and LangChain.

This script implements an agentic flow for analyzing stock forecasts and news
for each interested company. It uses open-source LLMs (Ollama with Llama 3.2)
to generate summaries and make investment decisions.

Flow:
1. Read forecast CSV and create verbal summary (< 300 chars)
2. Decision node: If promising, summarize news articles; else skip
3. Decision node: Make investment recommendation based on summaries
4. End flow and save results to tabulated text file

Requirements:
- Ollama running locally with llama3.2 model
- Install: pip install langchain langchain-ollama langgraph pandas
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import os  # For file and directory operations
import sys  # For system-level operations and stderr
import json  # For parsing JSON news files
import glob  # For finding files matching patterns
from datetime import datetime  # For generating timestamps
from typing import TypedDict, Literal, Optional, List, Dict, Any  # Type hints

# Third-party imports
import pandas as pd  # For reading CSV files and data manipulation

# LangChain imports for LLM interaction
from langchain_ollama import ChatOllama  # Ollama LLM integration
from langchain_core.messages import HumanMessage, SystemMessage  # Message types
from langchain_core.prompts import ChatPromptTemplate  # Prompt templates

# LangGraph imports for building the agentic flow
from langgraph.graph import StateGraph, END  # Graph construction


# =============================================================================
# CONFIGURATION
# =============================================================================

# Directory paths for data files
OUTPUTS_DIR = "outputs"  # Directory containing forecast CSVs and news JSONs
DATA_DIR = "data"  # Directory containing historical stock data

# List of stock symbols to analyze
STOCK_SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

# LLM model configuration - using Ollama with Llama 3.2 (open-source)
# Llama 3.2 is chosen for its good balance of performance and speed
# Alternative models: mistral, phi3, gemma2
LLM_MODEL = "llama3.2"  # Primary model for analysis tasks
LLM_MODEL_CODE = "llama3.2"  # Model for code interpretation tasks

# Character limits for summaries as per requirements
FORECAST_SUMMARY_LIMIT = 300  # Verbal forecast summary character limit
NEWS_SUMMARY_LIMIT = 1000  # Global news summary character limit
DECISION_SUMMARY_LIMIT = 200  # Final decision summary character limit


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def eprint(msg: str) -> None:
    """
    Print message to stderr for logging purposes.
    
    Args:
        msg: The message string to print to stderr
    """
    # Use stderr to separate logs from actual output
    print(msg, file=sys.stderr)


def load_forecast_csv(symbol: str) -> Optional[pd.DataFrame]:
    """
    Load the most recent forecast CSV file for a given stock symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
        
    Returns:
        DataFrame containing forecast data or None if not found
    """
    # Create glob pattern to match forecast files for this symbol
    pattern = os.path.join(OUTPUTS_DIR, f"{symbol}_forecast_*.csv")
    
    # Find all matching forecast files
    files = glob.glob(pattern)
    
    # Check if any files were found
    if not files:
        # Log error if no forecast files exist
        eprint(f"[ERROR] No forecast file found for {symbol}")
        return None
    
    # Sort files to get the most recent one (by filename date)
    files.sort(reverse=True)
    
    # Select the most recent file
    latest_file = files[0]
    
    # Log which file is being loaded
    eprint(f"[INFO] Loading forecast from {latest_file}")
    
    # Read and return the CSV file as a DataFrame
    return pd.read_csv(latest_file)


def load_news_json(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Load the most recent news JSON file for a given stock symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
        
    Returns:
        Dictionary containing news data or None if not found
    """
    # Construct path to symbol's news directory
    news_dir = os.path.join(OUTPUTS_DIR, symbol)
    
    # Create glob pattern to match news files
    pattern = os.path.join(news_dir, f"{symbol}_news_newsapi_*.json")
    
    # Find all matching news files
    files = glob.glob(pattern)
    
    # Check if any files were found
    if not files:
        # Log error if no news files exist
        eprint(f"[ERROR] No news file found for {symbol}")
        return None
    
    # Sort files to get the most recent one
    files.sort(reverse=True)
    
    # Select the most recent file
    latest_file = files[0]
    
    # Log which file is being loaded
    eprint(f"[INFO] Loading news from {latest_file}")
    
    # Open and parse the JSON file
    with open(latest_file, 'r', encoding='utf-8') as f:
        # Return parsed JSON as dictionary
        return json.load(f)


# =============================================================================
# STATE DEFINITION FOR LANGGRAPH
# =============================================================================

class StockAnalysisState(TypedDict):
    """
    State definition for the stock analysis graph.
    
    This TypedDict defines all the data that flows through the graph nodes.
    Each field represents a piece of information that nodes can read/write.
    """
    # Stock symbol being analyzed (e.g., 'AAPL')
    symbol: str
    
    # Raw forecast DataFrame converted to dict for serialization
    forecast_data: Optional[Dict]
    
    # Verbal summary of the forecast (< 300 chars)
    forecast_summary: str
    
    # Flag indicating if the forecast is promising
    is_promising: bool
    
    # Raw news articles list
    news_articles: Optional[List[Dict]]
    
    # Global summary of all news articles (< 1000 chars)
    news_summary: str
    
    # Investment decision: 'invest', 'avoid', or 'neutral'
    decision: str
    
    # Final decision summary (< 200 chars)
    decision_summary: str
    
    # Any error messages encountered during processing
    error: Optional[str]


# =============================================================================
# LLM INITIALIZATION
# =============================================================================

def get_llm(model: str = LLM_MODEL) -> ChatOllama:
    """
    Initialize and return an Ollama LLM instance.
    
    Args:
        model: Name of the Ollama model to use
        
    Returns:
        Configured ChatOllama instance
    """
    # Create ChatOllama instance with specified model
    # Temperature of 0.3 for more deterministic outputs
    return ChatOllama(
        model=model,  # Model name (e.g., 'llama3.2')
        temperature=0.3,  # Lower temperature for more focused responses
    )


# =============================================================================
# NODE FUNCTIONS FOR LANGGRAPH
# =============================================================================

def analyze_forecast_node(state: StockAnalysisState) -> StockAnalysisState:
    """
    Node 1: Analyze forecast data and create verbal summary.
    
    This node reads the forecast CSV, interprets the data using an LLM,
    and generates a verbal summary under 300 characters.
    
    Args:
        state: Current graph state containing symbol information
        
    Returns:
        Updated state with forecast_summary and is_promising flag
    """
    # Extract stock symbol from state
    symbol = state["symbol"]
    
    # Log node execution
    eprint(f"\n[NODE] Analyzing forecast for {symbol}...")
    
    # Load forecast data from CSV file
    forecast_df = load_forecast_csv(symbol)
    
    # Handle case where forecast file is not found
    if forecast_df is None:
        # Return state with error information
        return {
            **state,
            "forecast_summary": f"No forecast data available for {symbol}.",
            "is_promising": False,
            "error": f"Forecast file not found for {symbol}"
        }
    
    # Store forecast data in state (convert to dict for serialization)
    state["forecast_data"] = forecast_df.to_dict()
    
    # Calculate key metrics from forecast data for LLM analysis
    # Get the first predicted price (starting point)
    start_price = forecast_df['predicted_price'].iloc[0]
    
    # Get the last predicted price (end point)
    end_price = forecast_df['predicted_price'].iloc[-1]
    
    # Calculate percentage change over forecast period
    pct_change = ((end_price - start_price) / start_price) * 100
    
    # Get min and max prices in the forecast
    min_price = forecast_df['predicted_price'].min()
    max_price = forecast_df['predicted_price'].max()
    
    # Calculate average price
    avg_price = forecast_df['predicted_price'].mean()
    
    # Get forecast date range
    start_date = forecast_df['date'].iloc[0]
    end_date = forecast_df['date'].iloc[-1]
    
    # Get confidence interval width (if bounds exist)
    if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
        # Calculate average confidence interval width
        avg_interval = (forecast_df['upper_bound'] - forecast_df['lower_bound']).mean()
    else:
        # Default value if bounds not available
        avg_interval = 0
    
    # Initialize LLM for code interpretation task
    llm = get_llm(LLM_MODEL_CODE)
    
    # Create prompt for forecast analysis
    # This prompt simulates code interpreter by providing computed metrics
    analysis_prompt = f"""You are a stock market analyst. Analyze this forecast data and provide a verbal summary.

STOCK: {symbol}
FORECAST PERIOD: {start_date} to {end_date}
METRICS:
- Starting Price: ${start_price:.2f}
- Ending Price: ${end_price:.2f}
- Price Change: {pct_change:.2f}%
- Price Range: ${min_price:.2f} - ${max_price:.2f}
- Average Price: ${avg_price:.2f}
- Avg Confidence Interval Width: ${avg_interval:.2f}

Based on these metrics, provide:
1. A verbal summary of the forecast (MUST be under 300 characters)
2. Is this forecast promising? Answer YES or NO.

Format your response EXACTLY as:
SUMMARY: <your summary here>
PROMISING: <YES or NO>
"""
    
    # Send prompt to LLM and get response
    response = llm.invoke([HumanMessage(content=analysis_prompt)])
    
    # Extract response content
    response_text = response.content
    
    # Log raw LLM response for debugging
    eprint(f"[DEBUG] LLM Response: {response_text[:200]}...")
    
    # Parse the response to extract summary and promising flag
    # Initialize default values
    forecast_summary = f"{symbol}: Price predicted to change {pct_change:.1f}% from ${start_price:.2f} to ${end_price:.2f}."
    is_promising = pct_change > 0  # Default: promising if positive change
    
    # Try to parse structured response from LLM
    if "SUMMARY:" in response_text:
        # Extract summary portion
        summary_start = response_text.find("SUMMARY:") + len("SUMMARY:")
        summary_end = response_text.find("PROMISING:")
        if summary_end > summary_start:
            # Get the summary text and strip whitespace
            forecast_summary = response_text[summary_start:summary_end].strip()
        else:
            # If PROMISING not found, take rest of text
            forecast_summary = response_text[summary_start:].strip()
    
    # Parse promising flag from response
    if "PROMISING:" in response_text:
        # Extract the YES/NO answer
        promising_text = response_text.split("PROMISING:")[-1].strip().upper()
        # Check if response indicates promising
        is_promising = promising_text.startswith("YES")
    
    # Ensure summary is under character limit
    if len(forecast_summary) > FORECAST_SUMMARY_LIMIT:
        # Truncate and add ellipsis
        forecast_summary = forecast_summary[:FORECAST_SUMMARY_LIMIT - 3] + "..."
    
    # Log the extracted information
    eprint(f"[INFO] Forecast Summary ({len(forecast_summary)} chars): {forecast_summary}")
    eprint(f"[INFO] Is Promising: {is_promising}")
    
    # Return updated state with forecast analysis results
    return {
        **state,
        "forecast_data": forecast_df.to_dict(),
        "forecast_summary": forecast_summary,
        "is_promising": is_promising,
    }


def check_forecast_quality(state: StockAnalysisState) -> Literal["promising", "not_promising"]:
    """
    Decision node: Route based on forecast quality.
    
    This conditional edge function determines the next node based on
    whether the forecast is promising or not.
    
    Args:
        state: Current graph state with is_promising flag
        
    Returns:
        String indicating which path to take: 'promising' or 'not_promising'
    """
    # Log decision point
    eprint(f"\n[DECISION] Checking forecast quality for {state['symbol']}...")
    
    # Route based on is_promising flag
    if state.get("is_promising", False):
        # Forecast looks good, proceed to news analysis
        eprint("[DECISION] -> Forecast is PROMISING, will analyze news")
        return "promising"
    else:
        # Forecast doesn't look good, skip news analysis
        eprint("[DECISION] -> Forecast is NOT promising, skipping news")
        return "not_promising"


def summarize_news_node(state: StockAnalysisState) -> StockAnalysisState:
    """
    Node 2a: Summarize news articles for promising forecasts.
    
    This node reads news articles, extracts relevant portions,
    and creates a global summary under 1000 characters.
    
    Args:
        state: Current graph state with symbol information
        
    Returns:
        Updated state with news_summary
    """
    # Extract stock symbol from state
    symbol = state["symbol"]
    
    # Log node execution
    eprint(f"\n[NODE] Summarizing news for {symbol}...")
    
    # Load news data from JSON file
    news_data = load_news_json(symbol)
    
    # Handle case where news file is not found
    if news_data is None or not news_data.get("articles"):
        # Return state with default message
        return {
            **state,
            "news_articles": [],
            "news_summary": f"No news articles available for {symbol}."
        }
    
    # Extract articles from news data
    articles = news_data.get("articles", [])
    
    # Store articles in state
    state["news_articles"] = articles
    
    # Log number of articles found
    eprint(f"[INFO] Found {len(articles)} articles to summarize")
    
    # Initialize LLM for news summarization
    llm = get_llm(LLM_MODEL)
    
    # Process articles iteratively to build global summary
    # Start with empty summary
    running_summary = ""
    
    # Limit number of articles to process (for efficiency)
    max_articles = min(10, len(articles))
    
    # Iterate through articles
    for i, article in enumerate(articles[:max_articles]):
        # Extract article components
        title = article.get("title", "No title")
        text = article.get("text", article.get("description", ""))
        source = article.get("source", "Unknown")
        
        # Limit text length for processing
        text_preview = text[:1500] if text else "No content"
        
        # Log article being processed
        eprint(f"[INFO] Processing article {i+1}/{max_articles}: {title[:50]}...")
        
        # Create prompt for extracting relevant portions
        extract_prompt = f"""You are analyzing news for stock {symbol}.

ARTICLE {i+1}:
Title: {title}
Source: {source}
Text: {text_preview}

CURRENT SUMMARY SO FAR:
{running_summary if running_summary else "No summary yet."}

TASK:
1. Extract key points relevant to {symbol}'s stock performance
2. Update the running summary to incorporate this article's insights
3. Keep the total summary UNDER 1000 characters

Provide ONLY the updated summary, no explanations."""
        
        # Send prompt to LLM
        response = llm.invoke([HumanMessage(content=extract_prompt)])
        
        # Update running summary with LLM response
        running_summary = response.content.strip()
        
        # Ensure summary stays under limit
        if len(running_summary) > NEWS_SUMMARY_LIMIT:
            # Truncate if too long
            running_summary = running_summary[:NEWS_SUMMARY_LIMIT - 3] + "..."
    
    # Finalize the news summary
    news_summary = running_summary if running_summary else f"News analyzed for {symbol} but no significant insights found."
    
    # Ensure final summary is under limit
    if len(news_summary) > NEWS_SUMMARY_LIMIT:
        news_summary = news_summary[:NEWS_SUMMARY_LIMIT - 3] + "..."
    
    # Log the final summary
    eprint(f"[INFO] News Summary ({len(news_summary)} chars): {news_summary[:100]}...")
    
    # Return updated state with news summary
    return {
        **state,
        "news_articles": articles,
        "news_summary": news_summary,
    }


def skip_news_node(state: StockAnalysisState) -> StockAnalysisState:
    """
    Node 2b: Skip news analysis for non-promising forecasts.
    
    This node is executed when the forecast is not promising,
    setting a default message for the news summary.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with default news_summary
    """
    # Log node execution
    eprint(f"\n[NODE] Skipping news for {state['symbol']} (forecast not promising)")
    
    # Return state with default message
    return {
        **state,
        "news_articles": [],
        "news_summary": "Forecast is not promising. News analysis skipped.",
    }


def make_investment_decision_node(state: StockAnalysisState) -> StockAnalysisState:
    """
    Node 3: Make investment decision based on forecast and news summaries.
    
    This node reads both summaries and makes one of three decisions:
    - invest: Recommend buying the stock
    - avoid: Recommend not investing
    - neutral: 50/50 call, uncertain
    
    Args:
        state: Current graph state with both summaries
        
    Returns:
        Updated state with decision and decision_summary
    """
    # Extract information from state
    symbol = state["symbol"]
    forecast_summary = state.get("forecast_summary", "No forecast available")
    news_summary = state.get("news_summary", "No news available")
    
    # Log node execution
    eprint(f"\n[NODE] Making investment decision for {symbol}...")
    
    # Initialize LLM for decision making
    llm = get_llm(LLM_MODEL)
    
    # Create prompt for investment decision
    decision_prompt = f"""You are an investment advisor analyzing {symbol} stock.

FORECAST SUMMARY:
{forecast_summary}

NEWS SUMMARY:
{news_summary}

Based on this information, make ONE of these decisions:
1. INVEST - The stock looks promising, recommend buying
2. AVOID - The stock looks risky, recommend not investing  
3. NEUTRAL - Uncertain, it's a 50/50 call

Provide your response in this EXACT format:
DECISION: <INVEST or AVOID or NEUTRAL>
REASONING: <Your reasoning in under 200 characters>
"""
    
    # Send prompt to LLM
    response = llm.invoke([HumanMessage(content=decision_prompt)])
    
    # Extract response content
    response_text = response.content
    
    # Log raw response for debugging
    eprint(f"[DEBUG] Decision Response: {response_text[:200]}...")
    
    # Parse decision from response
    decision = "neutral"  # Default to neutral
    decision_summary = f"{symbol}: Unable to determine investment recommendation."
    
    # Extract decision keyword
    response_upper = response_text.upper()
    if "DECISION:" in response_upper:
        # Find the decision line
        decision_line = response_upper.split("DECISION:")[-1].split("\n")[0].strip()
        
        # Determine which decision was made
        if "INVEST" in decision_line and "AVOID" not in decision_line:
            decision = "invest"
        elif "AVOID" in decision_line:
            decision = "avoid"
        else:
            decision = "neutral"
    
    # Extract reasoning/summary
    if "REASONING:" in response_text:
        # Get the reasoning portion
        reasoning_start = response_text.find("REASONING:") + len("REASONING:")
        decision_summary = response_text[reasoning_start:].strip()
    elif "DECISION:" in response_text:
        # Use everything after decision as summary
        decision_idx = response_text.upper().find("DECISION:")
        remaining = response_text[decision_idx:].split("\n", 1)
        if len(remaining) > 1:
            decision_summary = remaining[1].strip()
    
    # Ensure decision summary is under limit
    if len(decision_summary) > DECISION_SUMMARY_LIMIT:
        decision_summary = decision_summary[:DECISION_SUMMARY_LIMIT - 3] + "..."
    
    # If summary is empty, create a default one
    if not decision_summary or len(decision_summary) < 10:
        if decision == "invest":
            decision_summary = f"{symbol}: Positive outlook based on forecast and market sentiment. Consider investing."
        elif decision == "avoid":
            decision_summary = f"{symbol}: Negative outlook. High risk detected. Avoid investing at this time."
        else:
            decision_summary = f"{symbol}: Mixed signals. This is a 50/50 call. Proceed with caution."
    
    # Ensure summary is under limit after default generation
    if len(decision_summary) > DECISION_SUMMARY_LIMIT:
        decision_summary = decision_summary[:DECISION_SUMMARY_LIMIT - 3] + "..."
    
    # Log the decision
    eprint(f"[INFO] Decision: {decision.upper()}")
    eprint(f"[INFO] Summary ({len(decision_summary)} chars): {decision_summary}")
    
    # Return updated state with decision information
    return {
        **state,
        "decision": decision,
        "decision_summary": decision_summary,
    }


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_stock_analysis_graph() -> StateGraph:
    """
    Build and return the LangGraph state graph for stock analysis.
    
    This function constructs the complete agentic workflow:
    1. Analyze forecast -> 2. Decision (promising?) -> 
       2a. Summarize news OR 2b. Skip news -> 
    3. Make decision -> END
    
    Returns:
        Compiled StateGraph ready for execution
    """
    # Log graph construction
    eprint("\n[GRAPH] Building stock analysis graph...")
    
    # Initialize StateGraph with our state definition
    graph = StateGraph(StockAnalysisState)
    
    # Add nodes to the graph
    # Node 1: Analyze forecast data
    graph.add_node("analyze_forecast", analyze_forecast_node)
    
    # Node 2a: Summarize news (for promising forecasts)
    graph.add_node("summarize_news", summarize_news_node)
    
    # Node 2b: Skip news (for non-promising forecasts)
    graph.add_node("skip_news", skip_news_node)
    
    # Node 3: Make investment decision
    graph.add_node("make_decision", make_investment_decision_node)
    
    # Set entry point of the graph
    graph.set_entry_point("analyze_forecast")
    
    # Add conditional edge after forecast analysis
    # Routes to either summarize_news or skip_news based on is_promising
    graph.add_conditional_edges(
        "analyze_forecast",  # Source node
        check_forecast_quality,  # Condition function
        {
            "promising": "summarize_news",  # If promising, analyze news
            "not_promising": "skip_news",  # If not promising, skip news
        }
    )
    
    # Add edges from news nodes to decision node
    graph.add_edge("summarize_news", "make_decision")
    graph.add_edge("skip_news", "make_decision")
    
    # Add edge from decision to END
    graph.add_edge("make_decision", END)
    
    # Compile the graph into an executable form
    compiled_graph = graph.compile()
    
    # Log successful compilation
    eprint("[GRAPH] Graph compiled successfully!")
    
    # Return the compiled graph
    return compiled_graph


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def analyze_stock(symbol: str, graph) -> StockAnalysisState:
    """
    Run the analysis graph for a single stock symbol.
    
    Args:
        symbol: Stock ticker symbol to analyze
        graph: Compiled LangGraph state graph
        
    Returns:
        Final state containing all analysis results
    """
    # Log analysis start
    eprint(f"\n{'='*60}")
    eprint(f"ANALYZING: {symbol}")
    eprint(f"{'='*60}")
    
    # Create initial state for this stock
    initial_state: StockAnalysisState = {
        "symbol": symbol,
        "forecast_data": None,
        "forecast_summary": "",
        "is_promising": False,
        "news_articles": None,
        "news_summary": "",
        "decision": "",
        "decision_summary": "",
        "error": None,
    }
    
    # Execute the graph with initial state
    # The graph will run through all nodes automatically
    final_state = graph.invoke(initial_state)
    
    # Log completion
    eprint(f"\n[COMPLETE] Analysis finished for {symbol}")
    
    # Return the final state with all results
    return final_state


def save_results_to_file(results: List[StockAnalysisState], output_path: str) -> None:
    """
    Save all analysis results to a formatted text file.
    
    Creates a tabulated summary of all stocks analyzed with their
    forecast summaries, news summaries, and investment decisions.
    
    Args:
        results: List of final states from all stock analyses
        output_path: Path to save the output text file
    """
    # Log save operation
    eprint(f"\n[SAVE] Writing results to {output_path}...")
    
    # Open file for writing
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("STOCK ANALYSIS AGENT REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Write summary table header
        f.write("SUMMARY TABLE\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Symbol':<8} {'Decision':<10} {'Promising':<10}\n")
        f.write("-" * 80 + "\n")
        
        # Write summary row for each stock
        for result in results:
            symbol = result.get("symbol", "N/A")
            decision = result.get("decision", "N/A").upper()
            is_promising = "YES" if result.get("is_promising", False) else "NO"
            f.write(f"{symbol:<8} {decision:<10} {is_promising:<10}\n")
        
        f.write("-" * 80 + "\n\n")
        
        # Write detailed section for each stock
        f.write("DETAILED ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        # Iterate through each result
        for result in results:
            # Extract data from result
            symbol = result.get("symbol", "N/A")
            forecast_summary = result.get("forecast_summary", "N/A")
            news_summary = result.get("news_summary", "N/A")
            decision = result.get("decision", "N/A").upper()
            decision_summary = result.get("decision_summary", "N/A")
            is_promising = result.get("is_promising", False)
            
            # Write stock header
            f.write(f"STOCK: {symbol}\n")
            f.write("-" * 40 + "\n")
            
            # Write forecast summary section
            f.write(f"\n[FORECAST SUMMARY] ({len(forecast_summary)} chars)\n")
            f.write(f"{forecast_summary}\n")
            
            # Write promising flag
            f.write(f"\n[FORECAST QUALITY]\n")
            f.write(f"{'Promising' if is_promising else 'Not Promising'}\n")
            
            # Write news summary section
            f.write(f"\n[NEWS SUMMARY] ({len(news_summary)} chars)\n")
            f.write(f"{news_summary}\n")
            
            # Write decision section
            f.write(f"\n[INVESTMENT DECISION]\n")
            f.write(f"Recommendation: {decision}\n")
            
            # Write decision summary
            f.write(f"\n[DECISION SUMMARY] ({len(decision_summary)} chars)\n")
            f.write(f"{decision_summary}\n")
            
            # Add separator between stocks
            f.write("\n" + "=" * 80 + "\n\n")
        
        # Write footer
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    # Log successful save
    eprint(f"[SAVE] Results saved successfully to {output_path}")


def main() -> int:
    """
    Main entry point for the stock analysis agent.
    
    Orchestrates the entire analysis flow:
    1. Build the LangGraph
    2. Iterate through each stock symbol
    3. Run analysis for each stock
    4. Save results to output file
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Print startup banner
    eprint("\n" + "=" * 60)
    eprint("STOCK ANALYSIS AGENT")
    eprint("Using LangGraph + LangChain + Ollama (Llama 3.2)")
    eprint("=" * 60)
    
    # Check if Ollama is available
    eprint("\n[INIT] Checking LLM availability...")
    try:
        # Test LLM connection
        test_llm = get_llm()
        test_response = test_llm.invoke([HumanMessage(content="Say 'OK' if you're ready.")])
        eprint(f"[INIT] LLM test response: {test_response.content[:50]}...")
    except Exception as e:
        # Handle LLM connection failure
        eprint(f"[ERROR] Failed to connect to Ollama LLM: {e}")
        eprint("[ERROR] Please ensure Ollama is running with llama3.2 model installed.")
        eprint("[ERROR] Install with: ollama pull llama3.2")
        return 1
    
    # Build the analysis graph
    try:
        graph = build_stock_analysis_graph()
    except Exception as e:
        # Handle graph build failure
        eprint(f"[ERROR] Failed to build graph: {e}")
        return 1
    
    # Initialize results list
    all_results: List[StockAnalysisState] = []
    
    # Process each stock symbol
    for symbol in STOCK_SYMBOLS:
        try:
            # Run analysis for this stock
            result = analyze_stock(symbol, graph)
            
            # Add result to list
            all_results.append(result)
            
        except Exception as e:
            # Handle analysis failure for individual stock
            eprint(f"[ERROR] Failed to analyze {symbol}: {e}")
            
            # Add error result
            error_result: StockAnalysisState = {
                "symbol": symbol,
                "forecast_data": None,
                "forecast_summary": f"Error analyzing {symbol}",
                "is_promising": False,
                "news_articles": None,
                "news_summary": f"Error: {str(e)}",
                "decision": "avoid",
                "decision_summary": f"Analysis failed for {symbol}. Error: {str(e)[:100]}",
                "error": str(e),
            }
            all_results.append(error_result)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"stock_analysis_report_{timestamp}.txt"
    output_path = os.path.join(OUTPUTS_DIR, output_filename)
    
    # Save results to file
    try:
        save_results_to_file(all_results, output_path)
    except Exception as e:
        eprint(f"[ERROR] Failed to save results: {e}")
        return 1
    
    # Print final summary to console
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nAnalyzed {len(all_results)} stocks:")
    
    # Print summary for each stock
    for result in all_results:
        symbol = result.get("symbol", "N/A")
        decision = result.get("decision", "N/A").upper()
        is_promising = "✓" if result.get("is_promising", False) else "✗"
        print(f"  {symbol}: {decision} (Promising: {is_promising})")
    
    print(f"\nDetailed report saved to: {output_path}")
    print("=" * 60)
    
    # Return success
    return 0


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Execute main function and exit with its return code
    raise SystemExit(main())
