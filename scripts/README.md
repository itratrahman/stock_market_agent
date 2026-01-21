# Scripts Directory

This directory contains utility scripts and automation tools.

## Purpose

Store standalone scripts for:
- Data preprocessing and cleaning
- Model training automation
- Backtesting and evaluation
- Deployment and monitoring
- Database management
- Report generation

## Examples

```
scripts/
├── train_model.py           # Training pipeline script
├── backtest.py              # Backtesting trading strategies
├── evaluate_performance.py  # Model evaluation metrics
├── export_predictions.py    # Export model predictions
└── cleanup_data.py          # Data maintenance utilities
```

## Usage

Scripts should be:
- Executable from the command line
- Well-documented with argument descriptions
- Idempotent when possible
- Include error handling and logging

Run scripts from the project root:
```bash
python scripts/train_model.py --config config.yaml
```
