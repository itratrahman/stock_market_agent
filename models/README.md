# Models Directory

This directory stores trained NeuralProphet models for stock price forecasting.

## Current Models

Trained NeuralProphet models for the following stocks:
- `AAPL_neuralprophet` - Apple Inc.
- `AMZN_neuralprophet` - Amazon.com Inc.
- `GOOGL_neuralprophet` - Alphabet Inc. (Google)
- `MSFT_neuralprophet` - Microsoft Corporation
- `NVDA_neuralprophet` - NVIDIA Corporation

## Model Details

- **Type**: NeuralProphet (PyTorch-based forecasting model)
- **Training**: 100 epochs (default)
- **Features**: 
  - Linear growth trend
  - Yearly seasonality
  - 10 changepoints
  - Huber loss function

## Loading Models

```python
from neuralprophet import load

# Load a trained model
model = load('models/AAPL_neuralprophet')

# Make predictions
future = model.make_future_dataframe(df, periods=30)
forecast = model.predict(future)
```

## Retraining

Retrain models with:
```bash
python train_models.py
```

## Note

Model files are not tracked in git due to their size. Training logs are available in `lightning_logs/` directory.
