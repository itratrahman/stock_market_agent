# Tests Directory

This directory contains unit tests and integration tests for the stock market agent project.

## Current Status

Test suite is under development. Planned test coverage:

## Planned Structure

```
tests/
├── test_pull_stock_data.py  # Tests for pull_latest_stock.py
├── test_train_models.py      # Tests for train_models.py
├── test_data_loader.py       # Tests for data loading utilities
├── test_models.py            # Tests for model predictions
└── test_integration.py       # End-to-end integration tests
```

## Running Tests

Run all tests:
```bash
pytest
```

Run specific test file:
```bash
pytest tests/test_data_loader.py
```

Run with coverage:
```bash
pytest --cov=src tests/
```

## Test Guidelines

- Write tests for all critical functionality
- Use meaningful test names that describe what is being tested
- Include both positive and negative test cases
- Mock external API calls to avoid dependencies
- Aim for high code coverage (>80%)
