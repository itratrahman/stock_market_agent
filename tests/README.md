# Tests Directory

This directory contains unit tests and integration tests for the project.

## Structure

Organize tests to mirror the source code structure:
```
tests/
├── test_data_loader.py      # Tests for data loading utilities
├── test_models.py            # Tests for model implementations
├── test_preprocessing.py     # Tests for data preprocessing
└── test_trading_agent.py     # Tests for trading logic
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
