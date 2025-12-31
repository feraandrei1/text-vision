# Contributing to Text Vision

Thank you for your interest in contributing to Text Vision!

## Code of Conduct

Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Bugs

1. Check existing issues first
2. Create a new issue with:
   - Clear description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (Python version, OS, dependencies)
   - Sample images (if applicable)

### Feature Requests

1. Check existing issues/discussions
2. Create an issue with "enhancement" label
3. Describe the feature and use case

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes following coding standards
4. Add/update tests
5. Update documentation
6. Commit: `git commit -m "Add: description"`
7. Push: `git push origin feature/your-feature`
8. Create Pull Request

## Development Setup

```bash
git clone https://github.com/text-vision/text-vision.git
cd text-vision
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
pytest tests/
```

## Coding Standards

- Follow PEP 8
- Use type hints
- Write docstrings for public methods
- Add unit tests for new functionality
- Keep functions focused and single-purpose

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_platevision.py -v
```

## Project Structure

```
text-vision/
├── src/                  # Source code
│   ├── text_identifier.py   # Unified API
│   ├── plates.py            # License plate recognition
│   ├── ids.py               # ID card text extraction
│   ├── documents.py         # PDF/document processing
│   └── ...
├── tests/                # Unit tests
├── examples/             # Usage examples
└── models/               # ML models (gitignored)
```

## Questions?

Open an issue with the "question" label.
