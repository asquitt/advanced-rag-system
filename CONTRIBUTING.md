# Contributing

## Development Setup

1. Clone repository
2. Create virtual environment: `python -m venv venv`
3. Install dependencies: `pip install -r requirements.txt`
4. Start services: `docker-compose up -d`
5. Run tests: `pytest`

## Code Style

- Use Black for formatting: `black src/`
- Type hints for all functions
- Docstrings for all public methods

## Testing

- Add tests for new features in `tests/`
- Ensure all tests pass before committing
- Aim for >80% coverage

## Pull Requests

- Create feature branch: `git checkout -b feature/your-feature`
- Make changes and test
- Submit PR with clear description
