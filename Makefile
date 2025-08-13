.PHONY: help install install-dev test test-cov lint format type-check security clean build docs

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black and isort"
	@echo "  type-check   - Run type checking with mypy"
	@echo "  security     - Run security checks"
	@echo "  clean        - Clean up generated files"
	@echo "  build        - Build package"
	@echo "  docs         - Build documentation"
	@echo "  ci           - Run all CI checks"
	@echo "  pre-commit   - Run pre-commit hooks"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -v -m "not slow"

test-integration:
	pytest tests/ -v -m "integration"

# Code quality
lint:
	flake8 src/ tests/ scripts/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src/ tests/ scripts/ --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

format-check:
	black --check src/ tests/ scripts/
	isort --check-only src/ tests/ scripts/

type-check:
	mypy src/ --ignore-missing-imports

# Security
security:
	bandit -r src/ -f json -o bandit-report.json
	safety check --json --output safety-report.json

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf docs/_build/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Build
build:
	python -m build

# Documentation
docs:
	cd docs && make html

# CI/CD
ci: lint format-check type-check test-cov security
	@echo "All CI checks passed!"

pre-commit: format lint type-check test-fast
	@echo "Pre-commit checks passed!"

# Development helpers
setup-dev: install-dev
	@echo "Development environment setup complete!"

run-demo:
	python scripts/demo_submission.py

run-analysis:
	python scripts/run_full_analysis.py



# Docker (if needed)
docker-build:
	docker build -t lrd-project .

docker-run:
	docker run -it --rm lrd-project

# Database operations (if needed)
db-migrate:
	@echo "Database migration placeholder"

db-seed:
	@echo "Database seeding placeholder"

# Performance testing
benchmark:
	python scripts/benchmark_analysis.py

# Release
release: clean build test-cov
	@echo "Release preparation complete!"

# Helpers for specific tasks
check-submission:
	python -c "from src.submission import *; print('Submission system imports successfully')"

validate-config:
	python scripts/test_config.py

# Environment setup
setup-env:
	python -m venv fractal-env
	@echo "Virtual environment created. Activate with: source fractal-env/bin/activate (Linux/Mac) or fractal-env\\Scripts\\activate (Windows)"

# Data operations
setup-data:
	python scripts/setup_data.py

generate-synthetic:
	python scripts/generate_synthetic_data.py

# Analysis pipeline
run-pipeline:
	python scripts/run_full_analysis.py

# Monitoring and logging
check-logs:
	@echo "Recent log entries:"
	@find . -name "*.log" -exec tail -n 10 {} \; 2>/dev/null || echo "No log files found"

# Backup and restore
backup:
	tar -czf backup-$(shell date +%Y%m%d-%H%M%S).tar.gz --exclude='.git' --exclude='fractal-env' --exclude='__pycache__' .

# Dependencies
update-deps:
	pip install --upgrade pip
	pip install --upgrade -r requirements.txt

check-deps:
	pip list --outdated

# Git helpers
git-status:
	git status

git-diff:
	git diff

git-log:
	git log --oneline -10

# Project statistics
stats:
	@echo "Project statistics:"
	@echo "Python files: $(shell find src/ -name '*.py' | wc -l)"
	@echo "Test files: $(shell find tests/ -name '*.py' | wc -l)"
	@echo "Total lines of code: $(shell find src/ -name '*.py' -exec wc -l {} + | tail -1)"
	@echo "Total test lines: $(shell find tests/ -name '*.py' -exec wc -l {} + | tail -1)"
