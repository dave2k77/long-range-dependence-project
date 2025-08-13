# Continuous Integration and Deployment Guide

This guide explains the CI/CD setup for the Long-Range Dependence Analysis Project, including testing, code quality checks, and deployment procedures.

## Overview

The project uses a comprehensive CI/CD pipeline with the following components:

- **GitHub Actions** for automated testing and deployment
- **Pre-commit hooks** for code quality enforcement
- **Docker** for containerized development and deployment
- **Comprehensive testing** with pytest and coverage reporting
- **Code quality tools** including linting, formatting, and type checking
- **Security scanning** with bandit and safety

## CI/CD Pipeline

### GitHub Actions Workflow

The main CI workflow (`.github/workflows/ci.yml`) includes:

1. **Multi-Python Testing**: Tests against Python 3.8, 3.9, 3.10, and 3.11
2. **Code Quality Checks**: Linting, formatting, and type checking
3. **Test Coverage**: Comprehensive test coverage reporting
4. **Security Scanning**: Automated security vulnerability checks
5. **Documentation Building**: Automated documentation generation
6. **Submission System Testing**: Specialized tests for the submission system

### Workflow Jobs

#### 1. Test Job
- Runs on multiple Python versions
- Installs dependencies
- Runs linting (flake8)
- Checks code formatting (black, isort)
- Performs type checking (mypy)
- Runs tests with coverage reporting
- Uploads coverage to Codecov

#### 2. Submission System Test Job
- Tests the submission system specifically
- Runs demo submission script
- Tests integration with full analysis pipeline

#### 3. Security Job
- Runs bandit for security analysis
- Checks dependencies with safety
- Generates security reports

#### 4. Documentation Job
- Builds documentation with Sphinx
- Uploads documentation artifacts

## Local Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Docker (optional)
- Make (optional, for using Makefile commands)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd long-range-dependence-project
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv fractal-env
   source fractal-env/bin/activate  # Linux/Mac
   # or
   fractal-env\Scripts\activate     # Windows
   ```

3. **Install dependencies**:
   ```bash
   # For development
   pip install -e ".[dev]"
   
   # Or using requirements.txt
   pip install -r requirements.txt
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

### Using Makefile Commands

The project includes a comprehensive Makefile with common development tasks:

```bash
# Show all available commands
make help

# Install dependencies
make install-dev

# Run tests
make test
make test-cov

# Code quality checks
make lint
make format
make type-check

# Security checks
make security

# Run all CI checks locally
make ci

# Clean up generated files
make clean
```

## Testing

### Test Structure

- **Unit Tests**: `tests/test_*.py`
- **Integration Tests**: Marked with `@pytest.mark.integration`
- **Submission System Tests**: `tests/test_submission_system.py`

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_submission_system.py -v

# Run tests by marker
pytest tests/ -v -m "integration"
pytest tests/ -v -m "not slow"

# Run tests in parallel
pytest tests/ -n auto
```

### Test Coverage

The project aims for at least 80% test coverage. Coverage reports are generated in:
- HTML format: `htmlcov/index.html`
- XML format: `coverage.xml` (for CI integration)

## Code Quality

### Linting

The project uses flake8 for linting with custom configuration:

```bash
# Run linting
flake8 src/ tests/ scripts/

# Check for specific error types
flake8 src/ tests/ scripts/ --select=E9,F63,F7,F82
```

### Code Formatting

Black and isort are used for consistent code formatting:

```bash
# Format code
black src/ tests/ scripts/
isort src/ tests/ scripts/

# Check formatting without changing files
black --check src/ tests/ scripts/
isort --check-only src/ tests/ scripts/
```

### Type Checking

MyPy is used for static type checking:

```bash
# Run type checking
mypy src/ --ignore-missing-imports
```

## Security

### Security Scanning

The project includes automated security checks:

```bash
# Run bandit security scanner
bandit -r src/ -f json -o bandit-report.json

# Check dependencies for vulnerabilities
safety check --json --output safety-report.json
```

### Security Best Practices

1. **Dependency Management**: Regular updates and vulnerability scanning
2. **Code Review**: All changes require code review
3. **Secrets Management**: Use environment variables for sensitive data
4. **Input Validation**: Validate all user inputs
5. **Error Handling**: Proper error handling without information leakage

## Docker Development

### Using Docker Compose

The project includes a comprehensive Docker Compose setup:

```bash
# Start all services
docker-compose up -d

# Run tests in container
docker-compose run test

# Run linting in container
docker-compose run lint

# Start development shell
docker-compose run dev

# Start Jupyter notebook
docker-compose up jupyter
```

### Docker Services

- **app**: Main application service
- **redis**: Caching and job queue (if needed)
- **jupyter**: Interactive development environment
- **test**: Testing service
- **lint**: Code quality checks
- **security**: Security scanning
- **docs**: Documentation building
- **dev**: Development shell

## Pre-commit Hooks

### Automatic Checks

Pre-commit hooks run automatically on every commit:

1. **File Formatting**: Trailing whitespace, end-of-file
2. **YAML/JSON Validation**: Syntax checking
3. **Code Formatting**: Black and isort
4. **Linting**: Flake8
5. **Type Checking**: MyPy
6. **Security**: Bandit and safety
7. **Custom Checks**: Submission system validation

### Manual Pre-commit

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Run specific hook
pre-commit run black

# Update pre-commit hooks
pre-commit autoupdate
```

## Deployment

### Production Deployment

1. **Build Docker Image**:
   ```bash
   docker build -t lrd-project .
   ```

2. **Run Container**:
   ```bash
   docker run -d -p 8000:8000 lrd-project
   ```

### Environment Variables

Configure the following environment variables for production:

```bash
# Application settings
PYTHONPATH=/app
PYTHONUNBUFFERED=1

# Database settings (if applicable)
DATABASE_URL=postgresql://user:pass@host:port/db

# Security settings
SECRET_KEY=your-secret-key
DEBUG=False

# External services
REDIS_URL=redis://redis:6379
```

## Monitoring and Logging

### Logging Configuration

The project uses structured logging:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Health Checks

Docker containers include health checks:

```bash
# Check container health
docker ps

# View container logs
docker logs lrd-project-app
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PYTHONPATH is set correctly
2. **Test Failures**: Check test dependencies and environment
3. **Docker Issues**: Verify Docker and Docker Compose installation
4. **Pre-commit Failures**: Run `pre-commit install` and update hooks

### Debugging

```bash
# Run tests with verbose output
pytest tests/ -v -s

# Debug Docker containers
docker-compose logs

# Check system resources
docker stats

# Access container shell
docker-compose exec app bash
```

## Best Practices

### Development Workflow

1. **Create Feature Branch**: `git checkout -b feature/new-feature`
2. **Make Changes**: Implement your feature
3. **Run Tests**: `make test` or `pytest tests/`
4. **Code Quality**: `make ci` to run all checks
5. **Commit**: `git commit -m "Add new feature"`
6. **Push**: `git push origin feature/new-feature`
7. **Create PR**: Submit pull request for review

### Code Review Checklist

- [ ] Tests pass
- [ ] Code coverage maintained
- [ ] Linting passes
- [ ] Type checking passes
- [ ] Security checks pass
- [ ] Documentation updated
- [ ] No breaking changes

### Performance Considerations

- Use appropriate test markers (`@pytest.mark.slow`)
- Optimize Docker images with multi-stage builds
- Cache dependencies in CI/CD
- Use parallel testing when possible

## Contributing

### Submission Guidelines

1. **Follow Code Style**: Use Black and isort formatting
2. **Write Tests**: Include tests for new functionality
3. **Update Documentation**: Keep docs current
4. **Security**: Follow security best practices
5. **Performance**: Consider performance implications

### Review Process

1. **Automated Checks**: CI/CD pipeline must pass
2. **Code Review**: At least one approval required
3. **Testing**: All tests must pass
4. **Documentation**: Documentation must be updated

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Pre-commit Documentation](https://pre-commit.com/)
- [Docker Documentation](https://docs.docker.com/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Black Documentation](https://black.readthedocs.io/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
