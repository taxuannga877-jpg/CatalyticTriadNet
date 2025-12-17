# Contributing to CatalyticTriadNet

Thank you for your interest in contributing to CatalyticTriadNet! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a branch for your changes

## How to Contribute

### Reporting Bugs

- Use the GitHub issue tracker
- Check if the issue already exists
- Include detailed steps to reproduce
- Provide system information (OS, Python version, PyTorch version)

### Suggesting Features

- Open an issue with the "enhancement" label
- Describe the feature and its use case
- Discuss implementation approaches if possible

### Code Contributions

1. **Bug fixes**: Reference the issue number in your PR
2. **New features**: Discuss in an issue first
3. **Documentation**: Always welcome!
4. **Tests**: Help improve coverage

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/CatalyticTriadNet.git
cd CatalyticTriadNet

# Create development environment
conda create -n catalytic-dev python=3.9
conda activate catalytic-dev

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Pull Request Process

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   pytest tests/
   ```

4. **Commit with clear messages**
   ```bash
   git commit -m "feat: add new triad detection algorithm"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **PR Review**
   - Address reviewer feedback
   - Keep the PR focused and reasonably sized

## Style Guidelines

### Python Code Style

- Follow PEP 8
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use meaningful variable names

```python
# Good
def detect_catalytic_triads(
    residues: List[Dict],
    coords: np.ndarray,
    threshold: float = 0.5
) -> List[Dict]:
    """
    Detect catalytic triads in protein structure.

    Args:
        residues: List of residue dictionaries
        coords: CA coordinates array [N, 3]
        threshold: Detection confidence threshold

    Returns:
        List of detected triads with confidence scores
    """
    pass
```

### Commit Messages

Follow conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `refactor:` Code refactoring
- `style:` Formatting changes

### Documentation

- Use Google-style docstrings
- Update README for user-facing changes
- Add examples for new features

## Questions?

Feel free to open an issue or reach out to the maintainers.

Thank you for contributing!
