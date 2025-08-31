# Contributing to RAG Knowledge Base System

Thank you for your interest in contributing to the RAG Knowledge Base System! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Bugs
- Use the GitHub issue tracker
- Include a clear description of the bug
- Provide steps to reproduce the issue
- Include your Python version and operating system

### Suggesting Enhancements
- Use the GitHub issue tracker with the "enhancement" label
- Describe the feature and its benefits
- Provide use cases if applicable

### Code Contributions
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the coding standards
4. Add tests for new functionality
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add some amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📋 Coding Standards

### Python Code Style
- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise

### File Naming
- Use snake_case for Python files
- Use descriptive names that indicate functionality
- Follow the existing naming conventions in the project

### Documentation
- Update README.md if adding new features
- Add docstrings to new functions
- Include examples in docstrings

## 🧪 Testing

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run tests with coverage
pytest --cov=.

# Run specific test file
pytest test_rag.py
```

### Writing Tests
- Write tests for new functionality
- Use descriptive test names
- Mock external dependencies (OpenAI API, file system)
- Test both success and error cases

## 🔧 Development Setup

### Local Development
1. Clone your fork
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Install development dependencies: `pip install -e .[dev]`
5. Set up your `.env` file with test API keys

### Pre-commit Hooks
Consider setting up pre-commit hooks for:
- Code formatting (black)
- Linting (flake8)
- Type checking (mypy)

## 📝 Pull Request Guidelines

### Before Submitting
- [ ] Code follows the project's style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] No sensitive information is included

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Other (please describe)

## Testing
- [ ] Tests added/updated
- [ ] All tests pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
```

## 🏷️ Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed

## 📞 Getting Help

- Check existing issues and pull requests
- Join discussions in issues
- Ask questions in issues with the "question" label

## 🎉 Recognition

Contributors will be recognized in:
- The project README
- Release notes
- GitHub contributors page

Thank you for contributing to the RAG Knowledge Base System!
