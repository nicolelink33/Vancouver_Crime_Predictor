# Contributing to Vancouver Crime Predictor

Thank you for your interest in contributing to the Vancouver Crime Predictor project! We welcome contributions from the community and appreciate your help in making this project better.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion for improvement:

1. **Check existing issues** first to see if someone has already reported it
2. **Open a new issue** on GitHub with a clear title and description
3. **Include details** such as:
   - What you expected to happen
   - What actually happened
   - Steps to reproduce the issue
   - Your environment (OS, Python version, etc.)

### Suggesting Enhancements

We're open to new ideas! If you have a feature request or enhancement suggestion:

1. Open an issue with the tag "enhancement"
2. Describe the feature and why it would be useful
3. If possible, outline how you think it could be implemented

### Submitting Pull Requests

We follow a standard GitHub workflow for contributions:

1. **Fork the repository** and create a new branch from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards (see below)

3. **Test your changes** to ensure they work as expected and don't break existing functionality

4. **Commit your changes** with clear, descriptive commit messages
   ```bash
   git commit -m "Add feature: brief description of what you did"
   ```

5. **Push to your fork** and submit a pull request to our `main` branch

6. **Wait for review** - one of the project maintainers will review your PR and may request changes

## Coding Standards

To keep the codebase consistent and readable:

- **Python style**: Follow PEP 8 guidelines
- **Code comments**: Add comments for complex logic, but keep code as self-explanatory as possible
- **Variable names**: Use descriptive names (e.g., `crime_count` instead of `cc`)
- **Notebook cells**: Keep cells focused on one task and include markdown explanations
- **Documentation**: Update the README or other docs if your changes affect usage

## Development Environment Setup

To set up your development environment:

1. Clone your fork of the repository
2. Create a conda environment using our `environment.yml`:
   ```bash
   conda env create -f environment.yml
   conda activate Vancouver_Crime_Predictor
   ```
3. Make your changes and test them locally

## Questions?

If you have questions about contributing, feel free to:
- Open an issue with the "question" tag
- Reach out to the project maintainers

## Code of Conduct

Please note that this project follows our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code and help maintain a welcoming environment for everyone.

---

Thank you for contributing to Vancouver Crime Predictor! 🚀
