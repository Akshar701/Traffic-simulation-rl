# Contributing to Traffic Simulation & RL Project

Thank you for your interest in contributing to our traffic simulation and reinforcement learning project! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug Reports**: Report bugs and issues you encounter
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit pull requests with code changes
- **Documentation**: Improve or add documentation
- **Testing**: Help test the project and report issues
- **Examples**: Create example scripts or use cases

### Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** to your local machine
3. **Create a new branch** for your changes
4. **Make your changes** following our guidelines
5. **Test your changes** thoroughly
6. **Submit a pull request** with a clear description

## 🏗️ Development Setup

### Prerequisites

- Python 3.8+
- SUMO (Simulation of Urban MObility)
- Git

### Local Development Environment

1. **Clone and setup**
   ```bash
   git clone https://github.com/yourusername/traffic-simulation-rl.git
   cd traffic-simulation-rl
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## 📝 Code Style Guidelines

### Python Code Style

- Follow **PEP 8** style guidelines
- Use **type hints** for function parameters and return values
- Keep functions focused and under 50 lines when possible
- Use descriptive variable and function names

### Example of Good Code Style

```python
from typing import List, Dict, Optional
import numpy as np

def calculate_traffic_flow(
    vehicle_counts: List[int], 
    time_intervals: List[float]
) -> Dict[str, float]:
    """
    Calculate traffic flow metrics from vehicle count data.
    
    Args:
        vehicle_counts: List of vehicle counts per interval
        time_intervals: List of time intervals in seconds
        
    Returns:
        Dictionary containing flow rate and average density
    """
    if not vehicle_counts or not time_intervals:
        raise ValueError("Input lists cannot be empty")
    
    flow_rate = sum(vehicle_counts) / sum(time_intervals)
    avg_density = np.mean(vehicle_counts)
    
    return {
        "flow_rate": flow_rate,
        "avg_density": avg_density
    }
```

### File Organization

- Keep related functionality in the same module
- Use clear, descriptive file names
- Group imports: standard library, third-party, local
- Separate configuration from logic

## 🧪 Testing Guidelines

### Writing Tests

- Write tests for all new functionality
- Use descriptive test names that explain the expected behavior
- Test both success and failure cases
- Mock external dependencies when appropriate

### Example Test

```python
import pytest
from traffic_metrics import calculate_traffic_flow

def test_calculate_traffic_flow_success():
    """Test successful traffic flow calculation."""
    vehicle_counts = [10, 15, 20]
    time_intervals = [60, 60, 60]
    
    result = calculate_traffic_flow(vehicle_counts, time_intervals)
    
    assert result["flow_rate"] == 0.75  # 45 vehicles / 180 seconds
    assert result["avg_density"] == 15.0

def test_calculate_traffic_flow_empty_input():
    """Test that empty input raises ValueError."""
    with pytest.raises(ValueError, match="Input lists cannot be empty"):
        calculate_traffic_flow([], [])
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_traffic_metrics.py

# Run tests in parallel
pytest -n auto
```

## 📚 Documentation Standards

### Code Documentation

- Use **docstrings** for all public functions and classes
- Follow **Google docstring** format
- Include examples in docstrings when helpful
- Document exceptions that may be raised

### Example Docstring

```python
def optimize_traffic_signals(
    intersection_data: Dict[str, Any],
    optimization_algorithm: str = "q_learning"
) -> Dict[str, Any]:
    """
    Optimize traffic signal timing using specified algorithm.
    
    Args:
        intersection_data: Dictionary containing intersection configuration
            and current traffic state
        optimization_algorithm: Algorithm to use for optimization.
            Options: "q_learning", "deep_q", "policy_gradient"
            
    Returns:
        Dictionary containing optimized signal timings and performance metrics
        
    Raises:
        ValueError: If optimization_algorithm is not supported
        RuntimeError: If optimization fails to converge
        
    Example:
        >>> data = {"lanes": 4, "traffic_flow": [100, 80, 120, 90]}
        >>> result = optimize_traffic_signals(data, "q_learning")
        >>> print(result["optimized_timings"])
        [30, 25, 35, 30]
    """
```

### README and Documentation

- Keep README.md up to date
- Document new features and changes
- Include usage examples
- Update installation instructions when dependencies change

## 🔄 Pull Request Process

### Before Submitting

1. **Ensure tests pass** locally
2. **Update documentation** if needed
3. **Check code style** with linters
4. **Rebase on main** if there are conflicts

### Pull Request Template

```markdown
## Description
Brief description of changes made

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (please describe)

## Testing
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes introduced

## Additional Notes
Any additional information or context
```

### Review Process

- All PRs require at least one review
- Address review comments promptly
- Maintainers may request changes before merging
- Squash commits before merging if requested

## 🐛 Bug Reports

### Bug Report Template

```markdown
**Bug Description**
Clear description of the bug

**Steps to Reproduce**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., macOS 12.0]
- Python Version: [e.g., 3.9.0]
- SUMO Version: [e.g., 1.8.0]

**Additional Context**
Any other relevant information
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Feature Description**
Clear description of the requested feature

**Use Case**
Why this feature would be useful

**Proposed Implementation**
How you think it could be implemented

**Alternatives Considered**
Other approaches you've considered

**Additional Context**
Any other relevant information
```

## 🏷️ Issue Labels

We use the following labels to categorize issues:

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `priority: high`: High priority issue
- `priority: low`: Low priority issue
- `status: blocked`: Blocked by other issues
- `status: in progress`: Work in progress

## 📞 Getting Help

### Communication Channels

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Requests**: For code contributions

### Before Asking for Help

1. **Check existing issues** for similar problems
2. **Review documentation** and README
3. **Search closed issues** for solutions
4. **Provide minimal reproduction** of the issue

## 🎯 Contribution Ideas

### Good First Issues

- Fix typos in documentation
- Add missing docstrings
- Improve error messages
- Add unit tests for existing functions
- Update requirements.txt versions

### Advanced Contributions

- Implement new RL algorithms
- Add new traffic scenarios
- Improve dashboard visualizations
- Optimize performance bottlenecks
- Add CI/CD pipelines

## 📋 Code of Conduct

### Our Standards

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Be patient with newcomers

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Publishing private information
- Any other unprofessional conduct

## 🏆 Recognition

### Contributors

- All contributors are listed in the README
- Significant contributions are highlighted
- Contributors are acknowledged in releases

### Hall of Fame

- Top contributors get special recognition
- Long-term contributors become maintainers
- Outstanding contributions are featured

---

Thank you for contributing to our project! Your contributions help make traffic simulation and RL more accessible and effective for everyone.
