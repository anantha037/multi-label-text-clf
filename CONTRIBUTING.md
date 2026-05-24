# Contributing to multi-label-text-clf

Thank you for your interest in contributing! We welcome all contributions to improve the pipeline and models.

## Development Environment Setup

1. Fork the repository and clone it locally.
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Or .\venv\Scripts\activate on Windows
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Verify your setup by running the smoke test or checking the codebase structure.

## Branching Convention

- All new development should happen on feature branches branching off from `main`.
- Name your branch descriptively, e.g., `feature/add-new-endpoint` or `bugfix/fix-preprocessing-bug`.
- Open a Pull Request targeting `main` when your work is ready for review.

## Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) standard. Please structure your commit messages as follows:

```
<type>: <description>
```

Accepted `<type>` values:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `ci`: Changes to our CI configuration files and scripts
