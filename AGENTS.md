# AGENTS.md

## Primary Directives

When uncertain about implementation details or requirements, ask for clarification before proceeding.
Break down the task at hand and chalk out a rough plan for carrying it out, referencing project conventions and best practices.

## Core Commands

Add a dependency
```bash
uv add pydantic~=2.9.0
```
Install dependencies
```bash
uv sync --group dev
```
Run tests
```bash
uv run pytest
```
Apply formatting and Check code quality and Validate types. Always run after the end of a task.
```bash
uv run ruff check --fix . && uv run ruff format . && uv run mypy .
```

## Code Documentation

- **Code Documentation**: Google-style docstrings for public functions/classes.
- **Enduser and Developer Documentation**: Markdown in ./docs folder.
  - Keep documentation short and to the point.
  - Information about implementation / usage should be in a single place. If needed somewhere else, use references.
  - Focus documentation on the use case. Ask if you want to add information like troubleshooting or performance optimization.

## Secure Coding Practices

- Store credentials in environment variables
- Validate all external inputs with pydantic or pydantic-settings
- Use parameterized queries for databases
- Apply principle of least privilege
- Base examples and documentation on fictional scenarios instead of references from instructions.

## Development Guidelines

- All changes must be tested. If you're not testing your changes, you're not done.
- If **outdated** tests block your implementation suggest changing them.
- Get your tests to pass. If you didn't run the tests, your code does not work.
- Follow existing code style. Check neighboring files for patterns.
- Avoid writing significant amounts of new code. This is often a sign that we're missing an existing method or mechanism that could help solve the problem. Look for existing utilities first.

## Git Workflow

- **Granular commits**: One logical change per commit. The LEAST is ONE commit per issue.
- **Clear commit messages**: Explain the *why*; link to issues/ADRs if architectural. If an issue is fixed reference it as closes #XX. One reference per issue.
- **Always use conventional commit standard with gitmoji**: e.g. "feat:" "perf: :zap:"
- **Never commit to main**: Always work with separate branches and Pull Requests.
- **Keep Commit messages concise** - Keep commit message length below 300 characters
