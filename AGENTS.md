# AGENTS.md

## Primary Directive
When uncertain about implementation details or requirements, ask for clarification before proceeding.

## Project Structure and Organization

### Directory Layout
```
project/
├── src/              # Primary workspace for all code changes
├── tests/            # Read-only - owned by humans
├── docs/             # Update when code changes affect usage
├── .github/          # Requires explicit permission
└── pyproject.toml    # Modifiable with approval
```

### Working Guidelines
- Focus modifications on `src/` directory
- Check for local AGENTS.md files in subdirectories before editing
- Preserve existing file organization patterns
- Maintain separation between source and test code

## Build, Test, and Development Commands

### Environment Setup
- Language: Python 3.12+
- Package Manager: UV

### Core Commands
```bash
uv sync --group dev        # Install dependencies
uv run pytest             # Verify changes work
uv run ruff check .       # Check code quality
uv run ruff format .      # Apply formatting
uv run mypy src/          # Validate types
```

### Command Usage
- Run tests after implementing features
- Apply formatting before committing
- Use linting feedback to improve code quality

## Code Style and Conventions

### Python Standards
```python
from __future__ import annotations  # Always include

def function_name(param: str) -> dict[str, Any]:
    """Google-style docstring for public functions."""
    return {"result": param}
```

### Naming Patterns
- Functions and variables: `snake_case`
- Classes: `PascalCase`
- Constants: `SCREAMING_SNAKE`

### String and Format Rules
- Use double quotes for strings
- Limit lines to 120 characters
- Sort imports alphabetically
- Type annotate public interfaces

## Architecture and Design Patterns

### Error Handling Architecture
```python
# exceptions.py - centralized exception definitions
class ProjectError(Exception):
    """Base exception for project."""

class ValidationError(ProjectError):
    """Input validation failures."""
```

### Resource Management
```python
# Use context managers
with open_database() as db:
    result = db.query(sql)

# Async cleanup pattern
try:
    result = await async_operation()
finally:
    await cleanup_resources()
```

### Code Documentation
```python
# AIDEV-NOTE: Critical performance path - benchmark before changing
# AIDEV-TODO: Refactor after v2.0 release
# AIDEV-QUESTION: Consider caching strategy here?
```

Documentation rules:
- Search for existing AIDEV-* comments first
- Update anchors when modifying related code
- Keep comments under 120 characters
- Place anchors above relevant code blocks

### Change Management
- Request approval for changes exceeding 300 lines
- Request approval for modifications spanning 4+ files
- Break large refactors into incremental commits

## Testing Guidelines

### Testing Boundaries
- Tests define behavioral contracts (human domain)
- Read test files to understand expected behavior
- Use test failures to guide implementation
- Report test gaps discovered during implementation

### Test-Driven Development Support
- Implement code to pass existing tests
- Suggest test scenarios for new functionality
- Maintain backward compatibility with test suite

## Security Considerations

### Secure Coding Practices
- Store credentials in environment variables
- Validate all external inputs
- Use parameterized queries for databases
- Apply principle of least privilege

### Configuration Security
```python
# Correct approach
api_key = os.environ.get("API_KEY")

# Instead of hardcoding
api_key = "sk-abc123"  # Never do this
```

## Workflow Integration

### Git Commit Standards
Format: `type: description [AI]`

Types:
- feat: New functionality
- fix: Bug repairs
- refactor: Code restructuring
- perf: Performance improvements
- docs: Documentation updates

Examples:
```
feat: implement user authentication [AI]
fix: correct memory leak in parser [AI]
refactor: simplify database connection logic [AI]
```

### Commit Practices
- Create one commit per logical change
- Reference issues using "closes #123"
- Work on feature branches
- Mark AI contributions with [AI] tag

### Task Execution Flow
1. Parse requirements and locate relevant AGENTS.md files
2. Request clarification for ambiguous requirements
3. Present implementation approach for complex tasks
4. Execute implementation following conventions
5. Update AIDEV-* markers in modified code
6. Report completion status to user

### Session Management
Start fresh session when:
- Task context differs significantly from current work
- Memory/context becomes unclear
- User requests unrelated functionality
