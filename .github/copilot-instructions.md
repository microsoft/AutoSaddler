# GitHub Copilot instructions (repo-wide)

## Environment & command execution (IMPORTANT)
- This repo uses a Python environment managed by `uv`.
- **Before running any Python-related command** (anything that invokes `python`, `pip`, `pytest`, `ruff`, `mypy`, `alembic`, `django-admin`, `flask`, `uvicorn`, `gunicorn`, `mkdocs`, etc.), always run it **inside the uv environment**.
- Prefer these patterns (pick the most direct one):
  - `uv run <command ...>` (preferred for one-off commands)
  - If an interactive shell is needed: `uv venv && source .venv/bin/activate` (or platform equivalent), then run commands.
- Do **not** assume the env is already active.
- Only skip `uv` wrapping for common non-Python shell commands like: `ls`, `cd`, `cat`, `echo`, `pwd`, `git ...`, `mkdir`, `rm`, `cp`, `mv`, `sed`, `awk`, `curl`, `wget` (unless they are part of a Python workflow step).

## Examples
- ✅ `uv run python -m pytest -q`
- ✅ `uv run ruff check .`
- ✅ `uv run python -m mypy src/`
- ✅ `uv run python scripts/sync_data.py --dry-run`
- ✅ `git status` (no `uv` needed)
- ✅ `ls -la` (no `uv` needed)
- ❌ `pytest` (should be `uv run pytest`)
- ❌ `python script.py` (should be `uv run python script.py`)

## Dependency management
- Add/remove deps using `uv add <pkg>` / `uv remove <pkg>` (or whatever standard this repo uses).
- If a lockfile is present, keep it updated consistently with `uv` workflows.

## Documentation & consistency (IMPORTANT)
- **After any code changes**, always check and update relevant documentation (`docs/`, `README.md`), output-related files (docstring, config references), and log/result schemas to reflect the latest changes.
- If inconsistencies are found between code and documentation, always correct them. If unsure how to resolve an inconsistency, ask the user before proceeding.
- Keep CLI help text, config file comments, and doc examples in sync with actual implementations.

## Testing (IMPORTANT)
- **Always run the test suite after significant code changes** (`uv run python -m pytest tests/ -v --tb=short`).
- Do not consider a task complete until tests pass.

## Error handling & debugging (IMPORTANT)
- **Minimize fallback logic.** Do not silently swallow errors or provide default values that hide incorrect state. Fallback paths easily mask bugs and make issues harder to diagnose. This applies to **all code** — not just error-handling paths but also normal implementation logic.
- **Prefer assertions and explicit checks.** Use `assert` statements or raise clear exceptions (`ValueError`, `TypeError`, etc.) to enforce invariants and fail fast when assumptions are violated.
- When a value *must* be present or a condition *must* hold, assert it rather than falling back to a default.
- ✅ `assert config is not None, "config must be loaded before use"`
- ✅ `if result is None: raise ValueError("Expected a result but got None")`
- ❌ `value = result or some_default  # silently hides a missing result`
- ❌ `try: ... except Exception: pass  # swallows all errors`

## Git & version control (IMPORTANT)
- **NEVER commit code changes without the user approval.**
