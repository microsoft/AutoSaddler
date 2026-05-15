from __future__ import annotations

import json
import os
import re
import subprocess

from autosaddler.v2.harness.git import (
    GitVerificationContext,
    GitVerificationVerdict,
)


class MetaAREVerifier:
    def __init__(
        self,
        *,
        import_check: str,
        verification_timeout_seconds: float,
        train_case_ids: tuple[str, ...],
    ) -> None:
        if verification_timeout_seconds <= 0:
            raise ValueError("Meta-ARE verification timeout must be positive")
        self.import_check = import_check
        self.verification_timeout_seconds = verification_timeout_seconds
        self.train_case_ids = train_case_ids

    def __call__(self, context: GitVerificationContext) -> GitVerificationVerdict:
        python_paths = tuple(path for path in context.changed_paths if path.suffix == ".py")
        for relative in python_paths:
            path = context.workspace.joinpath(*relative.parts)
            try:
                compile(path.read_bytes(), relative.as_posix(), "exec")
            except (OSError, SyntaxError, UnicodeError) as error:
                return _rejected("python_compile", f"{relative}: {error}")

        for relative in context.changed_paths:
            if relative.name != "hook.json":
                continue
            try:
                value = json.loads(context.workspace.joinpath(*relative.parts).read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                return _rejected("hook_config", f"{relative}: {error}")
            hook_error = _hook_config_error(value)
            if hook_error is not None:
                return _rejected("hook_config", f"{relative}: {hook_error}")

        if context.patch_label == "steering" and python_paths:
            return _rejected(
                "steering_scope",
                "Steering-phase mutations may change hook configuration only, not Python control flow",
            )

        for relative in context.changed_paths:
            path = context.workspace.joinpath(*relative.parts)
            if not path.is_file() or path.suffix not in {".py", ".json", ".md", ".toml", ".yaml", ".yml"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            matched = next((case_id for case_id in self.train_case_ids if case_id in text), None)
            if matched is not None:
                return _rejected(
                    "anti_cheating",
                    f"Candidate content contains a configured training case identifier: {matched}",
                )

        if python_paths and self.import_check:
            environment = os.environ.copy()
            existing_pythonpath = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = (
                str(context.workspace)
                if not existing_pythonpath
                else os.pathsep.join((str(context.workspace), existing_pythonpath))
            )
            try:
                completed = subprocess.run(
                    [
                        "uv",
                        "run",
                        "--project",
                        str(context.workspace),
                        "python",
                        "-c",
                        self.import_check,
                    ],
                    cwd=context.workspace,
                    env=environment,
                    text=True,
                    capture_output=True,
                    timeout=self.verification_timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return _rejected(
                    "import_check_timeout",
                    f"Candidate import check timed out after {self.verification_timeout_seconds} seconds",
                )
            if completed.returncode != 0:
                detail = completed.stderr[-2000:] or completed.stdout[-2000:]
                return _rejected("import_check", detail.strip() or "candidate import check failed")

        return GitVerificationVerdict(
            accepted=True,
            check="all",
            summary="All Meta-ARE mutation checks passed",
            artifacts=(),
        )


def _rejected(check: str, summary: str) -> GitVerificationVerdict:
    return GitVerificationVerdict(
        accepted=False,
        check=check,
        summary=summary,
        artifacts=(),
    )


def _hook_config_error(value: object) -> str | None:
    if not isinstance(value, dict):
        return "hook configuration must be an object"
    hooks = value.get("hooks")
    if not isinstance(hooks, dict):
        return "hooks must be an object"
    groups = hooks.get("PreToolUse")
    if not isinstance(groups, list):
        return "hooks.PreToolUse must be a list"
    for group_index, group in enumerate(groups):
        if not isinstance(group, dict):
            return f"PreToolUse group {group_index} must be an object"
        matcher = group.get("matcher")
        if not isinstance(matcher, str) or not matcher:
            return f"PreToolUse group {group_index} matcher must be a non-empty string"
        try:
            re.compile(matcher)
        except re.error as error:
            return f"PreToolUse group {group_index} matcher is invalid: {error}"
        description = group.get("description")
        if description is not None and not isinstance(description, str):
            return f"PreToolUse group {group_index} description must be a string"
        handlers = group.get("hooks")
        if not isinstance(handlers, list) or not handlers:
            return f"PreToolUse group {group_index} hooks must be a non-empty list"
        for handler_index, handler in enumerate(handlers):
            label = f"PreToolUse group {group_index} handler {handler_index}"
            if not isinstance(handler, dict):
                return f"{label} must be an object"
            handler_type = handler.get("type")
            if handler_type not in {"prompt", "agent", "reminder"}:
                return f"{label} type must be prompt, agent, or reminder"
            text_key = "reminder" if handler_type == "reminder" else "prompt"
            text = handler.get(text_key)
            if not isinstance(text, str) or not text.strip():
                return f"{label} {text_key} must be a non-empty string"
            timeout = handler.get("timeout")
            if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0):
                return f"{label} timeout must be a positive integer"
    return None
