from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from autosaddler.v2.core.domain import Case, canonical_json, sha256_digest
from autosaddler.v2.plugins.meta_are.responses_runtime import RESPONSES_RUNTIME_PATH


class ReleasableMaterialization(Protocol):
    root: Path
    candidate_id: str

    def release(self) -> None: ...


@dataclass(frozen=True, slots=True)
class MetaARERunResult:
    raw_results: Path
    hf_trace_dir: Path
    lite_trace_dir: Path


class MetaARERunner:
    def __init__(
        self,
        *,
        demo_filesystem_root: Path,
        agent: str,
        model: str,
        model_provider: str,
        judge_model: str,
        judge_provider: str,
        benchmark_config: str,
        benchmark_split: str,
        timeout_seconds: float,
        process_completion_grace_seconds: float,
        max_concurrent: int,
        dataset_root: Path | None = None,
        model_endpoint: str | None = None,
        reasoning_effort: str | None = None,
        judge_endpoint: str | None = None,
        model_wire_api: str = "chat_completions",
        responses_runtime_sha256: str | None = None,
    ) -> None:
        self.demo_filesystem_root = demo_filesystem_root.resolve()
        self.agent = agent
        self.model = model
        self.model_provider = model_provider
        self.judge_model = judge_model
        self.judge_provider = judge_provider
        self.benchmark_config = benchmark_config
        self.benchmark_split = benchmark_split
        self.timeout_seconds = timeout_seconds
        if process_completion_grace_seconds < 0:
            raise ValueError("Meta-ARE process completion grace cannot be negative")
        self.process_completion_grace_seconds = process_completion_grace_seconds
        self.max_concurrent = max_concurrent
        self.dataset_root = dataset_root.resolve() if dataset_root is not None else None
        self.model_endpoint = model_endpoint
        self.reasoning_effort = reasoning_effort
        self.judge_endpoint = judge_endpoint
        if model_wire_api not in {"chat_completions", "responses"}:
            raise ValueError("Unsupported Meta-ARE model wire API")
        if model_wire_api == "responses" and model_provider != "copilot":
            raise ValueError("Meta-ARE Responses transport requires the copilot provider")
        if model_wire_api == "responses" and responses_runtime_sha256 is None:
            raise ValueError("Meta-ARE Responses transport requires a pinned runtime digest")
        self.model_wire_api = model_wire_api
        self.responses_runtime_sha256 = responses_runtime_sha256

    def run(
        self,
        *,
        materialized: ReleasableMaterialization,
        cases: tuple[Case, ...],
        repetitions: int,
        attempt_dir: Path,
    ) -> MetaARERunResult:
        if not cases:
            raise ValueError("Meta-ARE runner requires at least one requested case")
        if repetitions < 1:
            raise ValueError("Meta-ARE repetitions must be positive")
        attempt_dir = attempt_dir.resolve()
        try:
            dataset_dir = attempt_dir / "dataset"
            output_dir = attempt_dir / "output"
            stdout_path = attempt_dir / "stdout.log"
            stderr_path = attempt_dir / "stderr.log"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            staged = self._stage_cases(cases, dataset_dir)
            for source, destination, expected_digest in staged:
                if sha256_digest(source.read_bytes()) != expected_digest:
                    raise ValueError(f"Meta-ARE source digest drift before subprocess: {source}")
                if sha256_digest(destination.read_bytes()) != expected_digest:
                    raise ValueError(f"Meta-ARE staged scenario digest drift before subprocess: {destination}")

            runtime_entrypoint = self._stage_responses_runtime(attempt_dir)
            command = self._command(
                materialized.root,
                dataset_dir,
                output_dir,
                repetitions,
                runtime_entrypoint,
            )
            environment = self._environment(materialized.root, attempt_dir)
            (attempt_dir / "command.json").write_text(
                canonical_json(
                    {
                        "argv": list(command),
                        "cwd": str(materialized.root),
                        "environment": {
                            key: environment[key]
                            for key in (
                                "PYTHONPATH",
                                "DEMO_FS_PATH",
                                "FS_PATH",
                                "HF_HOME",
                                "HF_DATASETS_CACHE",
                                "TRANSFORMERS_CACHE",
                                "HF_HUB_OFFLINE",
                                "HF_DATASETS_OFFLINE",
                                "TRANSFORMERS_OFFLINE",
                            )
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            completion_path = attempt_dir / "completion.json"
            raw_results = output_dir / "output.jsonl"
            if completion_path.is_file():
                completion = json.loads(completion_path.read_text(encoding="utf-8"))
                if not isinstance(completion, dict) or completion.get("command_sha256") != sha256_digest(
                    canonical_json(list(command))
                ):
                    raise ValueError("Meta-ARE durable runner command drifted on resume")
                returncode = completion.get("returncode")
                if isinstance(returncode, bool) or not isinstance(returncode, int):
                    raise TypeError("Meta-ARE durable runner returncode must be an integer")
                if returncode != 0:
                    detail = _failure_detail(stderr_path, stdout_path)
                    raise RuntimeError(
                        f"Meta-ARE benchmark exited with status {returncode}: {detail}"
                    )
                if not raw_results.is_file():
                    raise RuntimeError(
                        "Meta-ARE benchmark completed successfully but produced no output.jsonl"
                    )
                return self._result(
                    output_dir=output_dir,
                )

            completed = subprocess.run(
                command,
                cwd=materialized.root,
                env=environment,
                text=True,
                capture_output=True,
                timeout=self._process_timeout(len(cases), repetitions),
                shell=False,
                check=False,
            )
            stdout_path.write_text(completed.stdout, encoding="utf-8")
            stderr_path.write_text(completed.stderr, encoding="utf-8")
            completion_path.write_text(
                canonical_json(
                    {
                        "returncode": completed.returncode,
                        "command_sha256": sha256_digest(canonical_json(list(command))),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            if completed.returncode != 0:
                detail = completed.stderr[-2_000:] or completed.stdout[-2_000:]
                raise RuntimeError(
                    f"Meta-ARE benchmark exited with status {completed.returncode}: "
                    f"{detail.strip() or 'no process output'}"
                )
            if not raw_results.is_file():
                raise RuntimeError(
                    f"Meta-ARE benchmark produced no output.jsonl (exit={completed.returncode})"
                )
            return self._result(
                output_dir=output_dir,
            )
        finally:
            materialized.release()

    def _process_timeout(self, case_count: int, repetitions: int) -> float:
        work_items = case_count * repetitions
        concurrency_waves = (work_items + self.max_concurrent - 1) // self.max_concurrent
        return (
            concurrency_waves * self.timeout_seconds
            + self.process_completion_grace_seconds
        )

    def _stage_cases(
        self,
        cases: tuple[Case, ...],
        dataset_dir: Path,
    ) -> tuple[tuple[Path, Path, str], ...]:
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        staged = []
        for case in cases:
            raw_source = case.payload.get("source_path")
            raw_digest = case.payload.get("source_sha256")
            if not isinstance(raw_source, str) or not isinstance(raw_digest, str):
                raise TypeError(f"Meta-ARE case {case.case_id} lacks frozen source metadata")
            source = Path(raw_source)
            if not source.is_absolute():
                if self.dataset_root is None:
                    raise ValueError("Relative Meta-ARE case paths require dataset_root")
                source = self.dataset_root / source
            source = source.resolve(strict=True)
            expected_digest = raw_digest
            enforce_digest = len(expected_digest) == 71 and expected_digest.startswith("sha256:")
            if enforce_digest and sha256_digest(source.read_bytes()) != expected_digest:
                raise ValueError(f"Meta-ARE source digest drift: {source}")
            if not enforce_digest:
                expected_digest = sha256_digest(source.read_bytes())
            relative = Path(self.benchmark_config) / self.benchmark_split / source.name
            destination = dataset_dir / relative
            if destination.exists():
                raise ValueError(f"Meta-ARE staged scenario filename collision: {source.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            staged.append((source, destination, expected_digest))
        return tuple(staged)

    def _command(
        self,
        project_root: Path,
        dataset_dir: Path,
        output_dir: Path,
        repetitions: int,
        runtime_entrypoint: Path | None,
    ) -> tuple[str, ...]:
        values = [
            "uv",
            "run",
            "--project",
            str(project_root),
            "--with",
            "azure-identity-broker==1.3.0",
        ]
        if self.model_wire_api == "responses":
            assert runtime_entrypoint is not None
            values.extend(
                (
                    "python",
                    str(runtime_entrypoint),
                    "--model-wire-api",
                    "responses",
                    "--",
                )
            )
        else:
            assert runtime_entrypoint is None
            values.append("are-benchmark")
        values.extend(
            (
            "run",
            "--agent",
            self.agent,
            "--provider",
            self.model_provider,
            "--model",
            self.model,
            )
        )
        if self.model_endpoint is not None:
            values.extend(("--endpoint", self.model_endpoint))
        if self.reasoning_effort is not None:
            values.extend(("--reasoning_effort", self.reasoning_effort))
        values.extend(
            (
                "--dataset",
                str(dataset_dir),
                "--config",
                self.benchmark_config,
                "--split",
                self.benchmark_split,
                "--output_dir",
                str(output_dir),
                "--scenario_timeout",
                str(self.timeout_seconds),
                "--num_runs",
                str(repetitions),
                "--max_concurrent_scenarios",
                str(self.max_concurrent),
                "--trace_dump_format",
                "both",
                "--judge_provider",
                self.judge_provider,
                "--judge_model",
                self.judge_model,
            )
        )
        if self.judge_endpoint is not None:
            values.extend(("--judge_endpoint", self.judge_endpoint))
        return tuple(values)

    def _stage_responses_runtime(self, attempt_dir: Path) -> Path | None:
        if self.model_wire_api != "responses":
            return None
        assert self.responses_runtime_sha256 is not None
        source = RESPONSES_RUNTIME_PATH
        if sha256_digest(source.read_bytes()) != self.responses_runtime_sha256:
            raise ValueError("Meta-ARE Responses runtime source digest drift")
        destination = attempt_dir / "runtime" / source.name
        if destination.is_file():
            if sha256_digest(destination.read_bytes()) != self.responses_runtime_sha256:
                raise ValueError("Meta-ARE staged Responses runtime digest drift")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        return destination

    def _environment(self, candidate_root: Path, attempt_dir: Path) -> dict[str, str]:
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(candidate_root)
            if not existing_pythonpath
            else os.pathsep.join((str(candidate_root), existing_pythonpath))
        )
        environment.update(
            {
                "DEMO_FS_PATH": str(self.demo_filesystem_root),
                "FS_PATH": str(attempt_dir / "filesystem"),
                "HF_HOME": str(attempt_dir / "cache/hf"),
                "HF_DATASETS_CACHE": str(attempt_dir / "cache/datasets"),
                "TRANSFORMERS_CACHE": str(attempt_dir / "cache/transformers"),
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        for name in ("FS_PATH", "HF_HOME", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE"):
            Path(environment[name]).mkdir(parents=True, exist_ok=True)
        return environment

    @staticmethod
    def _result(
        *,
        output_dir: Path,
    ) -> MetaARERunResult:
        return MetaARERunResult(
            raw_results=output_dir / "output.jsonl",
            hf_trace_dir=output_dir / "hf",
            lite_trace_dir=output_dir / "lite",
        )


def _failure_detail(stderr_path: Path, stdout_path: Path) -> str:
    for path in (stderr_path, stdout_path):
        if path.is_file():
            detail = path.read_text(encoding="utf-8")[-2_000:].strip()
            if detail:
                return detail
    return "no process output"
