from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from autosaddler.v2.core.domain import Case


class MaterializedCandidate:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.candidate_id = "sha256:" + "c" * 64
        self.released = False

    def release(self) -> None:
        self.released = True


def test_runner_uses_pinned_uv_argv_local_environment_and_releases_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from autosaddler.v2.plugins.meta_are.runner import MetaARERunner
    from autosaddler.v2.plugins.meta_are.responses_runtime import (
        RESPONSES_RUNTIME_PATH,
    )
    from autosaddler.v2.core.domain import sha256_digest

    source = tmp_path / "meta-are"
    source.mkdir()
    demo = tmp_path / "demo"
    demo.mkdir()
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    materialized = MaterializedCandidate(candidate)
    case_file = tmp_path / "case.json"
    case_file.write_text('{"id":"case-a"}\n')
    case = Case("case-a", "train", {"source_path": str(case_file), "source_sha256": "sha256:fixture"})

    def fake_run(argv, **kwargs):
        output_dir = Path(argv[argv.index("--output_dir") + 1])
        output_dir.mkdir(parents=True)
        (output_dir / "output.jsonl").write_text(
            json.dumps({"metadata": {"scenario_id": "case-a", "run_number": 1}, "status": "failed", "score": 0.0}) + "\n"
        )
        assert isinstance(argv, tuple)
        assert argv[:4] == ("uv", "run", "--project", str(candidate))
        runtime_path = Path(argv[argv.index("python") + 1])
        assert runtime_path == tmp_path / "attempt/runtime/responses_runtime.py"
        assert sha256_digest(runtime_path.read_bytes()) == sha256_digest(
            RESPONSES_RUNTIME_PATH.read_bytes()
        )
        assert argv[argv.index("--model-wire-api") + 1] == "responses"
        assert argv[argv.index("--scenario_timeout") + 1] == "30"
        assert kwargs["cwd"] == candidate
        assert kwargs["shell"] is False
        assert kwargs["env"]["PYTHONPATH"].split(":")[0] == str(candidate)
        assert kwargs["env"]["DEMO_FS_PATH"] == str(demo)
        assert Path(kwargs["env"]["FS_PATH"]).is_relative_to(tmp_path / "attempt")
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = MetaARERunner(
        demo_filesystem_root=demo,
        agent="default",
        model="task-model",
        model_provider="copilot",
        model_wire_api="responses",
        responses_runtime_sha256=sha256_digest(RESPONSES_RUNTIME_PATH.read_bytes()),
        judge_model="judge-model",
        judge_provider="azure",
        benchmark_config="search",
        benchmark_split="validation",
        timeout_seconds=30,
        process_completion_grace_seconds=60,
        max_concurrent=1,
    )
    result = runner.run(materialized=materialized, cases=(case,), repetitions=1, attempt_dir=tmp_path / "attempt")

    assert result.raw_results.is_file()
    assert materialized.released
    command = json.loads((tmp_path / "attempt/command.json").read_text())
    assert "OPENAI_API_KEY" not in json.dumps(command)
    assert (tmp_path / "attempt/stdout.log").read_text() == "ok"


def test_runner_rejects_source_drift_before_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from autosaddler.v2.plugins.meta_are.runner import MetaARERunner

    source = tmp_path / "meta-are"
    source.mkdir()
    demo = tmp_path / "demo"
    demo.mkdir()
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    case_file = tmp_path / "case.json"
    case_file.write_text('{"id":"case-a","drifted":true}\n')
    case = Case("case-a", "train", {"source_path": str(case_file), "source_sha256": "sha256:" + "0" * 64})
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: pytest.fail("subprocess must not run"))
    runner = MetaARERunner(
        demo_filesystem_root=demo,
        agent="default",
        model="task-model",
        model_provider="openai",
        judge_model="judge-model",
        judge_provider="azure",
        benchmark_config="search",
        benchmark_split="validation",
        timeout_seconds=30,
        process_completion_grace_seconds=60,
        max_concurrent=1,
    )

    with pytest.raises(ValueError, match="digest|drift"):
        runner.run(
            materialized=MaterializedCandidate(candidate),
            cases=(case,),
            repetitions=1,
            attempt_dir=tmp_path / "attempt",
        )


def test_runner_rejects_nonzero_exit_even_with_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autosaddler.v2.plugins.meta_are.runner import MetaARERunner

    source = tmp_path / "meta-are"
    source.mkdir()
    demo = tmp_path / "demo"
    demo.mkdir()
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    case_file = tmp_path / "case.json"
    case_file.write_text('{"id":"case-a"}\n', encoding="utf-8")
    case = Case(
        "case-a",
        "train",
        {"source_path": str(case_file), "source_sha256": "sha256:fixture"},
    )

    def fake_run(argv, **_kwargs):
        output_dir = Path(argv[argv.index("--output_dir") + 1])
        output_dir.mkdir(parents=True)
        (output_dir / "output.jsonl").write_text(
            json.dumps(
                {
                    "metadata": {"scenario_id": "case-a", "run_number": 1},
                    "status": "failed",
                    "score": 0.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 1, stdout="partial", stderr="crashed")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = MetaARERunner(
        demo_filesystem_root=demo,
        agent="default",
        model="task-model",
        model_provider="openai",
        judge_model="judge-model",
        judge_provider="azure",
        benchmark_config="search",
        benchmark_split="validation",
        timeout_seconds=30,
        process_completion_grace_seconds=60,
        max_concurrent=1,
    )

    with pytest.raises(RuntimeError, match="exit|crashed|nonzero"):
        runner.run(
            materialized=MaterializedCandidate(candidate),
            cases=(case,),
            repetitions=1,
            attempt_dir=tmp_path / "attempt",
        )

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("failed durable attempt must not relaunch"),
    )
    with pytest.raises(RuntimeError, match="exit|crashed|nonzero"):
        runner.run(
            materialized=MaterializedCandidate(candidate),
            cases=(case,),
            repetitions=1,
            attempt_dir=tmp_path / "attempt",
        )


@pytest.mark.parametrize("returncode", [0, 1])
def test_runner_does_not_relaunch_outputless_durable_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
) -> None:
    from autosaddler.v2.plugins.meta_are.runner import MetaARERunner

    demo = tmp_path / "demo"
    demo.mkdir()
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    case_file = tmp_path / "case.json"
    case_file.write_text('{"id":"case-a"}\n', encoding="utf-8")
    case = Case(
        "case-a",
        "train",
        {"source_path": str(case_file), "source_sha256": "sha256:fixture"},
    )
    calls = 0

    def fake_run(argv, **_kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(
            argv,
            returncode,
            stdout="partial",
            stderr="crashed" if returncode else "",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = MetaARERunner(
        demo_filesystem_root=demo,
        agent="default",
        model="task-model",
        model_provider="openai",
        judge_model="judge-model",
        judge_provider="azure",
        benchmark_config="search",
        benchmark_split="validation",
        timeout_seconds=30,
        process_completion_grace_seconds=60,
        max_concurrent=1,
    )

    for _ in range(2):
        with pytest.raises(RuntimeError, match="output.jsonl|exited"):
            runner.run(
                materialized=MaterializedCandidate(candidate),
                cases=(case,),
                repetitions=1,
                attempt_dir=tmp_path / "attempt",
            )

    assert calls == 1


def test_runner_process_timeout_covers_all_concurrency_waves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autosaddler.v2.plugins.meta_are.runner import MetaARERunner

    demo = tmp_path / "demo"
    demo.mkdir()
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    cases = []
    for case_id in ("case-a", "case-b", "case-c"):
        case_file = tmp_path / f"{case_id}.json"
        case_file.write_text(json.dumps({"id": case_id}) + "\n", encoding="utf-8")
        cases.append(
            Case(
                case_id,
                "train",
                {"source_path": str(case_file), "source_sha256": "sha256:fixture"},
            )
        )

    def fake_run(argv, **kwargs):
        assert kwargs["timeout"] == 150
        output_dir = Path(argv[argv.index("--output_dir") + 1])
        output_dir.mkdir(parents=True)
        (output_dir / "output.jsonl").write_text("{}\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = MetaARERunner(
        demo_filesystem_root=demo,
        agent="default",
        model="task-model",
        model_provider="openai",
        judge_model="judge-model",
        judge_provider="azure",
        benchmark_config="search",
        benchmark_split="validation",
        timeout_seconds=30,
        process_completion_grace_seconds=60,
        max_concurrent=2,
    )

    runner.run(
        materialized=MaterializedCandidate(candidate),
        cases=tuple(cases),
        repetitions=2,
        attempt_dir=tmp_path / "attempt",
    )


def test_runner_stages_mixed_capabilities_in_one_requested_bucket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autosaddler.v2.core.domain import sha256_digest
    from autosaddler.v2.plugins.meta_are.runner import MetaARERunner

    source = tmp_path / "meta-are"
    source.mkdir()
    demo = tmp_path / "demo"
    demo.mkdir()
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    dataset = tmp_path / "dataset"
    cases = []
    for capability, case_id in (("adaptability", "case-a"), ("search", "case-b")):
        case_file = dataset / capability / "validation" / f"{case_id}.json"
        case_file.parent.mkdir(parents=True, exist_ok=True)
        case_file.write_text(json.dumps({"id": case_id}) + "\n", encoding="utf-8")
        cases.append(
            Case(
                case_id,
                "train",
                {
                    "source_path": case_file.relative_to(dataset).as_posix(),
                    "source_sha256": sha256_digest(case_file.read_bytes()),
                },
            )
        )

    def fake_run(argv, **_kwargs):
        staged = Path(argv[argv.index("--dataset") + 1]) / "autosaddler_selected/validation"
        assert sorted(path.name for path in staged.iterdir()) == ["case-a.json", "case-b.json"]
        output_dir = Path(argv[argv.index("--output_dir") + 1])
        output_dir.mkdir(parents=True)
        (output_dir / "output.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "metadata": {"scenario_id": case.case_id, "run_number": 1},
                        "status": "failed",
                        "score": 0.0,
                    }
                )
                for case in cases
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = MetaARERunner(
        demo_filesystem_root=demo,
        dataset_root=dataset,
        agent="default",
        model="task-model",
        model_provider="openai",
        judge_model="judge-model",
        judge_provider="azure",
        benchmark_config="autosaddler_selected",
        benchmark_split="validation",
        timeout_seconds=30,
        process_completion_grace_seconds=60,
        max_concurrent=1,
    )

    result = runner.run(
        materialized=MaterializedCandidate(candidate),
        cases=tuple(cases),
        repetitions=1,
        attempt_dir=tmp_path / "attempt",
    )

    assert result.raw_results.is_file()