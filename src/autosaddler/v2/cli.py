from __future__ import annotations

import argparse
from pathlib import Path

from autosaddler.v2.config.registry import build_runtime
from autosaddler.v2.storage.local import LocalRunStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AutoSaddler v0.2 engine")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--fork-from-run-id")
    parser.add_argument("--fork-through-sequence", type=int)
    arguments = parser.parse_args()
    if (arguments.fork_from_run_id is None) != (arguments.fork_through_sequence is None):
        parser.error("--fork-from-run-id and --fork-through-sequence must be provided together")
    runtime = build_runtime(arguments.config, run_id=arguments.run_id)
    if arguments.fork_from_run_id is not None:
        source = LocalRunStore(
            run_dir=runtime.config.storage.run_root / arguments.fork_from_run_id,
            run_id=arguments.fork_from_run_id,
        )
        runtime.store.fork_from(source, through_sequence=arguments.fork_through_sequence)
    result = runtime.engine.run()
    print(f"selected_candidate_id={result.selected_candidate_id}")
    print(f"development_score={result.development_score:.6f}")


if __name__ == "__main__":
    main()