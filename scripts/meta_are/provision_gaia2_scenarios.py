#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from autosaddler.v2.plugins.meta_are.scenario_provisioning import (
    DEFAULT_SCENARIO_REPO_ID,
    ScenarioProvisioningRequest,
    provision_gaia2_scenarios,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MANIFESTS = (
    _PROJECT_ROOT / "configs/datasets/GAIA2/train_smoke.json",
    _PROJECT_ROOT / "configs/datasets/GAIA2/val_smoke.json",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Provision the manifest-selected GAIA2 smoke scenarios from a pinned dataset revision.",
    )
    parser.add_argument(
        "--destination-root",
        type=Path,
        required=True,
        help="Absolute destination matching scenario.settings.dataset_root.",
    )
    parser.add_argument(
        "--revision",
        required=True,
        help="Full 40-character Hugging Face GAIA2 dataset Git commit.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_SCENARIO_REPO_ID,
        help="Hugging Face dataset repository ID.",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        dest="manifests",
        help="Split manifest to provision; repeat for multiple manifests (defaults to V2 smoke train and development).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    request = ScenarioProvisioningRequest(
        destination_root=args.destination_root,
        source_revision=args.revision,
        manifests=tuple(args.manifests or _DEFAULT_MANIFESTS),
        repo_id=args.repo_id,
    )
    result = provision_gaia2_scenarios(request)
    print(
        json.dumps(
            {
                "destination_root": str(result.destination_root),
                "source_descriptor": str(result.source_descriptor),
                "source_revision": result.source_revision,
                "file_count": result.file_count,
                "reused_count": result.reused_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())