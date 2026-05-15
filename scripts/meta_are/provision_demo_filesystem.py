#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Sequence

from autosaddler.v2.plugins.meta_are.provisioning import (
    DEFAULT_FILESYSTEM_REPO_ID,
    ProvisioningRequest,
    provision_demo_filesystem,
    run_meta_are_filesystem_probe,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Provision an immutable local Meta-ARE demo filesystem with verified provenance.",
    )
    parser.add_argument(
        "--destination-root",
        type=Path,
        required=True,
        help="Absolute user-owned parent for commit-addressed filesystem revisions.",
    )
    parser.add_argument(
        "--revision",
        required=True,
        help="Full 40-character Hugging Face dataset Git commit; symbolic revisions are rejected.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_FILESYSTEM_REPO_ID,
        help="Hugging Face dataset repository ID.",
    )
    parser.add_argument(
        "--meta-are-project",
        type=Path,
        required=True,
        help="Pinned external Meta-ARE checkout used for the read-only SandboxLocalFileSystem probe.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    request = ProvisioningRequest(
        destination_root=args.destination_root,
        source_revision=args.revision,
        repo_id=args.repo_id,
    )
    result = provision_demo_filesystem(
        request,
        probe=partial(run_meta_are_filesystem_probe, args.meta_are_project),
    )
    print(
        json.dumps(
            {
                "revision_root": str(result.revision_root),
                "demo_filesystem_root": str(result.filesystem_root),
                "source_descriptor": str(result.source_descriptor),
                "content_manifest": str(result.content_manifest),
                "content_digest": result.content_digest,
                "file_count": result.file_count,
                "total_bytes": result.total_bytes,
                "reused": result.reused,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())