"""EvolutionDAG: DAG data structure with JSON persistence.

Manages the evolution history of patch candidates as a directed acyclic graph.
Persisted to ``evolution_dag.json`` in the session root.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autosaddler.v1.proposer.autosaddler.models import (
    AccumulatedLessons,
    EvolutionEdge,
    EvolutionNode,
    PatchIntent,
    PatchVerdict,
    ReflectionEntry,
    ScenarioEntry,
    ScenarioImpact,
    SDKSessionInfo,
    SelectionDecision,
)

logger = logging.getLogger(__name__)


class EvolutionDAG:
    """Directed acyclic graph tracking the evolution of patch candidates.

    Nodes represent candidates; edges represent parent→child relationships
    (``base`` for the immediate predecessor, ``cherry_pick`` for candidates
    chosen in Session 0).  Persisted as ``evolution_dag.json``.
    """

    def __init__(self, session_root: str) -> None:
        self.session_root = session_root
        self._json_path = Path(session_root) / "evolution_dag.json"

        self.metadata: dict[str, Any] = {
            "session_root": session_root,
            "total_iterations": 0,
            "last_updated": "",
        }
        self.good_patch_definition: dict[str, str] = {
            "effectiveness": "Fix failing scenarios by addressing their root cause to flip FAIL to PASS",
            "safety": "Do not break scenarios that currently PASS (no PASS to FAIL regressions)",
        }

        self.nodes: dict[int, EvolutionNode] = {}
        self.edges: dict[str, EvolutionEdge] = {}  # key: "parent_idx->child_idx"
        self.scenario_registry: dict[str, ScenarioEntry] = {}
        self.accumulated_lessons: AccumulatedLessons = AccumulatedLessons()
        self._pending_reflections: dict[int, list[ReflectionEntry]] = {}

        self._next_idx = 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @property
    def _gz_path(self) -> Path:
        return self._json_path.with_suffix(".json.gz")

    @property
    def _diffs_dir(self) -> Path:
        return Path(self.session_root) / "diffs"

    def _persist_edge_diff(self, edge: EvolutionEdge, code_diff: str) -> None:
        """Persist edge diff externally and keep only a reference in the DAG."""
        if not code_diff:
            edge.code_diff = None
            edge.code_diff_path = None
            edge.code_diff_sha256 = None
            edge.code_diff_size_bytes = 0
            return

        raw = code_diff.encode("utf-8", errors="replace")
        digest = hashlib.sha256(raw).hexdigest()
        rel_path = f"diffs/{edge.edge_type}_{edge.parent_idx}_{edge.child_idx}_{digest[:12]}.diff.gz"
        abs_path = Path(self.session_root) / rel_path

        if not abs_path.exists():
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=str(abs_path.parent), suffix=".diff.gz.tmp")
            try:
                os.close(fd)
                with gzip.open(tmp_path, "wb", compresslevel=3) as gz:
                    gz.write(raw)
                os.replace(tmp_path, str(abs_path))
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

        edge.code_diff = None
        edge.code_diff_path = rel_path
        edge.code_diff_sha256 = digest
        edge.code_diff_size_bytes = len(raw)

    def _migrate_inline_diffs_to_external(self) -> int:
        """Move legacy inline code diffs to external files.

        Returns the number of migrated edges.
        """
        migrated = 0
        for edge in self.edges.values():
            if edge.code_diff and not edge.code_diff_path:
                self._persist_edge_diff(edge, edge.code_diff)
                migrated += 1
        return migrated

    def get_edge_diff_text(self, edge: EvolutionEdge) -> str | None:
        """Load edge diff text lazily, preserving CLI behavior without memory spikes."""
        if edge.code_diff:
            return edge.code_diff

        if not edge.code_diff_path:
            return None

        path = Path(self.session_root) / edge.code_diff_path
        if not path.exists():
            logger.warning("Missing diff file for edge C%s->C%s: %s", edge.parent_idx, edge.child_idx, path)
            return None

        try:
            if path.suffix == ".gz":
                with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                    text = f.read()
            else:
                with open(path, encoding="utf-8", errors="replace") as f:
                    text = f.read()
        except Exception:
            logger.exception("Failed to read diff file for edge C%s->C%s", edge.parent_idx, edge.child_idx)
            return None

        if edge.code_diff_sha256:
            actual = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
            if actual != edge.code_diff_sha256:
                logger.warning(
                    "Diff hash mismatch for edge C%s->C%s (expected=%s actual=%s)",
                    edge.parent_idx,
                    edge.child_idx,
                    edge.code_diff_sha256,
                    actual,
                )

        return text

    def save(self) -> None:
        """Serialize the DAG to gzip-compressed JSON.

        Uses atomic write (write to temp file, then rename) to avoid
        corruption if the process is killed mid-write.
        No ``indent`` is used to reduce both file size and peak memory.
        """
        self.metadata["last_updated"] = datetime.now(timezone.utc).isoformat()

        migrated = self._migrate_inline_diffs_to_external()
        if migrated:
            logger.info("Migrated %d inline edge diffs to external files", migrated)

        # Serialize pending reflections so CLI-written reflections survive reload
        pending = {}
        if hasattr(self, "_pending_reflections"):
            for idx, entries in self._pending_reflections.items():
                pending[str(idx)] = [e.to_dict() for e in entries]

        data = {
            "metadata": self.metadata,
            "good_patch_definition": self.good_patch_definition,
            "nodes": {str(k): v.to_dict() for k, v in self.nodes.items()},
            "edges": {k: v.to_dict() for k, v in self.edges.items()},
            "scenario_registry": {k: v.to_dict() for k, v in self.scenario_registry.items()},
            "accumulated_lessons": self.accumulated_lessons.to_dict(),
            "pending_reflections": pending,
        }

        self._json_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to a temp file in the same directory, then rename.
        dest = self._gz_path
        fd, tmp_path = tempfile.mkstemp(
            dir=str(dest.parent), suffix=".json.gz.tmp",
        )
        try:
            os.close(fd)
            with gzip.open(tmp_path, "wt", encoding="utf-8", compresslevel=3) as gz:
                json.dump(data, gz, ensure_ascii=False)
            os.replace(tmp_path, str(dest))
        except BaseException:
            # Clean up the temp file on any failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        # Remove legacy uncompressed file if it exists
        if self._json_path.exists():
            try:
                self._json_path.unlink()
            except OSError:
                pass

        logger.info("EvolutionDAG saved to %s", dest)

    def load(self) -> None:
        """Deserialize the DAG from JSON (gzip or plain)."""
        # Prefer gzip; fall back to legacy uncompressed
        if self._gz_path.exists():
            src = self._gz_path
            opener = lambda: gzip.open(src, "rt", encoding="utf-8")  # noqa: E731
        elif self._json_path.exists():
            src = self._json_path
            opener = lambda: open(src, encoding="utf-8")  # noqa: E731
        else:
            logger.info("No existing DAG at %s — starting fresh", self._json_path)
            return

        with opener() as f:
            data = json.load(f)

        self.metadata = data.get("metadata", self.metadata)
        self.good_patch_definition = data.get("good_patch_definition", self.good_patch_definition)

        self.nodes = {int(k): EvolutionNode.from_dict(v) for k, v in data.get("nodes", {}).items()}
        self.edges = {k: EvolutionEdge.from_dict(v) for k, v in data.get("edges", {}).items()}
        self.scenario_registry = {
            k: ScenarioEntry.from_dict(v) for k, v in data.get("scenario_registry", {}).items()
        }
        self.accumulated_lessons = AccumulatedLessons.from_dict(data.get("accumulated_lessons", {}))

        migrated = self._migrate_inline_diffs_to_external()
        if migrated:
            logger.info("Loaded legacy DAG with %d inline diffs (will persist externally on next save)", migrated)

        # Restore pending reflections written by CLI subprocess
        self._pending_reflections: dict[int, list[ReflectionEntry]] = {}
        for idx_str, entries in data.get("pending_reflections", {}).items():
            self._pending_reflections[int(idx_str)] = [
                ReflectionEntry.from_dict(e) for e in entries
            ]

        if self.nodes:
            self._next_idx = max(self.nodes.keys()) + 1
        logger.info("EvolutionDAG loaded: %d nodes, %d edges", len(self.nodes), len(self.edges))

    # ------------------------------------------------------------------
    # Node creation
    # ------------------------------------------------------------------

    def add_seed_node(self, worktree_path: str, score_val: float) -> EvolutionNode:
        """Add the initial seed node (iteration 0)."""
        node = EvolutionNode(
            idx=self._next_idx,
            iteration=0,
            created_at=datetime.now(timezone.utc).isoformat(),
            score_val=score_val,
            val_evaluated=True,
            worktree_path=worktree_path,
        )
        self.nodes[node.idx] = node
        self._next_idx += 1
        self.metadata["total_iterations"] = 0
        self.save()
        return node

    def add_node(
        self,
        *,
        iteration: int,
        worktree_path: str,
        base_parent_idx: int,
        mini_batch_ids: list[str] | None = None,
    ) -> EvolutionNode:
        """Add a new candidate node."""
        node = EvolutionNode(
            idx=self._next_idx,
            iteration=iteration,
            created_at=datetime.now(timezone.utc).isoformat(),
            base_parent_idx=base_parent_idx,
            worktree_path=worktree_path,
            mini_batch_ids=mini_batch_ids or [],
        )
        self.nodes[node.idx] = node
        self._next_idx += 1
        self.metadata["total_iterations"] = max(self.metadata["total_iterations"], iteration)
        return node

    # ------------------------------------------------------------------
    # Edge creation
    # ------------------------------------------------------------------

    @staticmethod
    def _edge_key(parent_idx: int, child_idx: int) -> str:
        return f"{parent_idx}->{child_idx}"

    def add_base_edge(self, parent_idx: int, child_idx: int) -> EvolutionEdge:
        """Create a base edge (direct predecessor → current). Impact fields are None initially."""
        edge = EvolutionEdge(
            parent_idx=parent_idx,
            child_idx=child_idx,
            edge_type="base",
        )
        self.edges[self._edge_key(parent_idx, child_idx)] = edge
        return edge

    def fill_base_edge_impact(
        self,
        parent_idx: int,
        child_idx: int,
        *,
        mini_batch_ids: list[str],
        score_before: float,
        score_after: float,
        scenario_impacts: list[ScenarioImpact],
        code_diff: str,
        files_changed: list[str],
    ) -> None:
        """Fill the impact fields of a base edge after initial/re-evaluation comparison."""
        key = self._edge_key(parent_idx, child_idx)
        edge = self.edges[key]

        self._persist_edge_diff(edge, code_diff)
        edge.files_changed = files_changed
        edge.mini_batch_ids = mini_batch_ids
        edge.score_before = score_before
        edge.score_after = score_after
        edge.score_delta = score_after - score_before
        edge.improved = score_after > score_before

        edge.scenarios_fixed = [si.scenario_id for si in scenario_impacts if si.status_change == "fixed"]
        edge.scenarios_regressed = [si.scenario_id for si in scenario_impacts if si.status_change == "regressed"]
        edge.scenarios_still_failing = [
            si.scenario_id for si in scenario_impacts if si.status_change == "still_failing"
        ]
        edge.scenarios_still_passing = [
            si.scenario_id for si in scenario_impacts if si.status_change == "still_passing"
        ]

    def add_cherry_pick_edges(self, child_idx: int, reference_indices: list[int]) -> list[EvolutionEdge]:
        """Create cherry_pick edges from referenced candidates to this child."""
        created: list[EvolutionEdge] = []
        for ref_idx in reference_indices:
            key = self._edge_key(ref_idx, child_idx)
            if key in self.edges:
                continue  # avoid duplicates (e.g. if ref_idx == base_parent)
            edge = EvolutionEdge(
                parent_idx=ref_idx,
                child_idx=child_idx,
                edge_type="cherry_pick",
            )
            self.edges[key] = edge
            created.append(edge)
        return created

    def fill_cherry_pick_edge_impact(
        self,
        parent_idx: int,
        child_idx: int,
        *,
        mini_batch_ids: list[str],
        score_before: float,
        score_after: float,
        scenario_impacts: list[ScenarioImpact],
        code_diff: str,
        files_changed: list[str],
    ) -> None:
        """Fill the impact fields of a cherry_pick edge after evaluation."""
        key = self._edge_key(parent_idx, child_idx)
        edge = self.edges[key]

        self._persist_edge_diff(edge, code_diff)
        edge.files_changed = files_changed
        edge.mini_batch_ids = mini_batch_ids
        edge.score_before = score_before
        edge.score_after = score_after
        edge.score_delta = score_after - score_before
        edge.improved = score_after > score_before

        edge.scenarios_fixed = [si.scenario_id for si in scenario_impacts if si.status_change == "fixed"]
        edge.scenarios_regressed = [si.scenario_id for si in scenario_impacts if si.status_change == "regressed"]
        edge.scenarios_still_failing = [
            si.scenario_id for si in scenario_impacts if si.status_change == "still_failing"
        ]
        edge.scenarios_still_passing = [
            si.scenario_id for si in scenario_impacts if si.status_change == "still_passing"
        ]

    # ------------------------------------------------------------------
    # Agent record updates
    # ------------------------------------------------------------------

    def update_patch_intent(self, idx: int, intent: PatchIntent) -> None:
        """Set the Agent's PatchIntent for a node."""
        self.nodes[idx].patch_intent = intent

    def update_selection_decision(
        self, idx: int, decision: SelectionDecision,
    ) -> None:
        """Record which candidate(s) were selected in Session 0.

        Creates cherry_pick edges from each selected candidate to this node.
        """
        self.nodes[idx].selection_decision = decision
        for parent_idx in decision.parent_candidates:
            key = self._edge_key(parent_idx, idx)
            if key not in self.edges:
                edge = EvolutionEdge(
                    parent_idx=parent_idx,
                    child_idx=idx,
                    edge_type="cherry_pick",
                )
                self.edges[key] = edge

    def set_reflections(self, idx: int, reflections: list[ReflectionEntry]) -> None:
        """Set the Agent's reflections for a node (called from CLI update-reflection)."""
        if self.nodes[idx].patch_verdict:
            self.nodes[idx].patch_verdict.reflections = reflections
        # Also store individually so CLI can append one by one
        self._pending_reflections: dict[int, list[ReflectionEntry]]
        if not hasattr(self, "_pending_reflections"):
            self._pending_reflections = {}
        self._pending_reflections[idx] = reflections

    def add_reflection(self, idx: int, reflection: ReflectionEntry) -> None:
        """Append a single reflection entry for a node."""
        if not hasattr(self, "_pending_reflections"):
            self._pending_reflections: dict[int, list[ReflectionEntry]] = {}
        self._pending_reflections.setdefault(idx, []).append(reflection)

    def get_pending_reflections(self, idx: int) -> list[ReflectionEntry]:
        """Get accumulated reflections for a node (before verdict is set)."""
        if not hasattr(self, "_pending_reflections"):
            return []
        return self._pending_reflections.get(idx, [])

    # ------------------------------------------------------------------
    # Outer loop updates
    # ------------------------------------------------------------------

    def update_patch_verdict(self, idx: int, verdict: PatchVerdict) -> None:
        """Set the computed PatchVerdict for a node."""
        self.nodes[idx].patch_verdict = verdict

    def update_val_score(self, idx: int, score_val: float) -> None:
        """Update development set score after full eval."""
        self.nodes[idx].score_val = score_val
        self.nodes[idx].val_evaluated = True

    def set_train_scores(self, idx: int, score_before: float, score_after: float) -> None:
        """Set the train initial/re-evaluation scores on the node."""
        self.nodes[idx].score_train_before = score_before
        self.nodes[idx].score_train_after = score_after

    def set_cycle_dirs(
        self,
        idx: int,
        train_before_cycle_dir: str | None = None,
        train_after_cycle_dir: str | None = None,
    ) -> None:
        """Set the cycle directory paths for initial/re-evaluation."""
        if train_before_cycle_dir:
            self.nodes[idx].train_before_cycle_dir = train_before_cycle_dir
        if train_after_cycle_dir:
            self.nodes[idx].train_after_cycle_dir = train_after_cycle_dir

    def set_sdk_session_info(
        self,
        idx: int,
        *,
        selection: SDKSessionInfo | None = None,
        patch: SDKSessionInfo | None = None,
        reflection: SDKSessionInfo | None = None,
    ) -> None:
        """Set SDK session metadata for a node."""
        if selection is not None:
            self.nodes[idx].sdk_session_selection = selection
        if patch is not None:
            self.nodes[idx].sdk_session_patch = patch
        if reflection is not None:
            self.nodes[idx].sdk_session_reflection = reflection

    def set_commit_hash(self, idx: int, commit_hash: str) -> None:
        """Record the commit hash after patching is complete."""
        self.nodes[idx].commit_hash = commit_hash

    def set_accepted(self, idx: int, accepted: bool) -> None:
        """Record the engine's acceptance decision for a node."""
        self.nodes[idx].accepted = accepted

    # ------------------------------------------------------------------
    # Query methods (used by CLI)
    # ------------------------------------------------------------------

    def get_summary(self) -> dict[str, Any]:
        """Return DAG-wide summary for ``evo-dag summary``."""
        best_val_idx: int | None = None
        best_val_score: float = -1.0
        for idx, node in self.nodes.items():
            if node.abandoned:
                continue
            if node.score_val is not None and node.score_val > best_val_score:
                best_val_score = node.score_val
                best_val_idx = idx

        # Current base parent = non-abandoned node with highest iteration
        eligible = [n for n in self.nodes.values() if not n.abandoned]
        if not eligible:
            eligible = list(self.nodes.values())
        current_base = max(eligible, key=lambda n: n.iteration) if eligible else None

        edge_summaries: list[dict[str, Any]] = []
        for edge in sorted(self.edges.values(), key=lambda e: (e.child_idx, e.parent_idx)):
            info: dict[str, Any] = {
                "parent": f"C{edge.parent_idx}" if edge.parent_idx > 0 else "seed",
                "child": f"C{edge.child_idx}",
                "type": edge.edge_type,
            }
            if edge.edge_type == "base" and edge.score_delta is not None:
                info["delta"] = edge.score_delta
                if edge.scenarios_regressed:
                    info["regression"] = True
            edge_summaries.append(info)

        return {
            "total_iterations": self.metadata["total_iterations"],
            "num_nodes": len(self.nodes),
            "best_val_idx": best_val_idx,
            "best_val_score": best_val_score if best_val_idx is not None else None,
            "current_base_idx": current_base.idx if current_base else None,
            "edges": edge_summaries,
        }

    def get_node(self, idx: int) -> EvolutionNode:
        """Get a node by index."""
        return self.nodes[idx]

    def get_edge(self, parent_idx: int, child_idx: int) -> EvolutionEdge:
        """Get a specific edge."""
        return self.edges[self._edge_key(parent_idx, child_idx)]

    def get_edges_for_node(self, idx: int) -> list[EvolutionEdge]:
        """Get all parent edges for a node."""
        return [e for e in self.edges.values() if e.child_idx == idx]

    def get_children_edges(self, idx: int) -> list[EvolutionEdge]:
        """Get all child edges from a node."""
        return [e for e in self.edges.values() if e.parent_idx == idx]

    def get_scenario(self, scenario_id: str) -> ScenarioEntry | None:
        """Get scenario registry entry."""
        return self.scenario_registry.get(scenario_id)

    def get_lessons(self) -> AccumulatedLessons:
        """Get accumulated lessons."""
        return self.accumulated_lessons

    def get_current_batch(self) -> dict[str, Any]:
        """Get the current iteration's mini-batch info."""
        if not self.nodes:
            return {"status": "no nodes"}

        latest = max(self.nodes.values(), key=lambda n: n.iteration)
        base_edge = None
        for e in self.edges.values():
            if e.child_idx == latest.idx and e.edge_type == "base":
                base_edge = e
                break

        return {
            "iteration": latest.iteration,
            "candidate_idx": latest.idx,
            "mini_batch_ids": latest.mini_batch_ids,
            "base_parent_idx": latest.base_parent_idx,
            "worktree_path": latest.worktree_path,
            "train_before_cycle_dir": latest.train_before_cycle_dir,
            "train_after_cycle_dir": latest.train_after_cycle_dir,
            "score_train_before": latest.score_train_before,
            "score_train_after": latest.score_train_after,
            "base_edge_score_before": base_edge.score_before if base_edge else None,
            "base_edge_score_after": base_edge.score_after if base_edge else None,
        }

    def get_lineage(self) -> dict[str, Any]:
        """Get DAG topology for visualization."""
        nodes_info: list[dict[str, Any]] = []
        for idx in sorted(self.nodes.keys()):
            node = self.nodes[idx]
            label = "seed" if idx == 0 else f"C{idx}"
            nodes_info.append({
                "idx": idx,
                "label": label,
                "iteration": node.iteration,
                "score_val": node.score_val,
            })

        edges_info: list[dict[str, Any]] = []
        for edge in self.edges.values():
            parent_label = "seed" if edge.parent_idx == 0 else f"C{edge.parent_idx}"
            child_label = f"C{edge.child_idx}"
            edge_info: dict[str, Any] = {
                "parent": parent_label,
                "child": child_label,
                "type": edge.edge_type,
            }
            # Include score delta only for base edges in lineage view
            if edge.edge_type == "base":
                edge_info["delta"] = edge.score_delta
            edges_info.append(edge_info)

        return {"nodes": nodes_info, "edges": edges_info}
