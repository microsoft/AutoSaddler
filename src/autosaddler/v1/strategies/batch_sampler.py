# Based on GEPA by Lakshya A Agrawal (github.com/gepa-ai/gepa)


from __future__ import annotations

import json
import logging
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from autosaddler.v1.core.adapter import DataInst
from autosaddler.v1.core.data_loader import DataId, DataLoader
from autosaddler.v1.core.state import GEPAState

if TYPE_CHECKING:
    from autosaddler.v1.proposer.autosaddler.pattern_registry import PatternRegistry

logger = logging.getLogger(__name__)


def _json_to_rng_state(s: list) -> tuple:
    """Convert a JSON-loaded ``random.getstate()`` payload back to a tuple.

    ``json`` serialises the ``(version, internalstate, gauss_next)`` tuple as
    nested lists; ``random.setstate`` requires the inner state to be a tuple.
    """
    return (s[0], tuple(s[1]), s[2])


class BatchSampler(Protocol[DataId, DataInst]):
    def next_minibatch_ids(self, loader: DataLoader[DataId, DataInst], state: GEPAState) -> list[DataId]: ...


class EpochShuffledBatchSampler(BatchSampler[DataId, DataInst]):
    """
    Mirrors the original batching logic:
    - Shuffle ids each epoch
    - Pad to minibatch size with least frequent ids
    - Deterministic via state.rng1
    """

    def __init__(self, minibatch_size: int, rng: random.Random | None = None):
        self.minibatch_size = minibatch_size
        self.shuffled_ids: list[DataId] = []
        self.epoch = -1
        self.id_freqs = Counter()
        self.last_trainset_size = 0
        if rng is None:
            self.rng = random.Random(0)
        else:
            self.rng = rng

    def _update_shuffled(self, loader: DataLoader[DataId, DataInst]):
        all_ids = list(loader.all_ids())
        trainset_size = len(loader)
        self.last_trainset_size = trainset_size

        if trainset_size == 0:
            self.shuffled_ids = []
            self.id_freqs = Counter()
            return

        self.shuffled_ids = list(all_ids)
        self.rng.shuffle(self.shuffled_ids)
        self.id_freqs = Counter(self.shuffled_ids)

        mod = trainset_size % self.minibatch_size
        num_to_pad = (self.minibatch_size - mod) if mod != 0 else 0
        if num_to_pad > 0:
            for _ in range(num_to_pad):
                selected_id = self.id_freqs.most_common()[::-1][0][0]
                self.shuffled_ids.append(selected_id)
                self.id_freqs[selected_id] += 1

    def next_minibatch_ids(self, loader: DataLoader[DataId, DataInst], state: GEPAState) -> list[DataId]:
        trainset_size = len(loader)
        if trainset_size == 0:
            raise ValueError("Cannot sample a minibatch from an empty loader.")

        base_idx = state.i * self.minibatch_size
        curr_epoch = 0 if self.epoch == -1 else base_idx // max(len(self.shuffled_ids), 1)

        needs_refresh = not self.shuffled_ids or trainset_size != self.last_trainset_size or curr_epoch > self.epoch
        if needs_refresh:
            self.epoch = curr_epoch
            self._update_shuffled(loader)

        assert len(self.shuffled_ids) >= self.minibatch_size
        assert len(self.shuffled_ids) % self.minibatch_size == 0

        base_idx = base_idx % len(self.shuffled_ids)
        end_idx = base_idx + self.minibatch_size
        assert end_idx <= len(self.shuffled_ids)
        return self.shuffled_ids[base_idx:end_idx]


class ActiveSaddlerBanditSampler(BatchSampler[DataId, DataInst]):
    """ActiveSaddler infinite-armed bandit sampler.

    Each call to :meth:`next_minibatch_ids` performs exactly ONE action,
    decided by the agent in Session 3.5 and passed in as ``forced_action``:

    * **Unseen draw** (``"draw"``, or when no arm exists yet): sample up to
      ``B`` scenarios from the unseen pool ``U_t = D_tr \\ Exec_t`` to open
      new failure-pattern regions.
    * **Arm pull** (``"pull"``): sample exactly ONE existing arm (failure
      pattern) ``p_t`` with probability proportional to a softmax of its
      agent learning-progress score ``phi_t(p)`` (temperature ``tau``, with a
      per-arm floor ``epsilon``), then execute up to ``B`` of that arm's
      scenarios (a uniform random subsample when it owns more than ``B``).

    Symbols:
      * ``|P_t|`` — number of instantiated arms (patterns owning >= 1 scenario).
      * ``N_t``  — cumulative number of DISTINCT ``(scenario, harness)`` probe
        points executed so far. Repeat rollouts of the same pair are NOT
        counted. Persisted (with ``Exec_t``) so it survives restarts.

    Scoring (Session 4, in :class:`PatternRegistry`), arm creation (Session
    3.5) and arm selection (softmax, here) are decoupled decisions.
    """

    def __init__(
        self,
        minibatch_size: int,
        pattern_registry: PatternRegistry,
        eta: float = 0.3,
        temperature: float = 0.5,
        min_prob: float = 0.05,
        scenario_to_idx: dict[str, DataId] | None = None,
        state_path: str | Path | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.minibatch_size = minibatch_size
        self.pattern_registry = pattern_registry
        self.eta = eta
        self.temperature = max(float(temperature), 1e-6)
        self.min_prob = min_prob
        self.rng = rng or random.Random(42)

        # Mapping from full scenario names (used in pattern tuples) to DataId
        # indices (used by the GEPA DataLoader). The pattern registry stores
        # full scenario names while the framework uses integer indices.
        self._scenario_to_idx: dict[str, DataId] = scenario_to_idx or {}

        # ---- Persistent bandit state (survives restarts) ----------------
        self._state_path = Path(state_path) if state_path else None
        # Scenarios executed at least once -> defines the unseen pool U_t.
        self._executed_ids: set[DataId] = set()
        # Distinct "(scenario, harness)" probe points -> defines N_t.
        self._probe_points: set[str] = set()
        # Canonical shuffled order for deterministic unseen draws (restored
        # from persisted state by _load_state() when resuming).
        self._shuffled_order: list[DataId] | None = None
        self._load_state()

        # IDs drawn from the unseen pool in the most recent call.
        self._last_unseen_ids: list[DataId] = []

        # Analysis snapshot of the most recent scoring/decision; persisted by
        # the proposer for later inspection. No effect on sampling.
        self.last_score_snapshot: dict | None = None

    # ------------------------------------------------------------------
    # Persistent state (N_t probe points + executed-scenario pool)
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        if self._state_path is None or not self._state_path.exists():
            return
        try:
            with open(self._state_path, encoding="utf-8") as f:
                data = json.load(f)
            required = {
                "executed_ids",
                "probe_points",
                "rng_state",
                "shuffled_order",
            }
            missing = sorted(required - set(data))
            if missing:
                raise ValueError(
                    "missing deterministic fields: " + ", ".join(missing)
                )
            self._executed_ids = set(data.get("executed_ids", []))
            self._probe_points = set(data.get("probe_points", []))
            # Restore RNG + shuffled order for deterministic resume. JSON stores
            # getstate() tuples as nested lists, so re-tuple before setstate().
            rng_state = data.get("rng_state")
            self.rng.setstate(_json_to_rng_state(rng_state))
            logger.info(
                "Restored bandit RNG state from %s", self._state_path,
            )
            shuffled = data.get("shuffled_order")
            self._shuffled_order = list(shuffled)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load deterministic bandit state from {self._state_path}"
            ) from exc

    def _save_state(self) -> None:
        if self._state_path is None:
            return
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "executed_ids": sorted(self._executed_ids),
                    "probe_points": sorted(self._probe_points),
                    # RNG + shuffled order for deterministic resume.
                    "rng_state": self.rng.getstate(),
                    "shuffled_order": self._shuffled_order,
                },
                f,
            )
        os.replace(tmp_path, self._state_path)

    def mark_executed(self, ids: list[DataId]) -> None:
        """Mark scenario IDs as executed (removes them from the unseen pool)."""
        before = len(self._executed_ids)
        self._executed_ids.update(ids)
        if len(self._executed_ids) != before:
            self._save_state()

    def record_probe_points(
        self,
        scenario_names: list[str],
        harness_tag: str,
    ) -> list[str]:
        """Record distinct ``(scenario, harness)`` probe points for ``N_t``.

        ``harness_tag`` identifies the harness version under which the
        scenarios were executed (a commit hash, or a stable fallback tag). A
        given ``(scenario, harness_tag)`` pair is counted once; re-running the
        same scenario on the same harness does not increase ``N_t``.
        """
        if not harness_tag:
            return []
        added_keys: list[str] = []
        for name in scenario_names:
            key = f"{name}\t{harness_tag}"
            if key not in self._probe_points:
                self._probe_points.add(key)
                added_keys.append(key)
        if added_keys:
            self._save_state()
        return added_keys

    @property
    def n_probes(self) -> int:
        """``N_t``: number of distinct (scenario, harness) probe points so far."""
        return len(self._probe_points)

    @property
    def last_unseen_ids(self) -> list[DataId]:
        """IDs drawn from the unseen pool in the most recent batch."""
        return list(self._last_unseen_ids)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _decide(
        self, loader: DataLoader[DataId, DataInst], state: GEPAState,
        forced_action: str | None = None,
    ) -> dict:
        """Compute the arm-creation action (pull vs unseen draw) with NO side effects.

        ``forced_action`` is the agent's explicit decision: ``"pull"`` forces an
        arm pull, ``"draw"`` forces an unseen draw. A cold start with no arms
        always draws regardless. Does not touch the RNG.
        """
        all_ids_ordered = list(loader.all_ids())
        all_ids = set(all_ids_ordered)

        # Instantiated arms = patterns owning >= 1 available scenario.
        arm_candidates: dict[str, list[DataId]] = {}
        for pid in self.pattern_registry.patterns:
            cand: list[DataId] = []
            for name in self.pattern_registry.get_scenarios_for_pattern(pid):
                did = self._scenario_to_idx.get(name)
                if did is not None and did in all_ids:
                    cand.append(did)
            if cand:
                arm_candidates[pid] = cand
        num_arms = len(arm_candidates)

        unseen_pool = all_ids - self._executed_ids
        n_probes = self.n_probes

        # (B) Arm-creation decision: the agent's explicit choice. A cold start
        # (no arms) always draws.
        if num_arms == 0:
            want_unseen = True
        elif forced_action == "draw":
            want_unseen = True
        elif forced_action == "pull":
            want_unseen = False
        else:
            raise ValueError(
                f"forced_action must be 'pull' or 'draw', got {forced_action!r}"
            )

        if want_unseen and unseen_pool:
            action = "unseen_draw"
        elif arm_candidates:
            action = "arm_pull"
        else:
            action = "empty"

        return {
            "all_ids_ordered": all_ids_ordered,
            "arm_candidates": arm_candidates,
            "num_arms": num_arms,
            "unseen_pool": unseen_pool,
            "n_probes": n_probes,
            "action": action,
        }

    def unseen_pool_size(self, loader: DataLoader[DataId, DataInst]) -> int:
        """Number of never-executed scenarios remaining (|U_t|).

        Context for the agent's pull-vs-draw decision.
        """
        return len(set(loader.all_ids()) - self._executed_ids)

    def _arm_scores(
        self, current_iter: int, breakdown: dict[str, dict],
    ) -> dict[str, float]:
        """Per-arm score phi_t(p): the agent's latest learning-progress score.

        Used DIRECTLY (no EMA). Populates ``breakdown`` for the snapshot.
        """
        return self.pattern_registry.compute_agent_scores(
            breakdown=breakdown,
            eta=self.eta,
            required_iteration=current_iter,
        )

    def next_minibatch_ids(
        self, loader: DataLoader[DataId, DataInst], state: GEPAState,
        forced_action: str | None = None,
    ) -> list[DataId]:
        """Return the next mini-batch via the agent's decision + softmax pull.

        See :meth:`_decide` for ``forced_action``.
        """
        decision = self._decide(loader, state, forced_action=forced_action)
        all_ids_ordered = decision["all_ids_ordered"]
        arm_candidates = decision["arm_candidates"]
        unseen_pool = decision["unseen_pool"]
        action = decision["action"]
        current_iter = state.i + 1  # 1-indexed iteration (display only)

        if self._shuffled_order is None:
            self._shuffled_order = list(all_ids_ordered)
            self.rng.shuffle(self._shuffled_order)

        # Scores for the analysis snapshot (and arm-pull selection).
        breakdown: dict[str, dict] = {}
        self._arm_scores(current_iter, breakdown)

        self._last_unseen_ids = []
        chosen_arm: str | None = None
        probs: dict[str, float] = {}

        if action == "unseen_draw":
            ordered_pool = [d for d in self._shuffled_order if d in unseen_pool]
            selected = ordered_pool[: self.minibatch_size]
            self._last_unseen_ids = list(selected)
        elif action == "arm_pull":
            arm_scores = {
                pid: breakdown.get(pid, {}).get("score", 0.0) for pid in arm_candidates
            }
            probs = self._softmax_floor(arm_scores)
            chosen_arm = self._sample_arm(probs)
            cand = arm_candidates[chosen_arm]
            if len(cand) <= self.minibatch_size:
                selected = list(cand)
            else:
                selected = self.rng.sample(cand, self.minibatch_size)
        else:
            # No arm and no unseen scenario left: nothing to sample.
            selected = []

        final_batch = list(selected)[: self.minibatch_size]
        self.last_score_snapshot = self._build_score_snapshot(
            current_iter=current_iter,
            action=action,
            num_arms=decision["num_arms"],
            n_probes=decision["n_probes"],
            unseen_pool=unseen_pool,
            all_ids_ordered=all_ids_ordered,
            breakdown=breakdown,
            arm_candidates=arm_candidates,
            probs=probs,
            chosen_arm=chosen_arm,
            final_batch=final_batch,
        )
        # Persist RNG + shuffled order every iteration. Boundary-consistent:
        # the RNG is only consumed above (shuffle/choices/sample), so this
        # captures the post-sampling state used by deterministic resume.
        self._save_state()
        return final_batch

    def _softmax_floor(self, scores: dict[str, float]) -> dict[str, float]:
        """Softmax over arm scores (temperature ``tau``) with a per-arm floor.

        Returns a probability distribution where every arm has probability at
        least ``min_prob`` (when feasible), so low-scoring arms remain eligible
        for periodic re-verification (silent-regression checks).
        """
        pids = list(scores.keys())
        k = len(pids)
        if k == 0:
            return {}
        if k == 1:
            return {pids[0]: 1.0}

        m = max(scores.values())
        exps = {pid: math.exp((scores[pid] - m) / self.temperature) for pid in pids}
        z = sum(exps.values()) or 1.0
        soft = {pid: exps[pid] / z for pid in pids}

        eps = self.min_prob
        if eps > 0.0 and eps * k < 1.0:
            return {pid: eps + (1.0 - eps * k) * soft[pid] for pid in pids}
        if eps > 0.0:
            # Floor too large to satisfy simultaneously -> uniform.
            return {pid: 1.0 / k for pid in pids}
        return soft

    def _sample_arm(self, probs: dict[str, float]) -> str:
        pids = list(probs.keys())
        weights = [probs[pid] for pid in pids]
        return self.rng.choices(pids, weights=weights, k=1)[0]

    def _build_score_snapshot(
        self,
        *,
        current_iter: int,
        action: str,
        num_arms: int,
        n_probes: int,
        unseen_pool: set,
        all_ids_ordered: list,
        breakdown: dict,
        arm_candidates: dict,
        probs: dict,
        chosen_arm: str | None,
        final_batch: list,
    ) -> dict:
        """Build a serialisable snapshot of this iteration's decision.

        Analysis/debugging only — no effect on sampling.
        """
        idx_to_scenario = {v: k for k, v in self._scenario_to_idx.items()}

        arms: list[dict] = []
        for pid, info in breakdown.items():
            if pid not in arm_candidates:
                continue  # not an instantiated arm (no available scenarios)
            arms.append({
                "pattern_id": pid,
                "label": info.get("label"),
                "scenarios": info.get("scenarios", []),
                "num_scenarios": len(arm_candidates.get(pid, [])),
                "observations": info.get("observations", []),
                "num_observations": info.get("num_observations"),
                "ema": info.get("ema"),
                "severity": info.get("severity"),
                "fixability": info.get("fixability"),
                "breadth": info.get("breadth"),
                "side_effect": info.get("side_effect"),
                "rationale": info.get("rationale"),
                "score": info.get("score"),
                "prob": probs.get(pid),
                "selected": pid == chosen_arm,
            })
        arms.sort(key=lambda a: (a["score"] is None, -(a["score"] or 0.0)))

        unseen_scenarios = [
            idx_to_scenario.get(did, str(did))
            for did in all_ids_ordered
            if did in unseen_pool
        ]

        return {
            "iteration": current_iter,
            "strategy": "activesaddler",
            "action": action,
            "eta": self.eta,
            "temperature": self.temperature,
            "min_prob": self.min_prob,
            "minibatch_size": self.minibatch_size,
            "num_arms": num_arms,
            "n_probes": n_probes,
            "chosen_arm": chosen_arm,
            "score_formula": (
                "phi(p) = mean(severity, fixability, breadth, 1-side_effect); "
                "P(pull p) = softmax(phi/tau) floored at min_prob"
            ),
            "selected_minibatch": [
                idx_to_scenario.get(did, str(did)) for did in final_batch
            ],
            "num_unseen_remaining": len(unseen_scenarios),
            "unseen_scenarios": unseen_scenarios,
            "probe_points_added": [],
            "arms": arms,
        }

