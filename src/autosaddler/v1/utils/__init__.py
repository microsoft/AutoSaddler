"""Utilities for optimization control.

Re-exports:
    **Stop conditions** — control when optimization terminates:
    ``MaxMetricCallsStopper``, ``FileStopper``, ``CompositeStopper``,
    ``MaxCandidateProposalsStopper``.
"""

from .stop_condition import (
    CompositeStopper,
    FileStopper,
    MaxCandidateProposalsStopper,
    MaxMetricCallsStopper,
    StopperProtocol,
)

__all__ = [
    "CompositeStopper",
    "FileStopper",
    "MaxCandidateProposalsStopper",
    "MaxMetricCallsStopper",
    "StopperProtocol",
]
