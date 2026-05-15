"""Legacy AutoSaddler V1 API."""

from .api import optimize
from .core.adapter import EvaluationBatch, GEPAAdapter
from .core.result import GEPAResult
from .utils.stop_condition import (
    CompositeStopper,
    FileStopper,
    MaxCandidateProposalsStopper,
    MaxMetricCallsStopper,
    StopperProtocol,
)

__all__ = [
    "CompositeStopper",
    "EvaluationBatch",
    "FileStopper",
    "GEPAAdapter",
    "GEPAResult",
    "MaxCandidateProposalsStopper",
    "MaxMetricCallsStopper",
    "StopperProtocol",
    "optimize",
]