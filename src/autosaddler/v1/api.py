import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from autosaddler.v1.core.callbacks import GEPACallback

from autosaddler.v1.core.adapter import DataInst, GEPAAdapter, RolloutOutput, Trajectory
from autosaddler.v1.core.data_loader import DataId, DataLoader, ensure_loader
from autosaddler.v1.core.engine import GEPAEngine
from autosaddler.v1.core.result import GEPAResult
from autosaddler.v1.core.state import EvaluationCache, FrontierType
from autosaddler.v1.logging.experiment_tracker import create_experiment_tracker
from autosaddler.v1.logging.logger import Logger, LoggerProtocol, StdOutLogger
from autosaddler.v1.proposer.base import ProposeNewCandidate
from autosaddler.v1.strategies.eval_policy import EvaluationPolicy, FullEvaluationPolicy
from autosaddler.v1.utils import FileStopper, StopperProtocol


def optimize(
    seed_candidate: dict[str, str],
    trainset: list[DataInst] | DataLoader[DataId, DataInst],
    valset: list[DataInst] | DataLoader[DataId, DataInst] | None = None,
    adapter: GEPAAdapter[DataInst, Trajectory, RolloutOutput] | None = None,
    *,
    reflective_proposer_override: ProposeNewCandidate[DataId],
    frontier_type: FrontierType = "instance",
    perfect_score: float = 1.0,
    max_metric_calls: int | None = None,
    stop_callbacks: StopperProtocol | Sequence[StopperProtocol] | None = None,
    logger: LoggerProtocol | None = None,
    run_dir: str | None = None,
    callbacks: "list[GEPACallback] | None" = None,
    display_progress_bar: bool = False,
    use_cloudpickle: bool = False,
    cache_evaluation: bool = False,
    seed: int = 0,
    raise_on_exception: bool = True,
    val_evaluation_policy: EvaluationPolicy[DataId, DataInst] | Literal["full_eval"] | None = None,
    # Legacy params accepted but ignored (for config compatibility)
    reflection_lm: Any = None,
    reflection_minibatch_size: int | None = None,
    candidate_selection_strategy: Any = "pareto",
    module_selector: Any = "all",
    skip_perfect_score: bool = True,
    use_merge: bool = False,
    reflection_prompt_template: Any = None,
    use_wandb: bool = False,
    wandb_api_key: str | None = None,
    wandb_init_kwargs: dict[str, Any] | None = None,
    use_mlflow: bool = False,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str | None = None,
    track_best_outputs: bool = False,
    batch_sampler: Any = "epoch_shuffled",
) -> GEPAResult[RolloutOutput, DataId]:
    """Run AutoSaddler optimization loop."""
    if seed_candidate is None or not seed_candidate:
        raise ValueError("seed_candidate must contain at least one component text.")

    if adapter is None:
        raise ValueError("An adapter must be provided.")

    train_loader = ensure_loader(trainset)
    val_loader = ensure_loader(valset) if valset is not None else train_loader

    # Build stop callbacks
    stop_callbacks_list: list[StopperProtocol] = []
    if stop_callbacks is not None:
        if isinstance(stop_callbacks, Sequence):
            stop_callbacks_list.extend(stop_callbacks)
        else:
            stop_callbacks_list.append(stop_callbacks)

    if run_dir is not None:
        stop_file_path = os.path.join(run_dir, "autosaddler.stop")
        stop_callbacks_list.append(FileStopper(stop_file_path))

    if max_metric_calls is not None:
        from autosaddler.v1.utils import MaxMetricCallsStopper
        stop_callbacks_list.append(MaxMetricCallsStopper(max_metric_calls))

    if not stop_callbacks_list:
        raise ValueError("Provide at least one of stop_callbacks or max_metric_calls.")

    stop_callback: StopperProtocol
    if len(stop_callbacks_list) == 1:
        stop_callback = stop_callbacks_list[0]
    else:
        from autosaddler.v1.utils import CompositeStopper
        stop_callback = CompositeStopper(*stop_callbacks_list)

    if logger is None:
        if run_dir is not None:
            os.makedirs(run_dir, exist_ok=True)
            logger = Logger(os.path.join(run_dir, "run_log.txt"))
        else:
            logger = StdOutLogger()

    if val_evaluation_policy is None or val_evaluation_policy == "full_eval":
        val_evaluation_policy = FullEvaluationPolicy()

    experiment_tracker = create_experiment_tracker(
        use_wandb=use_wandb,
        wandb_api_key=wandb_api_key,
        wandb_init_kwargs=wandb_init_kwargs,
        use_mlflow=use_mlflow,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )

    evaluation_cache: EvaluationCache[RolloutOutput, DataId] | None = None
    if cache_evaluation:
        evaluation_cache = EvaluationCache[RolloutOutput, DataId]()

    engine = GEPAEngine(
        adapter=adapter,
        run_dir=run_dir,
        valset=val_loader,
        seed_candidate=seed_candidate,
        perfect_score=perfect_score,
        seed=seed,
        reflective_proposer=reflective_proposer_override,
        merge_proposer=None,
        frontier_type=frontier_type,
        logger=logger,
        experiment_tracker=experiment_tracker,
        callbacks=callbacks,
        track_best_outputs=track_best_outputs,
        display_progress_bar=display_progress_bar,
        raise_on_exception=raise_on_exception,
        stop_callback=stop_callback,
        val_evaluation_policy=val_evaluation_policy,
        use_cloudpickle=use_cloudpickle,
        evaluation_cache=evaluation_cache,
    )

    with experiment_tracker:
        if isinstance(logger, Logger):
            with logger:
                state = engine.run()
        else:
            state = engine.run()

    return GEPAResult.from_state(state, run_dir=run_dir, seed=seed)
