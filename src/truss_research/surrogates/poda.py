"""Poda conservadora de búsquedas a partir de pliegues internos completos."""

import logging
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING

from truss_research.datos.procesados import JsonValue

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PruningParameters:
    enabled: bool = False
    startup_trials: int = 5
    minimum_folds: int = 2

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("pruning.enabled debe ser booleano")
        if type(self.startup_trials) is not int or self.startup_trials < 1:
            raise ValueError("La poda requiere ensayos iniciales completos positivos")
        if type(self.minimum_folds) is not int or self.minimum_folds < 2:
            raise ValueError("La poda requiere al menos dos pliegues completos")


@dataclass(frozen=True)
class FoldProgress:
    scores: tuple[float, ...]
    total_folds: int


def create_pruner(parameters: PruningParameters) -> "optuna.pruners.BasePruner":
    import optuna

    if not parameters.enabled:
        return optuna.pruners.NopPruner()
    return optuna.pruners.MedianPruner(
        n_startup_trials=parameters.startup_trials,
        n_warmup_steps=parameters.minimum_folds,
        n_min_trials=parameters.startup_trials,
    )


def report_fold(
    trial: "optuna.Trial", progress: FoldProgress, parameters: PruningParameters
) -> None:
    if not parameters.enabled:
        return
    import optuna

    completed = len(progress.scores)
    if not 0 < completed <= progress.total_folds or not all(
        map(isfinite, progress.scores)
    ):
        raise ValueError("Progreso o puntuaciones inválidas para poda")
    trial.set_user_attr("fold_scores", list(progress.scores))
    trial.set_user_attr("completed_folds", completed)
    trial.set_user_attr("planned_folds", progress.total_folds)
    trial.set_user_attr("skipped_folds", 0)
    if completed < parameters.minimum_folds:
        return
    trial.report(sum(progress.scores) / completed, step=completed)
    if completed == progress.total_folds or not trial.should_prune():
        return
    trial.set_user_attr("skipped_folds", progress.total_folds - completed)
    logger.info(
        "Poda: ensayo %s tras %s/%s pliegues",
        trial.number,
        completed,
        progress.total_folds,
    )
    raise optuna.TrialPruned(
        f"Candidato descartado tras {completed}/{progress.total_folds} pliegues"
    )


def pruning_summary(study: "optuna.Study") -> dict[str, JsonValue]:
    trials = study.get_trials(deepcopy=False)
    counts = {
        state: sum(trial.state.name == state for trial in trials)
        for state in ("COMPLETE", "PRUNED", "FAIL", "RUNNING")
    }
    return {
        "trials": len(trials),
        "states": counts,
        "reported_completed_folds": sum(
            int(trial.user_attrs.get("completed_folds", 0)) for trial in trials
        ),
        "skipped_folds": sum(
            int(trial.user_attrs.get("skipped_folds", 0)) for trial in trials
        ),
        "best_trial": study.best_trial.number if counts["COMPLETE"] else None,
    }
