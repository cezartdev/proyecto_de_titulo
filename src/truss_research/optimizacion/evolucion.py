"""NSGA-II de pymoo adaptado a consultas de un catálogo de diseños registrados."""

from pymoo.algorithms.moo.nsga2 import binary_tournament
from pymoo.core.mixed import (
    MixedVariableDuplicateElimination,
    MixedVariableMating,
)
from pymoo.core.problem import Problem
from pymoo.core.variable import Choice, Integer, Real
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.ux import UX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.mutation.rm import ChoiceRandomMutation
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.selection.tournament import TournamentSelection

from truss_research.cerchas.schemas import (
    DESIGN_BOUNDS,
    INPUT_COLUMNS,
    INTEGER_COLUMNS,
    WEB_CONFIG_VALUES,
)
from truss_research.evaluacion.pareto import OBJECTIVE_COUNT
from truss_research.optimizacion.contratos import (
    EvolutionParameters,
)
from truss_research.surrogates.configuracion import FloatArray

type DesignGenes = dict[str, int | float]


def mixed_problem() -> Problem:
    variables: dict[str, Choice | Integer | Real] = {
        name: Integer(bounds=DESIGN_BOUNDS[name])
        if name in INTEGER_COLUMNS
        else Real(bounds=DESIGN_BOUNDS[name])
        for name in INPUT_COLUMNS
    }
    variables[INPUT_COLUMNS[0]] = Choice(options=WEB_CONFIG_VALUES)
    return Problem(vars=variables, n_obj=OBJECTIVE_COUNT, n_ieq_constr=2)


def mixed_mating(parameters: EvolutionParameters) -> MixedVariableMating:
    probability = 1 / len(INPUT_COLUMNS)
    return MixedVariableMating(
        selection=TournamentSelection(func_comp=binary_tournament),
        crossover={
            Choice: UX(prob=parameters.crossover_probability),
            Real: SBX(
                prob=parameters.crossover_probability, eta=parameters.crossover_eta
            ),
            Integer: SBX(
                prob=parameters.crossover_probability,
                eta=parameters.crossover_eta,
                vtype=float,
                repair=RoundingRepair(),
            ),
        },
        mutation={
            Choice: ChoiceRandomMutation(prob=1.0, prob_var=probability),
            Real: PM(prob=1.0, prob_var=probability, eta=parameters.mutation_eta),
            Integer: PM(
                prob=1.0,
                prob_var=probability,
                eta=parameters.mutation_eta,
                vtype=float,
                repair=RoundingRepair(),
            ),
        },
        eliminate_duplicates=MixedVariableDuplicateElimination(),
        n_max_iterations=parameters.max_mating_attempts,
    )


def design_genes(values: FloatArray) -> DesignGenes:
    return {
        name: int(value) if name in INTEGER_COLUMNS else float(value)
        for name, value in zip(INPUT_COLUMNS, values, strict=True)
    }
