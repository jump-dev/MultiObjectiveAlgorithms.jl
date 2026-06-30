#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

"""
    GeneralDichotomy(precision::Int = 3)

`GeneralDichotomy` implements the algorithm from Buchet, S. and Defresne, M.
(2026). Efficient Enumeration of Supported Solutions for General Multi-Objective
Optimization Problems. ⟨hal-05514317⟩

This implementation was contributed to MultiObjectiveAlgorithms.jl by the
authors. Their upstream repository is: https://forge.inrae.com/opteam/generaldichotomy

## Supported problem classes

This algorithm supports all problem classes.

## Compat

To use this algorithm you MUST first load the Polyhedra.jl Julia package:

```julia
import MultiObjectiveAlgorithms as MOA
import Polyhedra
algorithm = MOA.GeneralDichotomy(3)
```

## Supported optimizer attributes

 * `MOI.TimeLimitSec()`: terminate if the time limit is exceeded and return the
   list of current solutions.

 * `MOA.SolutionLimit()`: terminate once this many solutions have been found.
"""
mutable struct GeneralDichotomy <: AbstractAlgorithm
    solution_limit::Union{Nothing,Int}
    max_iter::Int
    epsilon::Float64
    scaling::Float64

    function GeneralDichotomy(precision::Int = 3)
        return new(nothing, 0, 10.0^-precision, 10^precision)
    end
end

MOI.supports(::GeneralDichotomy, ::SolutionLimit) = true

function MOI.get(alg::GeneralDichotomy, attr::SolutionLimit)
    return something(alg.solution_limit, _default(alg, attr))
end

function MOI.set(alg::GeneralDichotomy, ::SolutionLimit, value)
    alg.solution_limit = value
    return
end

function _solve_weighted_sum(
    model::Optimizer,
    ::GeneralDichotomy,
    weight::Vector{Float64},
)
    f = _scalarise(model.f, weight)
    MOI.set(model.inner, MOI.ObjectiveFunction{typeof(f)}(), f)
    optimize_inner!(model)
    status = MOI.get(model.inner, MOI.TerminationStatus())
    if !_is_scalar_status_optimal(status)
        return status, nothing
    end
    variables = MOI.get(model.inner, MOI.ListOfVariableIndices())
    X, Y = _compute_point(model, variables, model.f)
    _log_subproblem_solve(model, Y)
    return status, SolutionPoint(X, Y)
end
