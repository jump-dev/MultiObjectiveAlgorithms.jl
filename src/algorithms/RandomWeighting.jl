#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

"""
    RandomWeighting()

A heuristic solver that works by repeatedly solving a weighted sum problem with
random weights.

## Supported optimizer attributes

 * `MOI.TimeLimitSec()`: terminate if the time limit is exceeded and return the
   list of current solutions.

 * `MOI.SolutionLimit()`: terminate once this many solutions have been found.

At least one of these two limits must be set.
"""
mutable struct RandomWeighting <: AbstractAlgorithm end

function optimize_multiobjective!(algorithm::RandomWeighting, model::Optimizer)
    if MOI.get(model, MOI.TimeLimitSec()) === nothing &&
       algorithm.solution_limit === nothing
        error("At least `MOI.TimeLimitSec` or `MOI.SolutionLimit` must be set")
    end
    start_time = time()
    solutions = SolutionPoint[]
    sense = MOI.get(model, MOI.ObjectiveSense())
    variables = MOI.get(model.inner, MOI.ListOfVariableIndices())
    limit = something(MOI.get(model, MOI.SolutionLimit()), typemax(Int))
    status = MOI.OPTIMAL
    while length(solutions) < limit
        if (ret = _check_premature_termination(model, start_time)) !== nothing
            status = ret
            break
        end
        weights = rand(MOI.output_dimension(model.f))
        f = _scalarise(model.f, weights)
        MOI.set(model.inner, MOI.ObjectiveFunction{typeof(f)}(), f)
        optimize_inner!(model)
        if _is_scalar_status_optimal(model)
            X, Y = _compute_point(model, variables, model.f)
            push!(solutions, SolutionPoint(X, Y))
        end
        if length(solutions) == limit
            solutions = filter_nondominated(sense, solutions)
        end
    end
    return status, filter_nondominated(sense, solutions)
end
