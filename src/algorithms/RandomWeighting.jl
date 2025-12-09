#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

"""
    RandomWeighting()

A heuristic solver that works by repeatedly solving a weighted sum problem with
random weights.

## Supported problem classes

This algorithm supports all problem classes.

## Supported optimizer attributes

 * `MOI.TimeLimitSec()`: terminate if the time limit is exceeded and return the
   list of current solutions.

 * `MOA.SolutionLimit()`: terminate once this many solutions have been found.

At least one of these two limits must be set.
"""
mutable struct RandomWeighting <: AbstractAlgorithm
    solution_limit::Union{Nothing,Int}
    RandomWeighting() = new(nothing)
end

MOI.supports(::RandomWeighting, ::SolutionLimit) = true

function MOI.set(alg::RandomWeighting, ::SolutionLimit, value)
    alg.solution_limit = value
    return
end

function MOI.get(alg::RandomWeighting, attr::SolutionLimit)
    return something(alg.solution_limit, _default(alg, attr))
end

function optimize_multiobjective!(algorithm::RandomWeighting, model::Optimizer)
    if MOI.get(model, MOI.TimeLimitSec()) === nothing &&
       algorithm.solution_limit === nothing
        error("At least `MOI.TimeLimitSec` or `MOA.SolutionLimit` must be set")
    end
    start_time = time()
    solutions = SolutionPoint[]
    sense = MOI.get(model, MOI.ObjectiveSense())
    P = MOI.output_dimension(model.f)
    variables = MOI.get(model.inner, MOI.ListOfVariableIndices())
    f = _scalarise(model.f, ones(P))
    MOI.set(model.inner, MOI.ObjectiveFunction{typeof(f)}(), f)
    optimize_inner!(model)
    status = MOI.get(model.inner, MOI.TerminationStatus())
    if _is_scalar_status_optimal(status)
        X, Y = _compute_point(model, variables, model.f)
        push!(solutions, SolutionPoint(X, Y))
    else
        return status, nothing
    end
    # This double loop is a bit weird:
    #   * the inner loop fills up SolutionLimit number of solutions. Then we cut
    #     it back to nondominated.
    #   * then the outer loop goes again
    while length(solutions) < MOI.get(algorithm, SolutionLimit())
        while length(solutions) < MOI.get(algorithm, SolutionLimit())
            ret = _check_premature_termination(model, start_time)
            if ret !== nothing
                return ret, filter_nondominated(sense, solutions)
            end
            weights = rand(P)
            f = _scalarise(model.f, weights)
            MOI.set(model.inner, MOI.ObjectiveFunction{typeof(f)}(), f)
            optimize_inner!(model)
            status = MOI.get(model.inner, MOI.TerminationStatus())
            if _is_scalar_status_optimal(status)
                X, Y = _compute_point(model, variables, model.f)
                push!(solutions, SolutionPoint(X, Y))
            end
        end
        solutions = filter_nondominated(sense, solutions)
    end
    return MOI.OPTIMAL, filter_nondominated(sense, solutions)
end
