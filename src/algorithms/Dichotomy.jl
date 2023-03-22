#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

"""
    Dichotomy()

A solver that implements the algorithm of:

Y. P. Aneja, K. P. K. Nair, (1979) Bicriteria Transportation Problem. Management
Science 25(1), 73-78.

## Supported optimizer attributes

 * `MOI.TimeLimitSec()`: terminate if the time limit is exceeded and return the
   list of current solutions.

 * `MOA.SolutionLimit()`: terminate once this many solutions have been found.
"""
mutable struct Dichotomy <: AbstractAlgorithm
    solution_limit::Union{Nothing,Int}

    Dichotomy() = new(nothing)
end

"""
    NISE()

A solver that implements the Non-Inferior Set Estimation algorithm of:

Cohon, J. L., Church, R. L., & Sheer, D. P. (1979). Generating multiobjective
trade‐offs: An algorithm for bicriterion problems. Water Resources Research,
15(5), 1001-1010.

!!! note
    This algorithm is identical to `Dichotomy()`, and it may be removed in a
    future release.

## Supported optimizer attributes

 * `MOA.SolutionLimit()`
"""
NISE() = Dichotomy()

MOI.supports(::Dichotomy, ::SolutionLimit) = true

function MOI.set(alg::Dichotomy, ::SolutionLimit, value)
    alg.solution_limit = value
    return
end

function MOI.get(alg::Dichotomy, attr::SolutionLimit)
    return something(alg.solution_limit, default(alg, attr))
end

function _solve_weighted_sum(model::Optimizer, alg::Dichotomy, weight::Float64)
    return _solve_weighted_sum(model, alg, [weight, 1 - weight])
end

function _solve_weighted_sum(
    model::Optimizer,
    ::Dichotomy,
    weights::Vector{Float64},
)
    f = _scalarise(model.f, weights)
    MOI.set(model.inner, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model.inner)
    status = MOI.get(model.inner, MOI.TerminationStatus())
    if !_is_scalar_status_optimal(status)
        return status, nothing
    end
    variables = MOI.get(model.inner, MOI.ListOfVariableIndices())
    X, Y = _compute_point(model, variables, model.f)
    return status, SolutionPoint(X, Y)
end

function optimize_multiobjective!(algorithm::Dichotomy, model::Optimizer)
    start_time = time()
    if MOI.output_dimension(model.f) > 2
        error("Only scalar or bi-objective problems supported.")
    end
    if MOI.output_dimension(model.f) == 1
        status, solution = _solve_weighted_sum(model, algorithm, [1.0])
        return status, [solution]
    end
    solutions = Dict{Float64,SolutionPoint}()
    for w in (0.0, 1.0)
        status, solution = _solve_weighted_sum(model, algorithm, w)
        if !_is_scalar_status_optimal(status)
            return status, nothing
        end
        solutions[w] = solution
    end
    queue = Tuple{Float64,Float64}[]
    if !(solutions[0.0] ≈ solutions[1.0])
        push!(queue, (0.0, 1.0))
    end
    limit = MOI.get(algorithm, SolutionLimit())
    status = MOI.OPTIMAL
    while length(queue) > 0 && length(solutions) < limit
        if _time_limit_exceeded(model, start_time)
            status = MOI.TIME_LIMIT
            break
        end
        (a, b) = popfirst!(queue)
        y_d = solutions[a].y .- solutions[b].y
        w = y_d[2] / (y_d[2] - y_d[1])
        status, solution = _solve_weighted_sum(model, algorithm, w)
        if !_is_scalar_status_optimal(status)
            # Exit the solve with some error.
            return status, nothing
        elseif solution ≈ solutions[a] || solution ≈ solutions[b]
            # We have found an existing solution. We're free to prune (a, b)
            # from the search space.
        else
            # Solution is identical to a and b, so search the domain (a, w) and
            # (w, b), and add solution as a new Pareto-optimal solution!
            push!(queue, (a, w))
            push!(queue, (w, b))
            solutions[w] = solution
        end
    end
    solution_list =
        [solutions[w] for w in sort(collect(keys(solutions)); rev = true)]
    return status, solution_list
end
