#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

"""
    NISE()

A solver that implements the Non-Inferior Set Estimation algorithm of:

Cohon, J. L., Church, R. L., & Sheer, D. P. (1979). Generating multiobjective
trade‐offs: An algorithm for bicriterion problems. Water Resources Research,
15(5), 1001-1010.

## Supported optimizer attributes

 * `MOA.SolutionLimit()`
"""
mutable struct NISE <: AbstractAlgorithm
    solution_limit::Union{Nothing,Int}

    NISE() = new(nothing)
end

MOI.supports(::NISE, ::SolutionLimit) = true

function MOI.set(alg::NISE, ::SolutionLimit, value)
    alg.solution_limit = value
    return
end

function MOI.get(alg::NISE, attr::SolutionLimit)
    return something(alg.solution_limit, default(alg, attr))
end

function _solve_weighted_sum(model::Optimizer, alg::NISE, weight::Float64)
    return _solve_weighted_sum(model, alg, [weight, 1 - weight])
end

function _solve_weighted_sum(model::Optimizer, ::NISE, weights::Vector{Float64})
    f = _scalarise(model.f, weights)
    MOI.set(model.inner, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model.inner)
    status = MOI.get(model.inner, MOI.TerminationStatus())
    if status != MOI.OPTIMAL
        return status, nothing
    end
    X = Dict{MOI.VariableIndex,Float64}(
        x => MOI.get(model.inner, MOI.VariablePrimal(), x) for
        x in MOI.get(model.inner, MOI.ListOfVariableIndices())
    )
    Y = MOI.Utilities.eval_variables(x -> X[x], model.f)
    return status, SolutionPoint(X, Y)
end

function optimize_multiobjective!(algorithm::NISE, model::Optimizer)
    if MOI.output_dimension(model.f) > 2
        error("Only scalar or bi-objective problems supported.")
    end
    if MOI.output_dimension(model.f) == 1
        status, solution = _solve_weighted_sum(model, algorithm, [1.0])
        return MOI.OPTIMAL, [solution]
    end
    solutions = Dict{Float64,SolutionPoint}()
    for w in (0.0, 1.0)
        status, solution = _solve_weighted_sum(model, algorithm, w)
        if status != MOI.OPTIMAL
            return MOI.OTHER_ERROR, nothing
        end
        solutions[w] = solution
    end
    queue = Tuple{Float64,Float64}[]
    if !(solutions[0.0] ≈ solutions[1.0])
        push!(queue, (0.0, 1.0))
    end
    limit = MOI.get(algorithm, SolutionLimit())
    while length(queue) > 0 && length(solutions) < limit
        (a, b) = popfirst!(queue)
        y_d = solutions[a].y .- solutions[b].y
        w = y_d[2] / (y_d[2] - y_d[1])
        status, solution = _solve_weighted_sum(model, algorithm, w)
        if status != MOI.OPTIMAL
            # Exit the solve with some error.
            return MOI.OTHER_ERROR, nothing
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
    return MOI.OPTIMAL, solution_list
end
