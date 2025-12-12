#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

"""
    Hierarchical()

`Hierarchical` implements an algorithm that returns a single point via an
iterative scheme.

First, it partitions the objectives into sets according to
`MOA.ObjectivePriority`. Then, in order of decreasing priority, it formulates a
single-objective problem by scalarizing all of the objectives with the same
priority using `MOA.ObjectiveWeight`. Next, it constrains those objectives such
that they can be at most `MOA.ObjectiveRelativeTolerance` worse than optimal in
future solves. Finally, it steps to the next set of prioritized objectives.

The solution is a single point that trades off the various objectives. It does
not record the partial solutions that were found along the way.

## Supported problem classes

This algorithm supports all problem classes.

## Supported optimizer attributes

 * `MOA.ObjectivePriority`
 * `MOA.ObjectiveWeight`
 * `MOA.ObjectiveRelativeTolerance`
"""
mutable struct Hierarchical <: AbstractAlgorithm
    priorities::Vector{Int}
    weights::Vector{Float64}
    rtol::Vector{Float64}

    Hierarchical() = new(Int[], Float64[], Float64[])
end

_describe(::Hierarchical) = "Hierarchical()"

MOI.supports(::Hierarchical, ::ObjectivePriority) = true

function MOI.get(alg::Hierarchical, attr::ObjectivePriority)
    return get(alg.priorities, attr.index, _default(alg, attr))
end

function _append_default(
    alg::Hierarchical,
    attr::AbstractAlgorithmAttribute,
    x::Vector,
)
    for _ in (1+length(x)):attr.index
        push!(x, _default(alg, attr))
    end
    return
end

function MOI.set(alg::Hierarchical, attr::ObjectivePriority, value)
    _append_default(alg, attr, alg.priorities)
    alg.priorities[attr.index] = value
    return
end

MOI.supports(::Hierarchical, ::ObjectiveWeight) = true

function MOI.get(alg::Hierarchical, attr::ObjectiveWeight)
    return get(alg.weights, attr.index, _default(alg, attr))
end

function MOI.set(alg::Hierarchical, attr::ObjectiveWeight, value)
    _append_default(alg, attr, alg.weights)
    alg.weights[attr.index] = value
    return
end

MOI.supports(::Hierarchical, ::ObjectiveRelativeTolerance) = true

function MOI.get(alg::Hierarchical, attr::ObjectiveRelativeTolerance)
    return get(alg.rtol, attr.index, _default(alg, attr))
end

function MOI.set(alg::Hierarchical, attr::ObjectiveRelativeTolerance, value)
    _append_default(alg, attr, alg.rtol)
    alg.rtol[attr.index] = value
    return
end

function _sorted_priorities(priorities::Vector{Int})
    unique_priorities = sort(unique(priorities); rev = true)
    return [findall(isequal(u), priorities) for u in unique_priorities]
end

function minimize_multiobjective!(algorithm::Hierarchical, model::Optimizer)
    @assert MOI.get(model.inner, MOI.ObjectiveSense()) == MOI.MIN_SENSE
    objectives = MOI.Utilities.eachscalar(model.f)
    N = length(objectives)
    variables = MOI.get(model.inner, MOI.ListOfVariableIndices())
    # Find list of objectives with same priority
    constraints = Any[]
    priorities = [MOI.get(algorithm, ObjectivePriority(i)) for i in 1:N]
    weights = [MOI.get(algorithm, ObjectiveWeight(i)) for i in 1:N]
    objective_subsets = _sorted_priorities(priorities)
    for (round, indices) in enumerate(objective_subsets)
        # Solve weighted sum
        new_vector_f = objectives[indices]
        new_f = _scalarise(new_vector_f, weights[indices])
        MOI.set(model.inner, MOI.ObjectiveFunction{typeof(new_f)}(), new_f)
        optimize_inner!(model)
        status = MOI.get(model.inner, MOI.TerminationStatus())
        if !_is_scalar_status_optimal(status)
            return status, nothing
        end
        if round == length(objective_subsets)
            break
        end
        if !model.silent
            X, Y = _compute_point(model, variables, model.f)
            _log_solution(model, Y)
        end
        # Add tolerance constraints
        X, Y = _compute_point(model, variables, new_vector_f)
        for (i, fi) in enumerate(MOI.Utilities.eachscalar(new_vector_f))
            rtol = MOI.get(algorithm, ObjectiveRelativeTolerance(i))
            set = MOI.LessThan(Y[i] + rtol * abs(Y[i]))
            ci = MOI.Utilities.normalize_and_add_constraint(model, fi, set)
            push!(constraints, ci)
        end
    end
    X, Y = _compute_point(model, variables, model.f)
    _log_solution(model, Y)
    # Remove tolerance constraints
    for c in constraints
        MOI.delete(model, c)
    end
    return MOI.OPTIMAL, [SolutionPoint(X, Y)]
end
