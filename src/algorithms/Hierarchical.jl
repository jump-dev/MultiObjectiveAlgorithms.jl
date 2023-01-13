#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

"""
    Hierarchical()

 * `ObjectivePriority`
 * `ObjectiveWeight`
 * `ObjectiveRelativeTolerance`
"""
mutable struct Hierarchical <: AbstractAlgorithm
    priorities::Vector{Int}
    weights::Vector{Float64}
    rtol::Vector{Float64}

    Hierarchical() = new(Int[], Float64[], Float64[])
end

function MOI.empty!(alg::Hierarchical)
    # @show alg
    # empty!(alg.priorities)
    # empty!(alg.weights)
    # empty!(alg.rtol)
    return
end

function _append_default(x, N, default)
    for _ in (1+length(x)):N
        push!(x, default)
    end
    return
end

abstract type _AbstractObjectiveAttribute <: AbstractAlgorithmAttribute end

MOI.supports(::Hierarchical, attr::_AbstractObjectiveAttribute) = true

function MOI.get(alg::Hierarchical, attr::_AbstractObjectiveAttribute)
    return get(_vector(alg, attr), attr.index, _default(attr))
end

function MOI.set(alg::Hierarchical, attr::_AbstractObjectiveAttribute, value)
    data = _vector(alg, attr)
    _append_default(data, attr.index, _default(attr))
    data[attr.index] = value
    return
end

struct ObjectivePriority <: _AbstractObjectiveAttribute
    index::Int
end

_default(::ObjectivePriority) = 0
_vector(alg::Hierarchical, ::ObjectivePriority) = alg.priorities

struct ObjectiveWeight <: _AbstractObjectiveAttribute
    index::Int
end

_default(::ObjectiveWeight) = 1.0
_vector(alg::Hierarchical, ::ObjectiveWeight) = alg.weights

struct ObjectiveRelativeTolerance <: _AbstractObjectiveAttribute
    index::Int
end

_default(::ObjectiveRelativeTolerance) = 0.01
_vector(alg::Hierarchical, ::ObjectiveRelativeTolerance) = alg.rtol

function _sorted_priorities(priorities::Vector{Int})
    unique_priorities = sort(unique(priorities); rev = true)
    return [findall(isequal(u), priorities) for u in unique_priorities]
end

function optimize_multiobjective!(algorithm::Hierarchical, model::Optimizer)
    objectives = MOI.Utilities.eachscalar(model.f)
    N = length(objectives)
    variables = MOI.get(model.inner, MOI.ListOfVariableIndices())
    # Find list of objectives with same priority
    constraints = Any[]
    objective_subsets = _sorted_priorities(algorithm.priorities)
    for (round, indices) in enumerate(objective_subsets)
        # Solve weighted sum
        new_vector_f = objectives[indices]
        new_f = _scalarise(new_vector_f, algorithm.weights[indices])
        MOI.set(model.inner, MOI.ObjectiveFunction{typeof(new_f)}(), new_f)
        MOI.optimize!(model.inner)
        if MOI.get(model.inner, MOI.TerminationStatus()) != MOI.OPTIMAL
            return MOI.OTHER_ERROR, nothing
        end
        if round == length(objective_subsets)
            break
        end
        # Add tolerance constraints
        X = Dict{MOI.VariableIndex,Float64}(
            x => MOI.get(model.inner, MOI.VariablePrimal(), x) for
            x in variables
        )
        Y = MOI.Utilities.eval_variables(x -> X[x], new_vector_f)
        sense = MOI.get(model.inner, MOI.ObjectiveSense())
        for (i, fi) in enumerate(MOI.Utilities.eachscalar(new_vector_f))
            rtol = MOI.get(algorithm, ObjectiveRelativeTolerance(i))
            set = if sense == MOI.MIN_SENSE
                MOI.LessThan(Y[i] * (1 + rtol))
            else
                MOI.GreaterThan(Y[i] * (1 - rtol))
            end
            push!(constraints, MOI.add_constraint(model, fi, set))
        end
    end
    X = Dict{MOI.VariableIndex,Float64}(
        x => MOI.get(model.inner, MOI.VariablePrimal(), x) for x in variables
    )
    Y = MOI.Utilities.eval_variables(x -> X[x], model.f)
    # Remove tolerance constraints
    for c in constraints
        MOI.delete(model, c)
    end
    return MOI.OPTIMAL, [ParetoSolution(X, Y)]
end
