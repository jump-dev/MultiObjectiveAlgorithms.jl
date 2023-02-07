#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

"""
    Lexicographic()

`Lexicographic` implements a lexigographic algorithm that returns a single point
on the frontier, corresponding to solving each objective in order.

## Supported optimizer attributes

 * `MOA.ObjectiveRelativeTolerance`
"""
mutable struct Lexicographic <: AbstractAlgorithm
    rtol::Vector{Float64}

    Lexicographic() = new(Float64[])
end

MOI.supports(::Lexicographic, ::ObjectiveRelativeTolerance) = true

function MOI.get(alg::Lexicographic, attr::ObjectiveRelativeTolerance)
    return get(alg.rtol, attr.index, default(alg, attr))
end

function MOI.set(alg::Lexicographic, attr::ObjectiveRelativeTolerance, value)
    for _ in (1+length(alg.rtol)):attr.index
        push!(alg.rtol, default(alg, attr))
    end
    alg.rtol[attr.index] = value
    return
end

function optimize_multiobjective!(algorithm::Lexicographic, model::Optimizer)
    variables = MOI.get(model.inner, MOI.ListOfVariableIndices())
    constraints = Any[]
    for (i, f) in enumerate(MOI.Utilities.eachscalar(model.f))
        MOI.set(model.inner, MOI.ObjectiveFunction{typeof(f)}(), f)
        MOI.optimize!(model.inner)
        if MOI.get(model.inner, MOI.TerminationStatus()) != MOI.OPTIMAL
            return MOI.OTHER_ERROR, nothing
        end
        # Add tolerance constraints
        X = Dict{MOI.VariableIndex,Float64}(
            x => MOI.get(model.inner, MOI.VariablePrimal(), x) for
            x in variables
        )
        Y = MOI.Utilities.eval_variables(x -> X[x], f)
        sense = MOI.get(model.inner, MOI.ObjectiveSense())
        rtol = MOI.get(algorithm, ObjectiveRelativeTolerance(i))
        set = if sense == MOI.MIN_SENSE
            MOI.LessThan(Y + rtol * abs(Y))
        else
            MOI.GreaterThan(Y - rtol * abs(Y))
        end
        push!(constraints, MOI.add_constraint(model, f, set))
    end
    X = Dict{MOI.VariableIndex,Float64}(
        x => MOI.get(model.inner, MOI.VariablePrimal(), x) for x in variables
    )
    Y = MOI.Utilities.eval_variables(x -> X[x], model.f)
    # Remove tolerance constraints
    for c in constraints
        MOI.delete(model, c)
    end
    return MOI.OPTIMAL, [SolutionPoint(X, Y)]
end
