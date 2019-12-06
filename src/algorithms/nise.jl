#  Copyright 2019, Oscar Dowson. This Source Code Form is subject to the terms
#  of the Mozilla Public License, v.2.0. If a copy of the MPL was not
#  distributed with this file, You can obtain one at
#  http://mozilla.org/MPL/2.0/.

"""
    NISE(optimizer)

A solver that implements the Non-Inferior Set Estimation algorithm of:

Cohon, J. L., Church, R. L., & Sheer, D. P. (1979). Generating multiobjective
trade‐offs: An algorithm for bicriterion problems. Water Resources Research,
15(5), 1001-1010.

Example:

    MOO.NISE(GLPK.Optimizer())
"""
mutable struct NISE <: MOI.AbstractOptimizer
    inner::MOI.AbstractOptimizer
    f::Union{Nothing, MOI.AbstractVectorFunction}
    solutions::Union{Nothing, Vector{ParetoSolution}}
    termination_status::MOI.TerminationStatusCode
    function NISE(optimizer)
        return new(optimizer, nothing, nothing, MOI.OPTIMIZE_NOT_CALLED)
    end
end

function MOI.supports(
    ::NISE,
    ::MOI.ObjectiveFunction{<:MOI.AbstractVectorFunction}
)
    return true
end

const _ATTRIBUTES = Union{
    MOI.AbstractConstraintAttribute,
    MOI.AbstractModelAttribute,
    MOI.AbstractOptimizerAttribute,
    MOI.AbstractVariableAttribute
}

MOI.supports(model::NISE, args...) = MOI.supports(model.inner, args...)

function MOI.set(model::NISE, attr::_ATTRIBUTES, args...)
    return MOI.set(model.inner, attr, args...)
end
function MOI.get(model::NISE, attr::_ATTRIBUTES, args...)
    return MOI.get(model.inner, attr, args...)
end

MOI.add_variable(model::NISE) = MOI.add_variable(model.inner)
MOI.add_variables(model::NISE, n::Int) = MOI.add_variables(model.inner, n)

function MOI.supports_constraint(model::NISE, args...)
    return MOI.supports_constraint(model.inner, args...)
end
function MOI.add_constraint(
    model::NISE, f::MOI.AbstractFunction, s::MOI.AbstractSet
)
    return MOI.add_constraint(model.inner, f, s)
end

MOI.get(model::NISE, ::MOI.ObjectiveFunctionType) = typeof(model.f)
function MOI.get(model::NISE, ::MOI.ObjectiveFunction)
    return model.f
end

function MOI.set(
    model::NISE, ::MOI.ObjectiveFunction{F}, f::F
) where {F <: MOI.AbstractVectorFunction}
    if MOI.output_dimension(f) != 2
        error("Only bi-objective problems supported.")
    end
    return model.f = f
end

MOI.get(model::NISE, ::MOI.ResultCount) = length(model.solutions)

function MOI.get(model::NISE, attr::MOI.VariablePrimal, x::MOI.VariableIndex)
    sol = model.solutions[attr.N]
    return sol.x[x]
end

function MOI.get(model::NISE, attr::MOI.ObjectiveValue)
    return model.solutions[attr.result_index].y
end

function MOI.get(model::NISE, attr::MOI.ObjectiveBound)
    bound = zeros(length(model.solutions[1].y))
    sense = MOI.get(model, MOI.ObjectiveSense())
    for i in 1:length(bound)
        if sense == MOI.MIN_SENSE
            bound[i] = minimum([sol.y[i] for sol in model.solutions])
        else
            bound[i] = maximum([sol.y[i] for sol in model.solutions])
        end
    end
    return bound
end

function MOI.get(model::NISE, ::MOI.TerminationStatus)
    return model.termination_status
end

function MOI.get(model::NISE, attr::MOI.PrimalStatus)
    if 1 <= attr.N <= length(model.solutions)
        return MOI.FEASIBLE_POINT
    end
    return MOI.NO_SOLUTION
end

MOI.get(::NISE, ::MOI.DualStatus) = MOI.NO_SOLUTION

function _solve_weighted_sum(model::NISE, weight::Float64)
    f = _scalarise(model.f, [weight, 1 - weight])
    MOI.set(model.inner, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model.inner)
    status = MOI.get(model.inner, MOI.TerminationStatus())
    X = Dict{MOI.VariableIndex, Float64}(
        x => MOI.get(model.inner, MOI.VariablePrimal(), x)
        for x in MOI.get(model.inner, MOI.ListOfVariableIndices())
    )
    Y = MOI.Utilities.eval_variables(x -> X[x], model.f)
    return status, ParetoSolution(X, Y)
end

function MOI.optimize!(model::NISE)
    model.solutions = nothing
    model.termination_status = MOI.OPTIMIZE_NOT_CALLED
    solutions = Dict{Float64, ParetoSolution}()
    for w in (0.0, 1.0)
        status, solution = _solve_weighted_sum(model, w)
        if status != MOI.OPTIMAL
            model.termination_status = status
            return
        end
        solutions[w] = solution
    end
    queue = Tuple{Float64, Float64}[]
    if !(solutions[0.0] ≈ solutions[1.0])
        push!(queue, (0.0, 1.0))
    end
    while length(queue) > 0
        (a, b) = pop!(queue)
        y_d = solutions[a].y .- solutions[b].y
        w = y_d[2] / (y_d[2] - y_d[1])
        status, solution = _solve_weighted_sum(model, w)
        if status != MOI.OPTIMAL
            # Exit the solve with some error.
            model.termination_status = status
            return
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
    model.termination_status = MOI.OPTIMAL
    model.solutions = [
        solutions[w] for w in sort(collect(keys(solutions)), rev=true)
    ]
    return
end
