#  Copyright 2019, Oscar Dowson. This Source Code Form is subject to the terms
#  of the Mozilla Public License, v.2.0. If a copy of the MPL was not
#  distributed with this file, You can obtain one at
#  http://mozilla.org/MPL/2.0/.

module MOO

import MathOptInterface

const MOI = MathOptInterface

struct ParetoSolution
    x::Dict{MOI.VariableIndex,Float64}
    y::Vector{Float64}
end

function Base.isapprox(a::ParetoSolution, b::ParetoSolution; kwargs...)
    return isapprox(a.y, b.y; kwargs...)
end

function _scalarise(f::MOI.VectorOfVariables, w::Vector{Float64})
    @assert MOI.output_dimension(f) == length(w)
    return MOI.ScalarAffineFunction(
        [MOI.ScalarAffineTerm(w[i], f.variables[i]) for i in 1:length(w)],
        0.0,
    )
end

function _scalarise(f::MOI.VectorAffineFunction, w::Vector{Float64})
    @assert MOI.output_dimension(f) == length(w)
    constant = sum(w[i] * f.constants[i] for i in 1:length(w))
    terms = MOI.ScalarAffineTerm{Float64}[
        MOI.ScalarAffineTerm(
            w[term.output_index] * term.scalar_term.coefficient,
            term.scalar_term.variable,
        ) for term in f.terms
    ]
    return MOI.ScalarAffineFunction(terms, constant)
end

abstract type AbstractAlgorithm end

mutable struct Optimizer <: MOI.AbstractOptimizer
    inner::MOI.AbstractOptimizer
    algorithm::AbstractAlgorithm
    f::Union{Nothing,MOI.AbstractVectorFunction}
    solutions::Union{Nothing,Vector{ParetoSolution}}
    termination_status::MOI.TerminationStatusCode

    function Optimizer(
        optimizer::MOI.AbstractOptimizer,
        algorithm::AbstractAlgorithm,
    )
        return new(
            optimizer,
            algorithm,
            nothing,
            nothing,
            MOI.OPTIMIZE_NOT_CALLED,
        )
    end
end

function MOI.empty!(model::Optimizer)
    MOI.empty!(model.inner)
    model.f = nothing
    model.solutions = nothing
    model.termination_status = MOI.OPTIMIZE_NOT_CALLED
    return
end

function MOI.is_empty(model::Optimizer)
    return MOI.is_empty(model) &&
           model.f === nothing &&
           model.solutions === nothing &&
           model.termination_status == MOI.OPTIMIZE_NOT_CALLED
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{<:MOI.AbstractVectorFunction},
)
    return true
end

const _ATTRIBUTES = Union{
    MOI.AbstractConstraintAttribute,
    MOI.AbstractModelAttribute,
    MOI.AbstractOptimizerAttribute,
    MOI.AbstractVariableAttribute,
}

function MOI.set(model::Optimizer, attr::MOI.RawOptimizerAttribute, value)
    if MOI.supports(model.algorithm, attr)
        MOI.set(model.algorithm, attr, value)
    else
        MOI.set(model.inner, attr, value)
    end
    return
end

function MOI.get(model::Optimizer, attr::MOI.RawOptimizerAttribute)
    if MOI.supports(model.algorithm, attr)
        return MOI.get(model.algorithm, attr)
    else
        return MOI.get(model.inner, attr)
    end
end

MOI.supports(model::Optimizer, args...) = MOI.supports(model.inner, args...)

function MOI.set(model::Optimizer, attr::_ATTRIBUTES, args...)
    return MOI.set(model.inner, attr, args...)
end

function MOI.get(model::Optimizer, attr::_ATTRIBUTES, args...)
    return MOI.get(model.inner, attr, args...)
end

function MOI.get(model::Optimizer, ::Type{MOI.VariableIndex}, args...)
    return MOI.get(model.inner, MOI.VariableIndex, args...)
end

function MOI.get(model::Optimizer, T::Type{<:MOI.ConstraintIndex}, args...)
    return MOI.get(model.inner, T, args...)
end

MOI.add_variable(model::Optimizer) = MOI.add_variable(model.inner)

MOI.add_variables(model::Optimizer, n::Int) = MOI.add_variables(model.inner, n)

function MOI.supports_constraint(
    model::Optimizer,
    F::Type{<:MOI.AbstractFunction},
    S::Type{<:MOI.AbstractSet},
)
    return MOI.supports_constraint(model.inner, F, S)
end

function MOI.add_constraint(
    model::Optimizer,
    f::MOI.AbstractFunction,
    s::MOI.AbstractSet,
)
    return MOI.add_constraint(model.inner, f, s)
end

function MOI.set(
    model::Optimizer,
    ::MOI.ObjectiveFunction{F},
    f::F,
) where {F<:MOI.AbstractVectorFunction}
    if MOI.output_dimension(f) != 2
        error("Only bi-objective problems supported.")
    end
    model.f = f
    return
end

MOI.get(model::Optimizer, ::MOI.ObjectiveFunctionType) = typeof(model.f)

MOI.get(model::Optimizer, ::MOI.ObjectiveFunction) = model.f

function MOI.optimize!(model::Optimizer)
    model.solutions = nothing
    model.termination_status = MOI.OPTIMIZE_NOT_CALLED
    status, solutions = optimize_multiobjective!(model.algorithm, model)
    model.termination_status = status
    model.solutions = solutions
    return
end

MOI.get(model::Optimizer, ::MOI.ResultCount) = length(model.solutions)

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    x::MOI.VariableIndex,
)
    sol = model.solutions[attr.result_index]
    return sol.x[x]
end

function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    return model.solutions[attr.result_index].y
end

function MOI.get(model::Optimizer, attr::MOI.ObjectiveBound)
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

MOI.get(model::Optimizer, ::MOI.TerminationStatus) = model.termination_status

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    if 1 <= attr.result_index <= length(model.solutions)
        return MOI.FEASIBLE_POINT
    end
    return MOI.NO_SOLUTION
end

MOI.get(::Optimizer, ::MOI.DualStatus) = MOI.NO_SOLUTION

for file in readdir(joinpath(@__DIR__, "algorithms"))
    include(joinpath(@__DIR__, "algorithms", file))
end

end
