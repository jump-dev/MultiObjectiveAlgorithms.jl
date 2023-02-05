#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

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

MOI.Utilities.map_indices(::Function, x::AbstractAlgorithm) = x

mutable struct Optimizer <: MOI.AbstractOptimizer
    inner::MOI.AbstractOptimizer
    algorithm::Union{Nothing,AbstractAlgorithm}
    f::Union{Nothing,MOI.AbstractVectorFunction}
    solutions::Union{Nothing,Vector{ParetoSolution}}
    termination_status::MOI.TerminationStatusCode

    function Optimizer(optimizer_factory)
        return new(
            MOI.instantiate(optimizer_factory),
            nothing,
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
    return MOI.is_empty(model.inner) &&
           model.f === nothing &&
           model.solutions === nothing &&
           model.termination_status == MOI.OPTIMIZE_NOT_CALLED
end

MOI.supports_incremental_interface(::Optimizer) = true

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    return MOI.Utilities.default_copy_to(dest, src)
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

### Algorithm

"""
    Algorithm <: MOI.AbstractOptimizerAttribute

An attribute to control the algorithm used by MOO.
"""
struct Algorithm <: MOI.AbstractOptimizerAttribute end

MOI.supports(::Optimizer, ::Algorithm) = true

MOI.get(model::Optimizer, ::Algorithm) = model.algorithm

function MOI.set(model::Optimizer, ::Algorithm, alg::AbstractAlgorithm)
    model.algorithm = alg
    return
end

default(::Algorithm) = Lexicographic()

### AbstractAlgorithmAttribute

"""
    AbstractAlgorithmAttribute <: MOI.AbstractOptimizerAttribute

A super-type for MOO-specific optimizer attributes.
"""
abstract type AbstractAlgorithmAttribute <: MOI.AbstractOptimizerAttribute end

function MOI.supports(model::Optimizer, attr::AbstractAlgorithmAttribute)
    return MOI.supports(model.algorithm, attr)
end

function MOI.set(model::Optimizer, attr::AbstractAlgorithmAttribute, value)
    MOI.set(model.algorithm, attr, value)
    return
end

function MOI.get(model::Optimizer, attr::AbstractAlgorithmAttribute)
    return MOI.get(model.algorithm, attr)
end

"""
    SolutionLimit <: AbstractAlgorithmAttribute -> Int

Terminate the algorithm once the set number of solutions have been found.

Defaults to `typemax(Int)`.
"""
struct SolutionLimit <: AbstractAlgorithmAttribute end

default(::SolutionLimit) = typemax(Int)

"""
    ObjectivePriority(index::Int) <: AbstractAlgorithmAttribute -> Int

Assign an `Int` priority to objective number `index`. This is most commonly
used to group the objectives into sets of equal priorities. Greater numbers
indicate higher priority.

Defaults to `0`.
"""
struct ObjectivePriority <: AbstractAlgorithmAttribute
    index::Int
end

default(::ObjectivePriority) = 0

"""
    ObjectiveWeight(index::Int) <: AbstractAlgorithmAttribute -> Float64

Assign a `Float64` weight to objective number `index`. This is most commonly
used to scalarize a set of objectives using their weighted sum.

Defaults to `1.0`.
"""
struct ObjectiveWeight <: AbstractAlgorithmAttribute
    index::Int
end

default(::ObjectiveWeight) = 1.0

"""
    ObjectiveRelativeTolerance(index::Int) <: AbstractAlgorithmAttribute -> Float64

Assign a `Float64` tolerance to objective number `index`. This is most commonly
used to constrain an objective to a range relative to the optimal objective
value of that objective.

Defaults to `0.0`.
"""
struct ObjectiveRelativeTolerance <: AbstractAlgorithmAttribute
    index::Int
end

default(::ObjectiveRelativeTolerance) = 0.0

function _append_default(attr::AbstractAlgorithmAttribute, x::Vector)
    for _ in (1+length(x)):attr.index
        push!(x, default(attr))
    end
    return
end

### RawOptimizerAttribute

function MOI.supports(model::Optimizer, attr::MOI.RawOptimizerAttribute)
    return MOI.supports(model.inner, attr)
end

function MOI.set(model::Optimizer, attr::MOI.RawOptimizerAttribute, value)
    MOI.set(model.inner, attr, value)
    return
end

function MOI.get(model::Optimizer, attr::MOI.RawOptimizerAttribute)
    return MOI.get(model.inner, attr)
end

### AbstractOptimizerAttribute

function MOI.supports(model::Optimizer, arg::MOI.AbstractOptimizerAttribute)
    return MOI.supports(model.inner, arg)
end

function MOI.set(model::Optimizer, attr::MOI.AbstractOptimizerAttribute, value)
    MOI.set(model.inner, attr, value)
    return
end

function MOI.get(model::Optimizer, attr::MOI.AbstractOptimizerAttribute)
    return MOI.get(model.inner, attr)
end

function MOI.get(model::Optimizer, ::MOI.SolverName)
    alg = typeof(something(model.algorithm, default(Algorithm())))
    inner = MOI.get(model.inner, MOI.SolverName())
    return "MOO[algorithm=$alg, optimizer=$inner]"
end

### AbstractModelAttribute

function MOI.supports(model::Optimizer, arg::MOI.AbstractModelAttribute)
    return MOI.supports(model.inner, arg)
end

### AbstractVariableAttribute

function MOI.supports(
    model::Optimizer,
    arg::MOI.AbstractVariableAttribute,
    ::Type{MOI.VariableIndex},
)
    return MOI.supports(model.inner, arg, MOI.VariableIndex)
end

### AbstractConstraintAttribute

function MOI.supports(
    model::Optimizer,
    arg::MOI.AbstractConstraintAttribute,
    ::Type{MOI.ConstraintIndex{F,S}},
) where {F<:MOI.AbstractFunction,S<:MOI.AbstractSet}
    return MOI.supports(model.inner, arg, MOI.ConstraintIndex{F,S})
end

function MOI.set(model::Optimizer, attr::_ATTRIBUTES, args...)
    return MOI.set(model.inner, attr, args...)
end

function MOI.get(model::Optimizer, attr::_ATTRIBUTES, args...)
    return MOI.get(model.inner, attr, args...)
end

function MOI.get(model::Optimizer, attr::_ATTRIBUTES, arg::Vector{T}) where {T}
    return MOI.get.(model, attr, arg)
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
    model.f = f
    return
end

MOI.get(model::Optimizer, ::MOI.ObjectiveFunctionType) = typeof(model.f)

MOI.get(model::Optimizer, ::MOI.ObjectiveFunction) = model.f

MOI.delete(model::Optimizer, i::MOI.Index) = MOI.delete(model.inner, i)

function MOI.optimize!(model::Optimizer)
    model.solutions = nothing
    model.termination_status = MOI.OPTIMIZE_NOT_CALLED
    algorithm = something(model.algorithm, default(Algorithm()))
    status, solutions = optimize_multiobjective!(algorithm, model)
    model.termination_status = status
    model.solutions = solutions
    return
end

MOI.get(model::Optimizer, ::MOI.ResultCount) = length(model.solutions)

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    n = MOI.get(model, MOI.ResultCount())
    return "Solve complete. Found $n solution(s)"
end

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
    objectives = MOI.Utilities.eachscalar(model.f)
    utopia = fill(NaN, length(objectives))
    for (i, f) in enumerate(objectives)
        MOI.set(model.inner, MOI.ObjectiveFunction{typeof(f)}(), f)
        MOI.optimize!(model.inner)
        if MOI.get(model.inner, MOI.TerminationStatus()) == MOI.OPTIMAL
            utopia[i] = MOI.get(model.inner, MOI.ObjectiveValue())
        end
    end
    return utopia
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
