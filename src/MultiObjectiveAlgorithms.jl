#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

module MultiObjectiveAlgorithms

import Combinatorics
import MathOptInterface as MOI

struct SolutionPoint
    x::Dict{MOI.VariableIndex,Float64}
    y::Vector{Float64}
end

function Base.isapprox(a::SolutionPoint, b::SolutionPoint; kwargs...)
    return isapprox(a.y, b.y; kwargs...)
end

Base.:(==)(a::SolutionPoint, b::SolutionPoint) = a.y == b.y

"""
    dominates(sense, a::SolutionPoint, b::SolutionPoint; atol::Float64)

Returns `true` if point `a` dominates point `b`.
"""
function dominates(
    sense::MOI.OptimizationSense,
    a::SolutionPoint,
    b::SolutionPoint;
    atol::Float64 = 1e-6,
)
    l, u = extrema(a.y - b.y)
    if sense == MOI.MIN_SENSE
        # At least one element must be strictly better => l < -atol
        # No element can be structly worse => u <= atol
        return l < -atol && u <= atol
    else
        # At least one element must be strictly better => u > atol
        # No element can be structly worse => l >= -atol
        return u > atol && l >= -atol
    end
end

_sort!(solutions::Vector{SolutionPoint}) = sort!(solutions; by = x -> x.y)

function filter_nondominated(
    sense,
    solutions::Vector{SolutionPoint};
    atol::Float64 = 1e-6,
)
    _sort!(solutions)
    nondominated_solutions = SolutionPoint[]
    for candidate in solutions
        if any(test -> dominates(sense, test, candidate; atol), solutions)
            # Point is dominated. Don't add
        elseif any(test -> ≈(test.y, candidate.y; atol), nondominated_solutions)
            # Point already added to nondominated solutions. Don't add
        else
            push!(nondominated_solutions, candidate)
        end
    end
    return nondominated_solutions
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

function _scalarise(f::MOI.VectorQuadraticFunction, w::Vector{Float64})
    @assert MOI.output_dimension(f) == length(w)
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[
        MOI.ScalarQuadraticTerm(
            w[term.output_index] * term.scalar_term.coefficient,
            term.scalar_term.variable_1,
            term.scalar_term.variable_2,
        ) for term in f.quadratic_terms
    ]
    affine_terms = MOI.ScalarAffineTerm{Float64}[
        MOI.ScalarAffineTerm(
            w[term.output_index] * term.scalar_term.coefficient,
            term.scalar_term.variable,
        ) for term in f.affine_terms
    ]
    constant = sum(w[i] * f.constants[i] for i in 1:length(w))
    return MOI.ScalarQuadraticFunction(quad_terms, affine_terms, constant)
end

function _scalarise(f::MOI.VectorNonlinearFunction, w::Vector{Float64})
    scalars = map(zip(w, f.rows)) do (wi, fi)
        return MOI.ScalarNonlinearFunction(:*, Any[wi, fi])
    end
    return MOI.ScalarNonlinearFunction(:+, scalars)
end

abstract type AbstractAlgorithm end

MOI.Utilities.map_indices(::Function, x::AbstractAlgorithm) = x

mutable struct Optimizer <: MOI.AbstractOptimizer
    inner::MOI.AbstractOptimizer
    algorithm::Union{Nothing,AbstractAlgorithm}
    f::Union{Nothing,MOI.AbstractVectorFunction}
    solutions::Vector{SolutionPoint}
    termination_status::MOI.TerminationStatusCode
    time_limit_sec::Union{Nothing,Float64}
    solve_time::Float64
    ideal_point::Vector{Float64}
    compute_ideal_point::Bool
    subproblem_count::Int
    optimizer_factory::Any

    function Optimizer(optimizer_factory)
        return new(
            MOI.instantiate(optimizer_factory; with_cache_type = Float64),
            nothing,
            nothing,
            SolutionPoint[],
            MOI.OPTIMIZE_NOT_CALLED,
            nothing,
            NaN,
            Float64[],
            default(ComputeIdealPoint()),
            0,
            optimizer_factory,
        )
    end
end

function MOI.empty!(model::Optimizer)
    MOI.empty!(model.inner)
    model.f = nothing
    empty!(model.solutions)
    model.termination_status = MOI.OPTIMIZE_NOT_CALLED
    model.solve_time = NaN
    empty!(model.ideal_point)
    model.subproblem_count = 0
    return
end

function MOI.is_empty(model::Optimizer)
    return MOI.is_empty(model.inner) &&
           model.f === nothing &&
           isempty(model.solutions) &&
           model.termination_status == MOI.OPTIMIZE_NOT_CALLED &&
           isnan(model.solve_time) &&
           isempty(model.ideal_point)
end

MOI.supports_incremental_interface(::Optimizer) = true

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    return MOI.Utilities.default_copy_to(dest, src)
end

### TimeLimitSec

function MOI.supports(model::Optimizer, attr::MOI.TimeLimitSec)
    return MOI.supports(model.inner, attr)
end

MOI.get(model::Optimizer, ::MOI.TimeLimitSec) = model.time_limit_sec

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, value::Real)
    model.time_limit_sec = Float64(value)
    return
end

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, ::Nothing)
    model.time_limit_sec = nothing
    return
end

### SolveTimeSec

function MOI.get(model::Optimizer, ::MOI.SolveTimeSec)
    return model.solve_time
end

### ObjectiveFunction

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{<:MOI.AbstractScalarFunction},
)
    return false
end

function MOI.supports(
    model::Optimizer,
    ::MOI.ObjectiveFunction{F},
) where {F<:MOI.AbstractVectorFunction}
    G = MOI.Utilities.scalar_type(F)
    H = MOI.Utilities.promote_operation(+, Float64, G, G)
    return MOI.supports(model.inner, MOI.ObjectiveFunction{G}()) &&
           MOI.supports(model.inner, MOI.ObjectiveFunction{H}())
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

An attribute to control the algorithm used by MOA.
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

A super-type for MOA-specific optimizer attributes.
"""
abstract type AbstractAlgorithmAttribute <: MOI.AbstractOptimizerAttribute end

default(::AbstractAlgorithm, attr::AbstractAlgorithmAttribute) = default(attr)

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

"""
    ObjectiveAbsoluteTolerance(index::Int) <: AbstractAlgorithmAttribute -> Float64

Assign a `Float64` tolerance to objective number `index`. This is most commonly
used to constrain an objective to a range in absolute terms to the optimal
objective value of that objective.

Defaults to `0.0`.
"""
struct ObjectiveAbsoluteTolerance <: AbstractAlgorithmAttribute
    index::Int
end

default(::ObjectiveAbsoluteTolerance) = 0.0

"""
    EpsilonConstraintStep <: AbstractAlgorithmAttribute -> Float64

The step `ε` to use in epsilon-constraint methods.

Defaults to `1.0`.
"""
struct EpsilonConstraintStep <: AbstractAlgorithmAttribute end

default(::EpsilonConstraintStep) = 1.0

"""
    LexicographicAllPermutations <: AbstractAlgorithmAttribute -> Bool

Controls whether to return the lexicographic solution for all permutations of
the scalar objectives (when `true`), or only the solution corresponding to the
lexicographic solution of the original objective function (when `false`).

Defaults to `true`.
"""
struct LexicographicAllPermutations <: AbstractAlgorithmAttribute end

default(::LexicographicAllPermutations) = true

"""
    ComputeIdealPoint <: AbstractOptimizerAttribute -> Bool

Controls whether to compute the ideal point.

Defaults to true`.

If this attribute is set to `true`, the ideal point can be queried using the
`MOI.ObjectiveBound` attribute.

Computing the ideal point requires as many solves as the dimension of the
objective function. Thus, if you do not need the ideal point information, you
can improve the performance of MOA by setting this attribute to `false`.
"""
struct ComputeIdealPoint <: MOI.AbstractOptimizerAttribute end

default(::ComputeIdealPoint) = true

MOI.supports(::Optimizer, ::ComputeIdealPoint) = true

function MOI.set(model::Optimizer, ::ComputeIdealPoint, value::Bool)
    model.compute_ideal_point = value
    return
end

MOI.get(model::Optimizer, ::ComputeIdealPoint) = model.compute_ideal_point

### SubproblemCount

"""
    SubproblemCount <: AbstractModelAttribute -> Int

A result attribute for querying the total number of subproblem solves by an
algorithm.
"""
struct SubproblemCount <: MOI.AbstractModelAttribute end

MOI.is_set_by_optimize(::SubproblemCount) = true

MOI.get(model::Optimizer, ::SubproblemCount) = model.subproblem_count

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
    return "MOA[algorithm=$alg, optimizer=$inner]"
end

### AbstractModelAttribute

function MOI.supports(model::Optimizer, arg::MOI.AbstractModelAttribute)
    return MOI.supports(model.inner, arg)
end

### AbstractVariableAttribute

function MOI.is_valid(model::Optimizer, x::MOI.VariableIndex)
    return MOI.is_valid(model.inner, x)
end

function MOI.supports(
    model::Optimizer,
    arg::MOI.AbstractVariableAttribute,
    ::Type{MOI.VariableIndex},
)
    return MOI.supports(model.inner, arg, MOI.VariableIndex)
end

function MOI.set(
    model::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    indices::Vector{<:MOI.VariableIndex},
    args::Vector{T},
) where {T}
    MOI.set.(model, attr, indices, args)
    return
end

### AbstractConstraintAttribute

function MOI.is_valid(model::Optimizer, ci::MOI.ConstraintIndex)
    return MOI.is_valid(model.inner, ci)
end

function MOI.supports(
    model::Optimizer,
    arg::MOI.AbstractConstraintAttribute,
    ::Type{MOI.ConstraintIndex{F,S}},
) where {F<:MOI.AbstractFunction,S<:MOI.AbstractSet}
    return MOI.supports(model.inner, arg, MOI.ConstraintIndex{F,S})
end

function MOI.set(
    model::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    indices::Vector{<:MOI.ConstraintIndex},
    args::Vector{T},
) where {T}
    MOI.set.(model, attr, indices, args)
    return
end

function MOI.set(model::Optimizer, attr::_ATTRIBUTES, args...)
    return MOI.set(model.inner, attr, args...)
end

function MOI.get(model::Optimizer, attr::_ATTRIBUTES, args...)
    if MOI.is_set_by_optimize(attr)
        msg = "MOA does not support querying this attribute."
        throw(MOI.GetAttributeNotAllowed(attr, msg))
    end
    return MOI.get(model.inner, attr, args...)
end

function MOI.get(model::Optimizer, attr::_ATTRIBUTES, arg::Vector{T}) where {T}
    if MOI.is_set_by_optimize(attr)
        msg = "MOA does not support querying this attribute."
        throw(MOI.GetAttributeNotAllowed(attr, msg))
    end
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

function MOI.get(model::Optimizer, attr::MOI.ListOfModelAttributesSet)
    ret = MOI.get(model.inner, attr)
    if model.f !== nothing
        F = MOI.get(model, MOI.ObjectiveFunctionType())
        push!(ret, MOI.ObjectiveFunction{F}())
    end
    return ret
end

function MOI.delete(model::Optimizer, x::MOI.VariableIndex)
    if model.f isa MOI.VectorNonlinearFunction
        throw(MOI.DeleteNotAllowed(x))
    end
    MOI.delete(model.inner, x)
    if model.f !== nothing
        model.f = MOI.Utilities.remove_variable(model.f, x)
        if MOI.output_dimension(model.f) == 0
            model.f = nothing
        end
    end
    return
end

function MOI.delete(model::Optimizer, ci::MOI.ConstraintIndex)
    MOI.delete(model.inner, ci)
    return
end

"""
    optimize_inner!(model::Optimizer)

A function that must be called instead of `MOI.optimize!(model.inner)` because
it also increments the `subproblem_count`.
"""
function optimize_inner!(model::Optimizer)
    MOI.optimize!(model.inner)
    model.subproblem_count += 1
    return
end

function _compute_ideal_point(model::Optimizer, start_time)
    for (i, f) in enumerate(MOI.Utilities.eachscalar(model.f))
        if _check_premature_termination(model, start_time) !== nothing
            return
        end
        if !isnan(model.ideal_point[i])
            continue  # The algorithm already updated this information
        end
        MOI.set(model.inner, MOI.ObjectiveFunction{typeof(f)}(), f)
        optimize_inner!(model)
        status = MOI.get(model.inner, MOI.TerminationStatus())
        if _is_scalar_status_optimal(status)
            model.ideal_point[i] = MOI.get(model.inner, MOI.ObjectiveValue())
        end
    end
    return
end

function optimize_multiobjective!(
    algorithm::AbstractAlgorithm,
    model::Optimizer,
)
    sense = MOI.get(model.inner, MOI.ObjectiveSense())
    if sense == MOI.FEASIBILITY_SENSE
        return MOI.INVALID_MODEL, nothing
    elseif sense == MOI.MAX_SENSE
        old_obj = copy(model.f)
        neg_obj = MOI.Utilities.operate(-, Float64, model.f)
        MOI.set(model, MOI.ObjectiveFunction{typeof(neg_obj)}(), neg_obj)
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        status, solutions = minimize_multiobjective!(algorithm, model)
        MOI.set(model, MOI.ObjectiveFunction{typeof(old_obj)}(), old_obj)
        MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
        if solutions !== nothing
            solutions = [SolutionPoint(s.x, -s.y) for s in solutions]
        end
        model.ideal_point *= -1
        return status, solutions
    end
    return minimize_multiobjective!(algorithm, model)
end

function _check_interrupt(f)
    try
        return reenable_sigint(f)
    catch ex
        if !(ex isa InterruptException)
            rethrow(ex)
        end
        return MOI.INTERRUPTED
    end
end

function _check_premature_termination(model::Optimizer, start_time::Float64)
    return _check_interrupt() do
        time_limit = MOI.get(model, MOI.TimeLimitSec())
        if time_limit !== nothing
            time_remaining = time_limit - (time() - start_time)
            if time_remaining <= 0
                return MOI.TIME_LIMIT
            end
            if MOI.supports(model.inner, MOI.TimeLimitSec())
                MOI.set(model.inner, MOI.TimeLimitSec(), time_remaining)
            end
        end
        return
    end
end

function MOI.optimize!(model::Optimizer)
    disable_sigint(() -> _optimize!(model))
    return
end

function _optimize!(model::Optimizer)
    start_time = time()
    empty!(model.solutions)
    model.termination_status = MOI.OPTIMIZE_NOT_CALLED
    model.subproblem_count = 0
    if model.f === nothing
        model.termination_status = MOI.INVALID_MODEL
        empty!(model.ideal_point)
        return
    end
    # We need to clear the ideal point prior to starting the solve. Algorithms
    # may update this during the solve, otherwise we will update it at the end.
    model.ideal_point = fill(NaN, MOI.output_dimension(model.f))
    algorithm = something(model.algorithm, default(Algorithm()))
    status, solutions = optimize_multiobjective!(algorithm, model)
    model.termination_status = status
    if solutions !== nothing
        model.solutions = solutions
        _sort!(model.solutions)
    end
    if MOI.get(model, ComputeIdealPoint())
        _compute_ideal_point(model, start_time)
    end
    if MOI.supports(model.inner, MOI.TimeLimitSec())
        MOI.set(model.inner, MOI.TimeLimitSec(), nothing)
    end
    model.solve_time = time() - start_time
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

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    x::Vector{MOI.VariableIndex},
)
    return MOI.get.(model, attr, x)
end

function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    return model.solutions[attr.result_index].y
end

MOI.get(model::Optimizer, ::MOI.ObjectiveBound) = model.ideal_point

MOI.get(model::Optimizer, ::MOI.TerminationStatus) = model.termination_status

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    if 1 <= attr.result_index <= length(model.solutions)
        return MOI.FEASIBLE_POINT
    end
    return MOI.NO_SOLUTION
end

MOI.get(::Optimizer, ::MOI.DualStatus) = MOI.NO_SOLUTION

function _compute_point(
    model::Optimizer,
    variables::Vector{MOI.VariableIndex},
    f,
)
    X = Dict{MOI.VariableIndex,Float64}(
        x => MOI.get(model.inner, MOI.VariablePrimal(), x) for x in variables
    )
    Y = MOI.Utilities.eval_variables(Base.Fix1(getindex, X), model, f)
    return X, Y
end

function _is_scalar_status_feasible_point(status::MOI.ResultStatusCode)
    return status == MOI.FEASIBLE_POINT
end

function _is_scalar_status_optimal(status::MOI.TerminationStatusCode)
    return status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
end

function _is_scalar_status_optimal(model::Optimizer)
    status = MOI.get(model.inner, MOI.TerminationStatus())
    return _is_scalar_status_optimal(status)
end

function _warn_on_nonfinite_anti_ideal(algorithm, sense, index)
    alg = string(typeof(algorithm))
    direction = sense == MOI.MIN_SENSE ? "above" : "below"
    bound = sense == MOI.MIN_SENSE ? "upper" : "lower"
    @warn(
        "Unable to solve the model using the `$alg` algorithm because the " *
        "anti-ideal point of objective $index is not bounded $direction, and the " *
        "algorithm requires a finitely bounded objective domain. The easiest " *
        "way to fix this is to add objective $index as a constraint with a " *
        "finite $bound. Alteratively, ensure that all of your decision " *
        "variables have finite lower and upper bounds."
    )
    return
end

function _project(x::Vector{Float64}, axis::Int)
    return [x[i] for i in 1:length(x) if i != axis]
end

for file in readdir(joinpath(@__DIR__, "algorithms"))
    # The check for .jl is necessary because some users may have other files
    # like .cov from running code coverage. See JuMP.jl#3746.
    if endswith(file, ".jl")
        include(joinpath(@__DIR__, "algorithms", file))
    end
end

end
