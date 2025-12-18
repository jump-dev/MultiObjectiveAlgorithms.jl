#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

"""
    EpsilonConstraint()

`EpsilonConstraint` implements the epsilon-constraint algorithm for
bi-objective programs.

## Supported problem classes

This algorithm is restricted to problems with:

 * exactly two objectives

## Supported optimizer attributes

 * `MOA.EpsilonConstraintStep()`: `EpsilonConstraint` uses this value
   as the epsilon by which it partitions the first-objective's space. The
   default is `1`, so that for a pure integer program this algorithm will
   enumerate all non-dominated solutions.

 * `MOA.SolutionLimit()`: if this attribute is set then, instead of using the
   `MOA.EpsilonConstraintStep`, with a slight abuse of notation,
   `EpsilonConstraint` divides the width of the first-objective's domain in
   objective space by `SolutionLimit` to obtain the epsilon to use when
   iterating. Thus, there can be at most `SolutionLimit` solutions returned, but
   there may be fewer.
"""
mutable struct EpsilonConstraint <: AbstractAlgorithm
    solution_limit::Union{Nothing,Int}
    atol::Union{Nothing,Float64}

    EpsilonConstraint() = new(nothing, nothing)
end

MOI.supports(::EpsilonConstraint, ::SolutionLimit) = true

function MOI.set(alg::EpsilonConstraint, ::SolutionLimit, value)
    alg.solution_limit = value
    return
end

function MOI.get(alg::EpsilonConstraint, attr::SolutionLimit)
    return something(alg.solution_limit, _default(alg, attr))
end

MOI.supports(::EpsilonConstraint, ::EpsilonConstraintStep) = true

function MOI.set(alg::EpsilonConstraint, ::EpsilonConstraintStep, value)
    alg.atol = value
    return
end

function MOI.get(alg::EpsilonConstraint, attr::EpsilonConstraintStep)
    return something(alg.atol, _default(alg, attr))
end

MOI.supports(::EpsilonConstraint, ::ObjectiveAbsoluteTolerance) = true

function MOI.set(alg::EpsilonConstraint, ::ObjectiveAbsoluteTolerance, value)
    @warn("This attribute is deprecated. Use `EpsilonConstraintStep` instead.")
    MOI.set(alg, EpsilonConstraintStep(), value)
    return
end

function MOI.get(alg::EpsilonConstraint, ::ObjectiveAbsoluteTolerance)
    @warn("This attribute is deprecated. Use `EpsilonConstraintStep` instead.")
    return MOI.get(alg, EpsilonConstraintStep())
end

function minimize_multiobjective!(
    algorithm::EpsilonConstraint,
    model::Optimizer,
    inner::MOI.ModelLike,
    f::MOI.AbstractVectorFunction,
)
    @assert MOI.get(inner, MOI.ObjectiveSense()) == MOI.MIN_SENSE
    if MOI.output_dimension(f) != 2
        error("EpsilonConstraint requires exactly two objectives")
    end
    # Compute the bounding box of the objectives using Hierarchical().
    alg = Hierarchical()
    MOI.set.(Ref(alg), ObjectivePriority.(1:2), [1, 0])
    status, solution_1 = minimize_multiobjective!(alg, model, inner, f)
    if !_is_scalar_status_optimal(status)
        return status, nothing
    end
    MOI.set(alg, ObjectivePriority(2), 2)
    status, solution_2 = minimize_multiobjective!(alg, model, inner, f)
    if !_is_scalar_status_optimal(status)
        return status, nothing
    end
    a, b = solution_1[1].y[1], solution_2[1].y[1]
    left, right = min(a, b), max(a, b)
    model.ideal_point .= min.(solution_1[1].y, solution_2[1].y)
    # Compute the epsilon that we will be incrementing by each iteration
    ε = MOI.get(algorithm, EpsilonConstraintStep())
    n_points = MOI.get(algorithm, SolutionLimit())
    if n_points != _default(algorithm, SolutionLimit())
        ε = abs(right - left) / (n_points - 1)
    end
    solutions = SolutionPoint[only(solution_1), only(solution_2)]
    f1, f2 = MOI.Utilities.eachscalar(inner)
    MOI.set(inner, MOI.ObjectiveFunction{typeof(f2)}(), f2)
    # Add epsilon constraint
    variables = MOI.get(inner, MOI.ListOfVariableIndices())
    bound = right - ε
    constant = MOI.constant(f1, Float64)
    ci = MOI.Utilities.normalize_and_add_constraint(
        model,
        f1,
        MOI.LessThan{Float64}(bound);
        allow_modify_function = true,
    )
    bound -= constant
    status = MOI.OPTIMAL
    for _ in 3:n_points
        if (ret = _check_premature_termination(model)) !== nothing
            status = ret
            break
        end
        MOI.set(model, MOI.ConstraintSet(), ci, MOI.LessThan{Float64}(bound))
        optimize_inner!(model)
        if !_is_scalar_status_optimal(model)
            break
        end
        X, Y = _compute_point(model, variables, inner)
        _log_subproblem_solve(model, Y)
        if isempty(solutions) || !(Y ≈ solutions[end].y)
            push!(solutions, SolutionPoint(X, Y))
        end
        bound = min(Y[1] - constant - ε, bound - ε)
    end
    MOI.delete(model, ci)
    return status, solutions
end
