#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

"""
    TambyVanderpooten()

`TambyVanderpooten` implements the algorithm of:

Satya Tamby, Daniel Vanderpooten (2021) Enumeration of the Nondominated Set
of Multiobjective Discrete Optimization Problems. INFORMS Journal on
Computing 33(1):72-85.

This is an algorithm to generate all nondominated solutions for multi-objective
discrete optimization problems. The algorithm maintains upper bounds (for
minimization problems) and their associated defining points. At each iteration,
one of the objectives and an upper bound is picked and the single objective
reformulation is solved using one of the defining points as a starting solution.

## Supported problem classes

This algorithm is restricted to problems with:

 * discrete variables only. It will fail to converge if the problem is purely
   continuous.

## Supported optimizer attributes

 * `MOI.TimeLimitSec()`: terminate if the time limit is exceeded and return the
    list of current solutions.
"""
struct TambyVanderpooten <: AbstractAlgorithm end

# This is Algorithm 1 from the paper.
function _update_search_region(
    U_N::Dict{Vector{Float64},Vector{Vector{Vector{Float64}}}},
    y::Vector{Float64},
    yN::Vector{Float64},
)
    p = length(y)
    bounds_to_remove = Vector{Float64}[]
    bounds_to_add = Dict{Vector{Float64},Vector{Vector{Vector{Float64}}}}()
    for (u, u_n) in U_N
        if _is_less(y, u, 0) # k=0 here because we want to check every element
            push!(bounds_to_remove, u)
            for l in 1:p
                u_l = _get_child(u, y, l)
                N = [
                    k == l ? [y] : [yi for yi in u_n[k] if yi[l] < y[l]] for
                    k in 1:p
                ]
                if all(!isempty(N[k]) for k in 1:p if k != l && u_l[k] != yN[k])
                    bounds_to_add[u_l] = N
                end
            end
        else
            for k in 1:p
                if _is_less(y, u, k)
                    push!(u_n[k], y)
                end
            end
        end
    end
    for u in bounds_to_remove
        delete!(U_N, u)
    end
    merge!(U_N, bounds_to_add)
    return
end

function _is_less(y, u, k)
    for i in 1:length(y)
        if i == k
            if !(y[i] == u[i])
                return false
            end
        else
            if !(y[i] < u[i])
                return false
            end
        end
    end
    return true
end

function _get_child(u::Vector{Float64}, y::Vector{Float64}, k::Int)
    @assert length(u) == length(y)
    child = copy(u)
    child[k] = y[k]
    return child
end

# This is function h(u, k) in the paper.
function _h(k::Int, u::Vector{Float64}, yI::Vector{Float64})
    h = 1.0
    for (i, (u_i, yI_i)) in enumerate(zip(u, yI))
        if i != k
            h *= u_i - yI_i
        end
    end
    return h
end

# This is Problem (6) in the paper. The upper bound M is our nadir point yN.
function _select_search_zone(
    U_N::Dict{Vector{Float64},Vector{Vector{Vector{Float64}}}},
    yI::Vector{Float64},
    yN::Vector{Float64},
)
    if isempty(U_N)
        return 1, yN
    end
    k_star, u_star, v_star = 1, yN, -Inf
    for k in 1:length(yI), u in keys(U_N)
        if !isapprox(u[k], yN[k]; atol = 1e-6)
            v = _h(k, u, yI)
            if v > v_star
                k_star, u_star, v_star = k, u, v
            end
        end
    end
    return k_star, u_star
end

function minimize_multiobjective!(
    algorithm::TambyVanderpooten,
    model::Optimizer,
)
    solutions = Dict{Vector{Float64},Dict{MOI.VariableIndex,Float64}}()
    status = _minimize_multiobjective!(
        algorithm,
        model,
        model.inner,
        model.f,
        solutions,
    )::MOI.TerminationStatusCode
    return status, SolutionPoint[SolutionPoint(X, Y) for (Y, X) in solutions]
end

# To ensure the type stability of `.inner` and `.f`, we use a function barrier.
#
# We also pass in `solutions` to simplify the function and to ensure that we can
# return at any point from the this inner function with a partial list of primal
# solutions.
function _minimize_multiobjective!(
    algorithm::TambyVanderpooten,
    model::Optimizer,
    inner::MOI.ModelLike,
    f::MOI.AbstractVectorFunction,
    solutions::Dict{Vector{Float64},Dict{MOI.VariableIndex,Float64}},
)
    @assert MOI.get(inner, MOI.ObjectiveSense()) == MOI.MIN_SENSE
    warm_start_supported =
        MOI.supports(inner, MOI.VariablePrimalStart(), MOI.VariableIndex)
    variables = MOI.get(inner, MOI.ListOfVariableIndices())
    n = MOI.output_dimension(f)
    # Compute the ideal (yI) and nadir (yN) points
    yI, yN = zeros(n), zeros(n)
    scalars = MOI.Utilities.scalarize(f)
    for (i, f_i) in enumerate(scalars)
        MOI.set(inner, MOI.ObjectiveFunction{typeof(f_i)}(), f_i)
        MOI.set(inner, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        optimize_inner!(model)
        status = MOI.get(inner, MOI.TerminationStatus())
        if !_is_scalar_status_optimal(status)
            return status
        end
        _, y_i = _compute_point(model, variables, f_i)
        _log_subproblem_solve(model, variables)
        model.ideal_point[i] = yI[i] = y_i
        MOI.set(inner, MOI.ObjectiveSense(), MOI.MAX_SENSE)
        optimize_inner!(model)
        status = MOI.get(inner, MOI.TerminationStatus())
        if !_is_scalar_status_optimal(status)
            _warn_on_nonfinite_anti_ideal(algorithm, MOI.MIN_SENSE, i)
            return status
        end
        _, y_i = _compute_point(model, variables, f_i)
        _log_subproblem_solve(model, variables)
        # This is not really the nadir point, it's an upper bound on the
        # nadir point because we later add epsilon constraints with a -1.
        yN[i] = y_i + 1
    end
    MOI.set(inner, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    # Instead of adding and deleting constraints during the algorithm, we add
    # them all here at the beginning.
    ε_constraints = [
        MOI.Utilities.normalize_and_add_constraint(
            inner,
            f_i,
            MOI.LessThan(yN[i]),
        ) for (i, f_i) in enumerate(scalars)
    ]
    obj_constants = MOI.constant.(scalars)
    # U_N:
    #  keys: upper bound vectors
    #  values: a vector with n elements, U_N[u][k] is a vector of y
    U_N = Dict{Vector{Float64},Vector{Vector{Vector{Float64}}}}(
        # The nadir point, except for the ideal point in position k
        yN => [[_get_child(yN, yI, k)] for k in 1:n],
    )
    V = [Tuple{Vector{Float64},Vector{Float64}}[] for k in 1:n]
    status = MOI.OPTIMAL
    sum_f = sum(1.0 * s for s in scalars)
    while !isempty(U_N)
        if (ret = _check_premature_termination(model)) !== nothing
            status = ret
            break
        end
        k, u = _select_search_zone(U_N, yI, yN)
        # Solve problem Π¹(k, u)
        MOI.set(inner, MOI.ObjectiveFunction{typeof(scalars[k])}(), scalars[k])
        # Update the constraints y_i < u_i. Note that this is a strict
        # equality. We use an ε of 1.0. This is also why we use yN+1 when
        # computing the nadir point.
        for i in 1:n
            u_i = ifelse(i == k, yN[i], u[i])
            set = MOI.LessThan(u_i - 1.0 - obj_constants[i])
            MOI.set(inner, MOI.ConstraintSet(), ε_constraints[i], set)
        end
        # The isapprox is another way of saying does U_N[u][k] exist
        if warm_start_supported && !isapprox(u[k], yN[k]; atol = 1e-6)
            for (x_i, start_i) in solutions[last(U_N[u][k])]
                MOI.set(inner, MOI.VariablePrimalStart(), x_i, start_i)
            end
        end
        optimize_inner!(model)
        # We don't log this first-stage subproblem.
        status = MOI.get(inner, MOI.TerminationStatus())
        if !_is_scalar_status_optimal(status)
            break
        end
        # Now solve problem Π²(k, u)
        y_k = MOI.get(inner, MOI.ObjectiveValue())::Float64
        set = MOI.LessThan(y_k - obj_constants[k])
        MOI.set(inner, MOI.ConstraintSet(), ε_constraints[k], set)
        MOI.set(inner, MOI.ObjectiveFunction{typeof(sum_f)}(), sum_f)
        optimize_inner!(model)
        status = MOI.get(inner, MOI.TerminationStatus())
        if !_is_scalar_status_optimal(status)
            break
        end
        X, Y = _compute_point(model, variables, f)
        _log_subproblem_solve(model, Y)
        push!(V[k], (u, Y))
        # We want `if !(Y in U_N[u][k])` but this tests exact equality. We
        # want an approximate comparison.
        if all(!isapprox(Y; atol = 1e-6), U_N[u][k])
            _update_search_region(U_N, Y, yN)
            solutions[Y] = X
        end
        _clean_search_region(U_N, yI, V, k)
    end
    MOI.delete.(model, ε_constraints)
    return status
end

# This function is lines 10-17 of the paper. We re-order things a bit. The outer
# loop over u′ is the same, but we break the inner `foreach k` loop up into
# separate loops so that we don't need to loop over all `V` if one of the u′ has
# reached the ideal point.
#
# TODO: this loop is a good candidate for parallelisation.
#
# TODO: we could probably also be cleverer here, and just do a partial update
# based on the most recent changes to V. Do we need to keep re-checking
# everything?
function _clean_search_region(U_N, yI, V, k)
    for u′ in keys(U_N)
        if _clean_search_region_inner(u′, U_N, yI, V, k)
            delete!(U_N, u′)
        end
    end
    return
end

function _clean_search_region_inner(u′, U_N, yI, V, k)
    for (u′_k, yI_k) in zip(u′, yI)
        if isapprox(u′_k, yI_k; atol = 1e-6)
            return true
        end
    end
    for (k, V_k) in enumerate(V)
        for (u, y_k) in V_k
            if _comparison_line_16(u′, u, y_k, k)
                return true
            end
        end
    end
    return false
end

function _comparison_line_16(u′, u, y_k, k)
    if !≈(y_k[k], u′[k]; atol = 1e-6)
        return false
    end
    for (i, (u′_i, u_i)) in enumerate(zip(u′, u))
        if i != k && !(u′_i <= u_i)
            return false
        end
    end
    return true
end
