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

"""
    _round_objective(y::Float64, constant::Float64)::Int

Return `y - constant`, rounded to the nearest integer.

The paper assumes exact arithmetic, and the algorithm relies on exact
comparisons, like whether `y[k] == u[k]` or `y[k] < u[k]`, to decide if a point
is on the edge of a search zone or strictly inside it. Solvers have tolerances,
so the same nondominated point can be returned with different noise, for
example, as `1234.00005` and `1233.99998`. If we compare these values directly,
the algorithm can revisit the same point and cycle.

The algorithm works only for problems with integer-valued objectives (we use an
ε of `1.0` for the strict inequalities), so we round the non-constant part of
every objective value to the nearest integer and store it as an `Int`. This
ensures that all comparisons are exact.

All objective values in the algorithm (`yI`, `yN`, `U_N`, `V`, and the keys of
`solutions`) are stored without the objective constant. This does not change
the algorithm because the comparisons are elementwise and `_h` uses only
differences between points. It also means that the right-hand side of the
ε-constraints, which have their constant moved to the set, is the stored value.
"""
function _round_objective(y::Float64, constant::Float64)
    return round(Int, y - constant)
end

# This is Algorithm 1 from the paper.
function _update_search_region(
    U_N::Dict{Vector{Int},Vector{Vector{Vector{Int}}}},
    y::Vector{Int},
    yN::Vector{Int},
)
    p = length(y)
    bounds_to_remove = Vector{Int}[]
    bounds_to_add = Dict{Vector{Int},Vector{Vector{Vector{Int}}}}()
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

function _get_child(u::Vector{Int}, y::Vector{Int}, k::Int)
    @assert length(u) == length(y)
    child = copy(u)
    child[k] = y[k]
    return child
end

# This is function h(u, k) in the paper. We use Float64 because the product can
# overflow an Int.
function _h(k::Int, u::Vector{Int}, yI::Vector{Int})
    h = 1.0
    for (i, (u_i, yI_i)) in enumerate(zip(u, yI))
        if i != k
            h *= Float64(u_i - yI_i)
        end
    end
    return h
end

# This is Problem (6) in the paper. The upper bound M is our nadir point yN.
function _select_search_zone(
    U_N::Dict{Vector{Int},Vector{Vector{Vector{Int}}}},
    yI::Vector{Int},
    yN::Vector{Int},
)
    k_star, u_star, v_star = 1, yN, -Inf
    for k in 1:length(yI), u in keys(U_N)
        if u[k] != yN[k]
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
    inner::MOI.ModelLike,
    f::MOI.AbstractVectorFunction,
)
    solutions = Dict{Vector{Int},Dict{MOI.VariableIndex,Float64}}()
    status = _minimize_multiobjective!(
        algorithm,
        model,
        inner,
        f,
        solutions,
    )::MOI.TerminationStatusCode
    obj_constants = MOI.constant.(MOI.Utilities.scalarize(f))
    return status, SolutionPoint[
        SolutionPoint(X, Y .+ obj_constants) for (Y, X) in solutions
    ]
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
    solutions::Dict{Vector{Int},Dict{MOI.VariableIndex,Float64}},
)
    @assert MOI.get(inner, MOI.ObjectiveSense()) == MOI.MIN_SENSE
    warm_start_supported =
        MOI.supports(inner, MOI.VariablePrimalStart(), MOI.VariableIndex)
    variables = MOI.get(inner, MOI.ListOfVariableIndices())
    n = MOI.output_dimension(f)
    # Compute the ideal (yI) and nadir (yN) points
    yI, yN = zeros(Int, n), zeros(Int, n)
    scalars = MOI.Utilities.scalarize(f)
    obj_constants = MOI.constant.(scalars)
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
        yI[i] = _round_objective(y_i, obj_constants[i])
        model.ideal_point[i] = yI[i] + obj_constants[i]
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
        yN[i] = _round_objective(y_i, obj_constants[i]) + 1
    end
    MOI.set(inner, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    # Instead of adding and deleting constraints during the algorithm, we add
    # them all here at the beginning.
    ε_constraints = [
        MOI.Utilities.normalize_and_add_constraint(
            inner,
            f_i,
            MOI.LessThan(yN[i] + obj_constants[i]),
        ) for (i, f_i) in enumerate(scalars)
    ]
    U_N = Dict{Vector{Int},Vector{Vector{Vector{Int}}}}(
        # The nadir point, except for the ideal point in position k
        yN => [[_get_child(yN, yI, k)] for k in 1:n],
    )
    V = [Tuple{Vector{Int},Vector{Int}}[] for k in 1:n]
    status = MOI.OPTIMAL
    sum_f = MOI.Utilities.operate(+, Float64, scalars...)
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
            set = MOI.LessThan(Float64(u_i - 1))
            MOI.set(inner, MOI.ConstraintSet(), ε_constraints[i], set)
        end
        if warm_start_supported && !isempty(solutions)
            for (x_i, start_i) in solutions[last(U_N[u][k])]
                MOI.set(inner, MOI.VariablePrimalStart(), x_i, start_i)
            end
        end
        optimize_inner!(model)  # We don't log this first-stage subproblem.
        status = MOI.get(inner, MOI.TerminationStatus())
        if !_is_scalar_status_optimal(status)
            break
        end
        # Now solve problem Π²(k, u)
        obj = MOI.get(inner, MOI.ObjectiveValue())::Float64
        y_k = _round_objective(obj, obj_constants[k])
        set = MOI.LessThan(Float64(y_k))
        MOI.set(inner, MOI.ConstraintSet(), ε_constraints[k], set)
        MOI.set(inner, MOI.ObjectiveFunction{typeof(sum_f)}(), sum_f)
        optimize_inner!(model)
        status = MOI.get(inner, MOI.TerminationStatus())
        if !_is_scalar_status_optimal(status)
            break
        end
        X, Y_float = _compute_point(model, variables, f)
        Y = _round_objective.(Y_float, obj_constants)
        _log_subproblem_solve(model, Y .+ obj_constants)
        push!(V[k], (u, Y))
        if !(Y in U_N[u][k])
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
function _clean_search_region(U_N, yI, V, k)
    map_keys = collect(enumerate(keys(U_N)))
    to_delete = fill(false, length(map_keys))
    Threads.@threads for (i, u′) in map_keys
        to_delete[i] = _clean_search_region_inner(u′, U_N, yI, V, k)
    end                                                                         # COV_EXCL_LINE
    for (i, u′) in map_keys
        if to_delete[i]
            delete!(U_N, u′)
        end
    end
    return
end

function _clean_search_region_inner(u′, U_N, yI, V, k)
    for (u′_k, yI_k) in zip(u′, yI)
        if u′_k == yI_k
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
    if y_k[k] != u′[k]
        return false
    end
    for (i, (u′_i, u_i)) in enumerate(zip(u′, u))
        if i != k && !(u′_i <= u_i)
            return false
        end
    end
    return true
end
