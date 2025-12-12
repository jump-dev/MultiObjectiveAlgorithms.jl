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

_describe(::TambyVanderpooten) = "TambyVanderpooten()"

function _update_search_region(
    U_N::Dict{Vector{Float64},Vector{Vector{Vector{Float64}}}},
    y::Vector{Float64},
    yN::Vector{Float64},
)
    p = length(y)
    bounds_to_remove = Vector{Float64}[]
    bounds_to_add = Dict{Vector{Float64},Vector{Vector{Vector{Float64}}}}()
    for u in keys(U_N)
        if all(y .< u)
            push!(bounds_to_remove, u)
            for l in 1:p
                u_l = _get_child(u, y, l)
                N = [
                    k == l ? [y] : [yi for yi in U_N[u][k] if yi[l] < y[l]]
                    for k in 1:p
                ]
                if all(!isempty(N[k]) for k in 1:p if k != l && u_l[k] != yN[k])
                    bounds_to_add[u_l] = N
                end
            end
        else
            for k in 1:p
                if (y[k] == u[k]) && all(_project(y, k) .< _project(u, k))
                    push!(U_N[u][k], y)
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

function _get_child(u::Vector{Float64}, y::Vector{Float64}, k::Int)
    @assert length(u) == length(y)
    return vcat(u[1:(k-1)], y[k], u[(k+1):length(y)])
end

function _select_search_zone(
    U_N::Dict{Vector{Float64},Vector{Vector{Vector{Float64}}}},
    yI::Vector{Float64},
    yN::Vector{Float64},
)
    upper_bounds = collect(keys(U_N))
    p = length(yI)
    hvs = [
        u[k] == yN[k] ? 0.0 : prod(_project(u, k) .- _project(yI, k)) for
        k in 1:p, u in upper_bounds
    ]
    k_star, j_star = argmax(hvs).I
    return k_star, upper_bounds[j_star]
end

function minimize_multiobjective!(
    algorithm::TambyVanderpooten,
    model::Optimizer,
)
    @assert MOI.get(model.inner, MOI.ObjectiveSense()) == MOI.MIN_SENSE
    warm_start_supported = false
    if MOI.supports(model, MOI.VariablePrimalStart(), MOI.VariableIndex)
        warm_start_supported = true
    end
    solutions = Dict{Vector{Float64},Dict{MOI.VariableIndex,Float64}}()
    variables = MOI.get(model.inner, MOI.ListOfVariableIndices())
    n = MOI.output_dimension(model.f)
    yI, yN = zeros(n), zeros(n)
    scalars = MOI.Utilities.scalarize(model.f)
    for (i, f_i) in enumerate(scalars)
        MOI.set(model.inner, MOI.ObjectiveFunction{typeof(f_i)}(), f_i)
        MOI.set(model.inner, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        optimize_inner!(model)
        status = MOI.get(model.inner, MOI.TerminationStatus())
        if !_is_scalar_status_optimal(status)
            return status, nothing
        end
        _, Y = _compute_point(model, variables, f_i)
        _log_solution(model, variables)
        yI[i] = Y
        model.ideal_point[i] = Y
        MOI.set(model.inner, MOI.ObjectiveSense(), MOI.MAX_SENSE)
        optimize_inner!(model)
        status = MOI.get(model.inner, MOI.TerminationStatus())
        if !_is_scalar_status_optimal(status)
            _warn_on_nonfinite_anti_ideal(algorithm, MOI.MIN_SENSE, i)
            return status, nothing
        end
        _, Y = _compute_point(model, variables, f_i)
        _log_solution(model, variables)
        yN[i] = Y + 1
    end
    MOI.set(model.inner, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    U_N = Dict{Vector{Float64},Vector{Vector{Vector{Float64}}}}()
    V = [Tuple{Vector{Float64},Vector{Float64}}[] for k in 1:n]
    U_N[yN] = [[_get_child(yN, yI, k)] for k in 1:n]
    status = MOI.OPTIMAL
    while !isempty(U_N)
        if (ret = _check_premature_termination(model)) !== nothing
            status = ret
            break
        end
        k, u = _select_search_zone(U_N, yI, yN)
        MOI.set(
            model.inner,
            MOI.ObjectiveFunction{typeof(scalars[k])}(),
            scalars[k],
        )
        ε_constraints = Any[]
        for (i, f_i) in enumerate(scalars)
            if i != k
                ci = MOI.Utilities.normalize_and_add_constraint(
                    model.inner,
                    f_i,
                    MOI.LessThan{Float64}(u[i] - 1),
                )
                push!(ε_constraints, ci)
            end
        end
        if u[k] ≠ yN[k]
            if warm_start_supported
                variables_start = solutions[first(U_N[u][k])]
                for x_i in variables
                    MOI.set(
                        model.inner,
                        MOI.VariablePrimalStart(),
                        x_i,
                        variables_start[x_i],
                    )
                end
            end
        end
        optimize_inner!(model)
        _log_solution(model, "auxillary subproblem")
        status = MOI.get(model.inner, MOI.TerminationStatus())
        if !_is_scalar_status_optimal(status)
            MOI.delete.(model, ε_constraints)
            return status, nothing
        end
        y_k = MOI.get(model.inner, MOI.ObjectiveValue())
        sum_f = sum(1.0 * s for s in scalars)
        MOI.set(model.inner, MOI.ObjectiveFunction{typeof(sum_f)}(), sum_f)
        y_k_constraint = MOI.Utilities.normalize_and_add_constraint(
            model.inner,
            scalars[k],
            MOI.EqualTo(y_k),
        )
        optimize_inner!(model)
        status = MOI.get(model.inner, MOI.TerminationStatus())
        if !_is_scalar_status_optimal(status)
            MOI.delete.(model, ε_constraints)
            MOI.delete(model, y_k_constraint)
            return status, nothing
        end
        X, Y = _compute_point(model, variables, model.f)
        _log_solution(model, Y)
        MOI.delete.(model, ε_constraints)
        MOI.delete(model, y_k_constraint)
        push!(V[k], (u, Y))
        # We want `if !(Y in U_N[u][k])` but this tests exact equality. We want
        # an approximate comparison.
        if all(!isapprox(Y; atol = 1e-6), U_N[u][k])
            _update_search_region(U_N, Y, yN)
            solutions[Y] = X
        end
        bounds_to_remove = Vector{Float64}[]
        for u_i in keys(U_N)
            for k in 1:n
                if isapprox(u_i[k], yI[k]; atol = 1e-6)
                    push!(bounds_to_remove, u_i)
                else
                    for (u_j, y_j) in V[k]
                        if all(_project(u_i, k) .<= _project(u_j, k)) &&
                           isapprox(y_j[k], u_i[k]; atol = 1e-6)
                            push!(bounds_to_remove, u_i)
                        end
                    end
                end
            end
        end
        if !isempty(bounds_to_remove)
            for bound_to_remove in bounds_to_remove
                delete!(U_N, bound_to_remove)
            end
        end
    end
    return status, [SolutionPoint(X, Y) for (Y, X) in solutions]
end
