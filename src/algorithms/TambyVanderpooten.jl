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

## Supported optimizer attributes

 * `MOI.TimeLimitSec()`: terminate if the time limit is exceeded and return the
    list of current solutions.
"""
mutable struct TambyVanderpooten <: AbstractAlgorithm end

function _update_search_region(
    U_N::Dict{Vector{Float64},Vector{Vector{Vector{Float64}}}},
    y::Vector{Float64},
    yN::Vector{Float64},
)
    bounds_to_remove = Vector{Float64}[]
    p = length(y)
    for u in keys(U_N)
        if all(y .< u)
            push!(bounds_to_remove, u)
            for l in 1:p
                u_l = _get_child(u, y, l)
                N = [
                    k != l ? [yi for yi in U_N[u][k] if yi[l] < y[l]] : [y]
                    for k in 1:p
                ]
                if all(!isempty(N[k]) for k in 1:p if u_l[k] ≠ yN[k])
                    U_N[u_l] = N
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
    for bound_to_remove in bounds_to_remove
        delete!(U_N, bound_to_remove)
    end
    return
end

function _get_child(u::Vector{Float64}, y::Vector{Float64}, k::Int)
    @assert length(u) == length(y)
    return vcat(u[1:k-1], y[k], u[k+1:length(y)])
end

function _select_search_zone(
    U_N::Dict{Vector{Float64},Vector{Vector{Vector{Float64}}}},
    yI::Vector{Float64},
)
    i, j =
        argmax([
            prod(_project(u, k) - _project(yI, k)) for k in 1:length(yI),
            u in keys(U_N)
        ]).I
    return i, collect(keys(U_N))[j]
end

function optimize_multiobjective!(
    algorithm::TambyVanderpooten,
    model::Optimizer,
)
    start_time = time()
    sense = MOI.get(model.inner, MOI.ObjectiveSense())
    if sense == MOI.MAX_SENSE
        old_obj, neg_obj = copy(model.f), -model.f
        MOI.set(model, MOI.ObjectiveFunction{typeof(neg_obj)}(), neg_obj)
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        status, solutions = optimize_multiobjective!(algorithm, model)
        MOI.set(model, MOI.ObjectiveFunction{typeof(old_obj)}(), old_obj)
        MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
        if solutions !== nothing
            solutions = [SolutionPoint(s.x, -s.y) for s in solutions]
        end
        return status, solutions
    end
    warm_start_supported = false
    if MOI.supports(model, MOI.VariablePrimalStart(), MOI.VariableIndex)
        warm_start_supported = true
    end
    solutions = Dict{Vector{Float64},Dict{MOI.VariableIndex,Float64}}()
    YN = Vector{Float64}[]
    variables = MOI.get(model.inner, MOI.ListOfVariableIndices())
    n = MOI.output_dimension(model.f)
    yI, yN = zeros(n), zeros(n)
    scalars = MOI.Utilities.scalarize(model.f)
    for (i, f_i) in enumerate(scalars)
        MOI.set(model.inner, MOI.ObjectiveFunction{typeof(f_i)}(), f_i)
        MOI.set(model.inner, MOI.ObjectiveSense(), sense)
        MOI.optimize!(model.inner)
        status = MOI.get(model.inner, MOI.TerminationStatus())
        if !_is_scalar_status_optimal(status)
            return status, nothing
        end
        _, Y = _compute_point(model, variables, f_i)
        yI[i] = Y + 1
        MOI.set(model.inner, MOI.ObjectiveSense(), MOI.MAX_SENSE)
        MOI.optimize!(model.inner)
        status = MOI.get(model.inner, MOI.TerminationStatus())
        if !_is_scalar_status_optimal(status)
            _warn_on_nonfinite_anti_ideal(algorithm, sense, i)
            return status, nothing
        end
        _, Y = _compute_point(model, variables, f_i)
        yN[i] = Y
    end
    MOI.set(model.inner, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    U_N = Dict{Vector{Float64},Vector{Vector{Vector{Float64}}}}()
    V = [Tuple{Vector{Float64},Vector{Float64}}[] for k in 1:n]
    U_N[yN] = [[_get_child(yN, yI, k)] for k in 1:n]
    status = MOI.OPTIMAL
    while !isempty(U_N)
        if _time_limit_exceeded(model, start_time)
            status = MOI.TIME_LIMIT
            break
        end
        k, u = _select_search_zone(U_N, yI)
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
        MOI.optimize!(model.inner)
        if !_is_scalar_status_optimal(model)
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
        MOI.optimize!(model.inner)
        if !_is_scalar_status_optimal(model)
            return status, nothing
        end
        X, Y = _compute_point(model, variables, model.f)
        MOI.delete.(model, ε_constraints)
        MOI.delete(model, y_k_constraint)
        push!(V[k], (u, Y))
        if Y ∉ U_N[u][k]
            _update_search_region(U_N, Y, yN)
            solutions[Y] = X
        end
        bounds_to_remove = Vector{Float64}[]
        for u_i in keys(U_N)
            for k in 1:n
                if u_i[k] == yI[k]
                    push!(bounds_to_remove, u_i)
                else
                    for (u_j, y_j) in V[k]
                        if all(_project(u_i, k) .<= _project(u_j, k)) &&
                           (y_j[k] == u_i[k])
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
    solutions = [SolutionPoint(X, Y) for (Y, X) in solutions]
    return status, solutions
end
