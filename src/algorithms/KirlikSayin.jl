#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

"""
    KirlikSayin()

`KirlikSayin` implements the algorithm of:

Kirlik, G., & Sayın, S. (2014). A new algorithm for generating all nondominated
solutions of multiobjective discrete optimization problems. European Journal of
Operational Research, 232(3), 479-488.

This is an algorithm to generate all nondominated solutions for multi-objective
discrete optimization problems. The algorithm maintains `(p-1)`-dimensional
rectangle regions in the solution space, and a two-stage optimization problem
is solved for each rectangle.

## Supported optimizer attributes

 * `MOI.TimeLimitSec()`: terminate if the time limit is exceeded and return the
    list of current solutions.
"""
mutable struct KirlikSayin <: AbstractAlgorithm end

struct _Rectangle
    l::Vector{Float64}
    u::Vector{Float64}

    function _Rectangle(l::Vector{Float64}, u::Vector{Float64})
        @assert length(l) == length(u) "Dimension mismatch between l and u"
        return new(l, u)
    end
end

_volume(r::_Rectangle, l::Vector{Float64}) = prod(r.u - l)

function Base.issubset(x::_Rectangle, y::_Rectangle)
    @assert length(x.l) == length(y.l) "Dimension mismatch"
    return all(x.l .>= y.l) && all(x.u .<= y.u)
end

function _remove_rectangle(L::Vector{_Rectangle}, R::_Rectangle)
    index_to_remove = Int[t for (t, x) in enumerate(L) if issubset(x, R)]
    deleteat!(L, index_to_remove)
    return
end

function _split_rectangle(r::_Rectangle, axis::Int, f::Float64)
    l = [i != axis ? r.l[i] : f for i in 1:length(r.l)]
    u = [i != axis ? r.u[i] : f for i in 1:length(r.l)]
    return _Rectangle(r.l, u), _Rectangle(l, r.u)
end

function _update_list(L::Vector{_Rectangle}, f::Vector{Float64})
    L_new = _Rectangle[]
    for Rᵢ in L
        lᵢ, uᵢ = Rᵢ.l, Rᵢ.u
        T = [Rᵢ]
        for j in 1:length(f)
            if lᵢ[j] < f[j] < uᵢ[j]
                T̄ = _Rectangle[]
                for Rₜ in T
                    a, b = _split_rectangle(Rₜ, j, f[j])
                    push!(T̄, a)
                    push!(T̄, b)
                end
                T = T̄
            end
        end
        append!(L_new, T)
    end
    return L_new
end

function optimize_multiobjective!(algorithm::KirlikSayin, model::Optimizer)
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
    solutions = SolutionPoint[]
    # Problem with p objectives.
    # Set k = 1, meaning the nondominated points will get projected
    # down to the objective {2, 3, ..., p}
    k = 1
    YN = Vector{Float64}[]
    variables = MOI.get(model.inner, MOI.ListOfVariableIndices())
    n = MOI.output_dimension(model.f)
    yI, yN = zeros(n), zeros(n)
    δ = sense == MOI.MIN_SENSE ? -1 : 1
    scalars = MOI.Utilities.scalarize(model.f)
    # Ideal and Nadir point estimation
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
        MOI.set(
            model.inner,
            MOI.ObjectiveSense(),
            sense == MOI.MIN_SENSE ? MOI.MAX_SENSE : MOI.MIN_SENSE,
        )
        MOI.optimize!(model.inner)
        status = MOI.get(model.inner, MOI.TerminationStatus())
        if !_is_scalar_status_optimal(status)
            _warn_on_nonfinite_anti_ideal(algorithm, sense, i)
            return status, nothing
        end
        _, Y = _compute_point(model, variables, f_i)
        yN[i] = Y
    end
    # Reset the sense after modifying it.
    MOI.set(model.inner, MOI.ObjectiveSense(), sense)
    L = [_Rectangle(_project(yI, k), _project(yN, k))]
    SetType = ifelse(
        sense == MOI.MIN_SENSE,
        MOI.LessThan{Float64},
        MOI.GreaterThan{Float64},
    )
    status = MOI.OPTIMAL
    while !isempty(L)
        if _time_limit_exceeded(model, start_time)
            status = MOI.TIME_LIMIT
            break
        end
        Rᵢ = L[argmax([_volume(Rᵢ, _project(yI, k)) for Rᵢ in L])]
        lᵢ, uᵢ = Rᵢ.l, Rᵢ.u
        # Solving the first stage model: P_k(ε)
        # Set ε := uᵢ
        ε = insert!(copy(uᵢ), k, 0.0)
        ε_constraints = Any[]
        MOI.set(
            model.inner,
            MOI.ObjectiveFunction{typeof(scalars[k])}(),
            scalars[k],
        )
        for (i, f_i) in enumerate(scalars)
            if i != k
                ci = MOI.Utilities.normalize_and_add_constraint(
                    model.inner,
                    f_i,
                    SetType(ε[i] + δ),
                )
                push!(ε_constraints, ci)
            end
        end
        MOI.optimize!(model.inner)
        if !_is_scalar_status_optimal(model)
            _remove_rectangle(L, _Rectangle(_project(yI, k), uᵢ))
            MOI.delete.(model, ε_constraints)
            continue
        end
        zₖ = MOI.get(model.inner, MOI.ObjectiveValue())
        # Solving the second stage model: Q_k(ε, zₖ)
        # Set objective sum(model.f)
        sum_f = sum(1.0 * s for s in scalars)
        MOI.set(model.inner, MOI.ObjectiveFunction{typeof(sum_f)}(), sum_f)
        # Constraint to eliminate weak dominance
        zₖ_constraint = MOI.Utilities.normalize_and_add_constraint(
            model.inner,
            scalars[k],
            MOI.EqualTo(zₖ),
        )
        MOI.optimize!(model.inner)
        MOI.delete.(model, ε_constraints)
        MOI.delete(model, zₖ_constraint)
        if !_is_scalar_status_optimal(model)
            _remove_rectangle(L, _Rectangle(_project(yI, k), uᵢ))
            continue
        end
        X, Y = _compute_point(model, variables, model.f)
        Y_proj = _project(Y, k)
        if !(Y in YN)
            push!(solutions, SolutionPoint(X, Y))
            push!(YN, Y)
            L = _update_list(L, Y_proj)
        end
        _remove_rectangle(L, _Rectangle(Y_proj, uᵢ))
    end
    return status, solutions
end
