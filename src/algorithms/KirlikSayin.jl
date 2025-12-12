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

## Supported problem classes

This algorithm is restricted to problems with:

 * discrete variables only. It will fail to converge if the problem is purely
   continuous.

## Supported optimizer attributes

 * `MOI.TimeLimitSec()`: terminate if the time limit is exceeded and return the
    list of current solutions.
"""
struct KirlikSayin <: AbstractAlgorithm end

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

function minimize_multiobjective!(algorithm::KirlikSayin, model::Optimizer)
    @assert MOI.get(model.inner, MOI.ObjectiveSense()) == MOI.MIN_SENSE
    solutions = SolutionPoint[]
    # Problem with p objectives.
    # Set k = 1, meaning the nondominated points will get projected
    # down to the objective {2, 3, ..., p}
    k = 1
    YN = Vector{Float64}[]
    variables = MOI.get(model.inner, MOI.ListOfVariableIndices())
    n = MOI.output_dimension(model.f)
    yI, yN = zeros(n), zeros(n)
    # This tolerance is really important!
    δ = 1.0
    scalars = MOI.Utilities.scalarize(model.f)
    # Ideal and Nadir point estimation
    for (i, f_i) in enumerate(scalars)
        # Ideal point
        MOI.set(model.inner, MOI.ObjectiveFunction{typeof(f_i)}(), f_i)
        optimize_inner!(model)
        status = MOI.get(model.inner, MOI.TerminationStatus())
        if !_is_scalar_status_optimal(status)
            return status, nothing
        end
        _, Y = _compute_point(model, variables, f_i)
        _log_subproblem_solve(model, variables)
        model.ideal_point[i] = yI[i] = Y
        # Nadir point
        MOI.set(model.inner, MOI.ObjectiveSense(), MOI.MAX_SENSE)
        optimize_inner!(model)
        status = MOI.get(model.inner, MOI.TerminationStatus())
        if !_is_scalar_status_optimal(status)
            # Repair ObjectiveSense before exiting
            MOI.set(model.inner, MOI.ObjectiveSense(), MOI.MIN_SENSE)
            _warn_on_nonfinite_anti_ideal(algorithm, MOI.MIN_SENSE, i)
            return status, nothing
        end
        _, Y = _compute_point(model, variables, f_i)
        _log_subproblem_solve(model, variables)
        yN[i] = Y + δ
        MOI.set(model.inner, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    end
    L = [_Rectangle(_project(yI, k), _project(yN, k))]
    status = MOI.OPTIMAL
    while !isempty(L)
        if (ret = _check_premature_termination(model)) !== nothing
            status = ret
            break
        end
        max_volume_index = argmax([_volume(Rᵢ, _project(yI, k)) for Rᵢ in L])
        uᵢ = L[max_volume_index].u
        # Solving the first stage model: P_k(ε)
        #   minimize: f_1(x)
        #       s.t.: f_i(x) <= u_i - δ
        @assert k == 1
        MOI.set(
            model.inner,
            MOI.ObjectiveFunction{typeof(scalars[k])}(),
            scalars[k],
        )
        ε_constraints = Any[]
        for (i, f_i) in enumerate(scalars)
            if i == k
                continue
            end
            ci = MOI.Utilities.normalize_and_add_constraint(
                model.inner,
                f_i,
                MOI.LessThan{Float64}(uᵢ[i-1] - δ),
            )
            push!(ε_constraints, ci)
        end
        optimize_inner!(model)
        _log_subproblem_solve(model, "auxillary subproblem")
        if !_is_scalar_status_optimal(model)
            # If this fails, it likely means that the solver experienced a
            # numerical error with this box. Just skip it.
            _remove_rectangle(L, _Rectangle(_project(yI, k), uᵢ))
            MOI.delete.(model, ε_constraints)
            continue
        end
        zₖ = MOI.get(model.inner, MOI.ObjectiveValue())
        # Solving the second stage model: Q_k(ε, zₖ)
        # Set objective sum(model.f)
        sum_f = MOI.Utilities.operate(+, Float64, scalars...)
        MOI.set(model.inner, MOI.ObjectiveFunction{typeof(sum_f)}(), sum_f)
        # Constraint to eliminate weak dominance
        zₖ_constraint = MOI.Utilities.normalize_and_add_constraint(
            model.inner,
            scalars[k],
            MOI.EqualTo(zₖ),
        )
        optimize_inner!(model)
        if !_is_scalar_status_optimal(model)
            _log_subproblem_solve(model, "subproblem not optimal")
            # If this fails, it likely means that the solver experienced a
            # numerical error with this box. Just skip it.
            MOI.delete.(model, ε_constraints)
            MOI.delete(model, zₖ_constraint)
            _remove_rectangle(L, _Rectangle(_project(yI, k), uᵢ))
            continue
        end
        X, Y = _compute_point(model, variables, model.f)
        _log_subproblem_solve(model, Y)
        Y_proj = _project(Y, k)
        if !(Y in YN)
            push!(solutions, SolutionPoint(X, Y))
            push!(YN, Y)
            L = _update_list(L, Y_proj)
        end
        _remove_rectangle(L, _Rectangle(Y_proj, uᵢ))
        MOI.delete.(model, ε_constraints)
        MOI.delete(model, zₖ_constraint)
    end
    return status, filter_nondominated(MOI.MIN_SENSE, solutions)
end
