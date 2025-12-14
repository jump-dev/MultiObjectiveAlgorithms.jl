#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

"""
    DominguezRios()

`DominguezRios` implements the algorithm of:

Dominguez-Rios, M.A. & Chicano, F., & Alba, E. (2021). Effective anytime
algorithm for multiobjective combinatorial optimization problems. Information
Sciences, 565(7), 210-228.

## Supported problem classes

This algorithm is restricted to problems with:

 * discrete variables only. It will fail to converge if the problem is purely
   continuous.

## Supported optimizer attributes

 * `MOI.TimeLimitSec()`: terminate if the time limit is exceeded and return the
   list of current solutions.
"""
struct DominguezRios <: AbstractAlgorithm end

mutable struct _DominguezRiosBox
    l::Vector{Float64}
    u::Vector{Float64}
    priority::Float64
    function _DominguezRiosBox(
        l::Vector{Float64},
        u::Vector{Float64},
        p::Float64 = 0.0,
    )
        @assert length(l) == length(u) "Dimension mismatch between l and u"
        return new(l, u, p)
    end
end

function _reduced_scaled_priority(
    l::Vector{Float64},
    u::Vector{Float64},
    i::Int,
    z::Vector{Float64},
    yI::Vector{Float64},
    yN::Vector{Float64},
)
    ret = prod((u - l) ./ (yN - yI))
    if i != length(z)
        return ret
    end
    return ret - prod((z - l) ./ (yN - yI))
end

function _p_partition(
    B::_DominguezRiosBox,
    z::Vector{Float64},
    yI::Vector{Float64},
    yN::Vector{Float64},
)
    ẑ = max.(z, B.l)
    ret = _DominguezRiosBox[]
    for i in 1:length(z)
        new_l = vcat(B.l[1:i], ẑ[(i+1):end])
        new_u = vcat(B.u[1:(i-1)], ẑ[i], B.u[(i+1):end])
        new_priority = _reduced_scaled_priority(new_l, new_u, i, ẑ, yI, yN)
        push!(ret, _DominguezRiosBox(new_l, new_u, new_priority))
    end
    return ret
end

function _select_next_box(L::Vector{Vector{_DominguezRiosBox}}, k::Int)
    p = length(L)
    @assert any(!isempty(l) for l in L)
    k = k % p + 1
    while isempty(L[k])
        k = k % p + 1
    end
    i = argmax([B.priority for B in L[k]])
    return i, k
end

function _join(
    A::_DominguezRiosBox,
    B::_DominguezRiosBox,
    i::Int,
    z::Vector{Float64},
    yI::Vector{Float64},
    yN::Vector{Float64},
)
    lᵃ, uᵃ, lᵇ, uᵇ = A.l, A.u, B.l, B.u
    @assert all(uᵃ .<= uᵇ) "`join` operation not valid. (uᵃ ≰ uᵇ)"
    lᶜ, uᶜ = min.(lᵃ, lᵇ), uᵇ
    ẑ = max.(z, lᶜ)
    priority = _reduced_scaled_priority(lᶜ, uᶜ, i, ẑ, yI, yN)
    return _DominguezRiosBox(lᶜ, uᶜ, priority)
end

function Base.isempty(B::_DominguezRiosBox)
    return any(isapprox(B.l[i], B.u[i]) for i in 1:length(B.u))
end

function _update!(
    L::Vector{Vector{_DominguezRiosBox}},
    z::Vector{Float64},
    yI::Vector{Float64},
    yN::Vector{Float64},
)
    T = [_DominguezRiosBox[] for _ in 1:length(L)]
    for j in 1:length(L)
        for B in L[j]
            if all(z .< B.u)
                for (i, Bᵢ) in enumerate(_p_partition(B, z, yI, yN))
                    if !isempty(Bᵢ)
                        push!(T[i], Bᵢ)
                    end
                end
            else
                push!(T[j], B)
            end
        end
    end
    L .= T
    for k in 1:length(L)
        i = 1
        N = length(L[k])
        while i < N
            index_to_remove = Int[]
            for j in i:N
                if i != j
                    if all(L[k][i].u .<= L[k][j].u)
                        L[k][i] = _join(L[k][i], L[k][j], k, z, yI, yN)
                        push!(index_to_remove, j)
                    elseif all(L[k][i].u .>= L[k][j].u)
                        L[k][i] = _join(L[k][j], L[k][i], k, z, yI, yN)
                        push!(index_to_remove, j)
                    end
                end
            end
            i += 1
            N -= length(index_to_remove)
            deleteat!(L[k], index_to_remove)
        end
    end
    return
end

function minimize_multiobjective!(algorithm::DominguezRios, model::Optimizer)
    @assert MOI.get(model.inner, MOI.ObjectiveSense()) == MOI.MIN_SENSE
    n = MOI.output_dimension(model.f)
    L = [_DominguezRiosBox[] for i in 1:n]
    scalars = MOI.Utilities.scalarize(model.f)
    variables = MOI.get(model.inner, MOI.ListOfVariableIndices())
    yI, yN = zeros(n), zeros(n)
    # Ideal and Nadir point estimation
    for (i, f_i) in enumerate(scalars)
        MOI.set(model.inner, MOI.ObjectiveFunction{typeof(f_i)}(), f_i)
        MOI.set(model.inner, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        optimize_inner!(model)
        status = MOI.get(model.inner, MOI.TerminationStatus())
        if !_is_scalar_status_optimal(status)
            return status, nothing
        end
        _, Y = _compute_point(model, variables, f_i)
        _log_subproblem_solve(model, variables)
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
        _log_subproblem_solve(model, variables)
        yN[i] = Y + 1
    end
    MOI.set(model.inner, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    ϵ = 1 / (2 * n * (maximum(yN - yI) - 1))
    # If ϵ is small, then the scalar objectives can contain terms that fall
    # below the tolerance level of the solver. To fix this, we rescale the
    # objective so that the coefficients have magnitude `1e+00` or larger.
    scale = max(1.0, 1 / ϵ)
    push!(L[1], _DominguezRiosBox(yI, yN, 0.0))
    t_max = MOI.add_variable(model.inner)
    solutions = SolutionPoint[]
    k = 0
    status = MOI.OPTIMAL
    B_prevs = Vector{Union{Nothing,_DominguezRiosBox}}(nothing, n)
    iter = 0
    while any(!isempty(l) for l in L)
        iter += 1
        if (ret = _check_premature_termination(model)) !== nothing
            status = ret
            break
        end
        i, k = _select_next_box(L, k)
        B = L[k][i]
        if iter > n
            if !isnothing(B_prevs[k])
                if (B_prevs[k].l ≈ B.l) && (B_prevs[k].u ≈ B.u)
                    deleteat!(L[k], i)
                    continue
                end
            end
        end
        B_prevs[k] = B
        # We're going to scale `w` here by `scale` instead of the usual
        # `1 / max(...)`. It will show up in a few places bbelow.
        w = scale ./ max.(1, B.u - yI)
        constraints = [
            MOI.Utilities.normalize_and_add_constraint(
                model.inner,
                # `w` is the scaled version here. This epigraph constraint will
                # make `t_max` similarly scaled.
                t_max - (w[i] * (scalars[i] - yI[i])),
                MOI.GreaterThan(0.0),
            ) for i in 1:n
        ]
        # There's no need to scale anything explicitly here:
        #  * t_max is already covered in `constraints`
        #  * the `ϵ` term is already covered in `w`
        new_f = t_max + ϵ * sum(w[i] * (scalars[i] - yI[i]) for i in 1:n)
        MOI.set(model.inner, MOI.ObjectiveFunction{typeof(new_f)}(), new_f)
        optimize_inner!(model)
        if _is_scalar_status_optimal(model)
            X, Y = _compute_point(model, variables, model.f)
            _log_subproblem_solve(model, Y)
            obj = MOI.get(model.inner, MOI.ObjectiveValue())
            # We need to undo the scaling of the scalar objective. There's no
            # need to unscale `Y` because we have evaluated this explicitly from
            # the modified `model.f`.
            obj /= scale
            if (obj < 1) && all(yI .< B.u)
                push!(solutions, SolutionPoint(X, Y))
                _update!(L, Y, yI, yN)
            else
                deleteat!(L[k], i)
            end
        else
            # In theory, this shouldn't happen, because this subproblem is meant
            # to always be feasible. However, in some of our testing, HiGHS will
            # fail and return something like OTHER_ERROR (e.g., because the
            # numerics are challenging). Rather than error completely, let's
            # just skip this box.
            _log_subproblem_solve(model, "subproblem not optimal")
            deleteat!(L[k], i)
        end
        MOI.delete.(model.inner, constraints)
    end
    MOI.delete(model.inner, t_max)
    return status, solutions
end
