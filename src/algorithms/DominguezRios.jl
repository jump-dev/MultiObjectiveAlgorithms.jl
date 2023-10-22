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

## Supported optimizer attributes

 * `MOI.TimeLimitSec()`: terminate if the time limit is exceeded and return the
   list of current solutions.
"""
mutable struct DominguezRios <: AbstractAlgorithm end

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
        new_l = vcat(B.l[1:i], ẑ[i+1:end])
        new_u = vcat(B.u[1:i-1], ẑ[i], B.u[i+1:end])
        new_priority = _reduced_scaled_priority(new_l, new_u, i, ẑ, yI, yN)
        push!(ret, _DominguezRiosBox(new_l, new_u, new_priority))
    end
    return ret
end

function _select_next_box(L::Vector{Vector{_DominguezRiosBox}}, k::Int)
    p = length(L)
    if any(.!isempty.(L))
        k = k % p + 1
        while isempty(L[k])
            k = k % p + 1
        end
        i = argmax([B.priority for B in L[k]])
    end
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

function optimize_multiobjective!(algorithm::DominguezRios, model::Optimizer)
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
    n = MOI.output_dimension(model.f)
    L = [_DominguezRiosBox[] for i in 1:n]
    scalars = MOI.Utilities.scalarize(model.f)
    variables = MOI.get(model.inner, MOI.ListOfVariableIndices())
    yI, yN = zeros(n), zeros(n)
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
        yI[i] = Y
        rev_sense = sense == MOI.MIN_SENSE ? MOI.MAX_SENSE : MOI.MIN_SENSE
        MOI.set(model.inner, MOI.ObjectiveSense(), rev_sense)
        MOI.optimize!(model.inner)
        status = MOI.get(model.inner, MOI.TerminationStatus())
        if !_is_scalar_status_optimal(status)
            _warn_on_nonfinite_anti_ideal(algorithm, sense, i)
            return status, nothing
        end
        _, Y = _compute_point(model, variables, f_i)
        yN[i] = Y + 1
    end
    MOI.set(model.inner, MOI.ObjectiveSense(), sense)
    ϵ = 1 / (2 * n * (maximum(yN - yI) - 1))
    push!(L[1], _DominguezRiosBox(yI, yN, 0.0))
    t_max = MOI.add_variable(model.inner)
    solutions = SolutionPoint[]
    k = 0
    status = MOI.OPTIMAL
    while any(!isempty(l) for l in L)
        if _time_limit_exceeded(model, start_time)
            status = MOI.TIME_LIMIT
            break
        end
        i, k = _select_next_box(L, k)
        B = L[k][i]
        w = 1 ./ max.(1, B.u - yI)
        constraints = [
            MOI.Utilities.normalize_and_add_constraint(
                model.inner,
                t_max - (w[i] * (scalars[i] - yI[i])),
                MOI.GreaterThan(0.0),
            ) for i in 1:n
        ]
        new_f = t_max + ϵ * sum(w[i] * (scalars[i] - yI[i]) for i in 1:n)
        MOI.set(model.inner, MOI.ObjectiveFunction{typeof(new_f)}(), new_f)
        MOI.optimize!(model.inner)
        if _is_scalar_status_optimal(model)
            X, Y = _compute_point(model, variables, model.f)
            obj = MOI.get(model.inner, MOI.ObjectiveValue())
            if (obj < 1) && all(yI .< B.u)
                push!(solutions, SolutionPoint(X, Y))
                _update!(L, Y, yI, yN)
            else
                deleteat!(L[k], i)
            end
        end
        MOI.delete.(model.inner, constraints)
    end
    MOI.delete(model.inner, t_max)
    return status, solutions
end
