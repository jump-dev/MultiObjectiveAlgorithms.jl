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
"""

mutable struct DominguezRios <: AbstractAlgorithm end

mutable struct _Box
    l::Vector{Float64}
    u::Vector{Float64}
    priority::Float64
    function _Box(l::Vector{Float64}, u::Vector{Float64}, p::Float64 = 0.0)
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
    B::_Box,
    z::Vector{Float64},
    yI::Vector{Float64},
    yN::Vector{Float64},
)
    ẑ = max.(z, B.l)
    ret = _Box[]
    for i in 1:length(z)
        new_l = vcat(B.l[1:i], ẑ[i+1:end])
        new_u = vcat(B.u[1:i-1], ẑ[i], B.u[i+1:end])
        new_priority = _reduced_scaled_priority(new_l, new_u, i, ẑ, yI, yN)
        push!(ret, _Box(new_l, new_u, new_priority))
    end
    return ret
end

function _select_next_box(L::Vector{Vector{_Box}}, k::Int)
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
    A::_Box,
    B::_Box,
    i::Int,
    z::Vector{Float64},
    yI::Vector{Float64},
    yN::Vector{Float64},
)
    lᵃ, uᵃ, lᵇ, uᵇ = A.l, A.u, B.l, B.u
    @assert all(uᵃ .<= uᵇ) "`join` operation not valid. (uᵃ ≰ uᵇ)"
    lᶜ, uᶜ = min.(lᵃ, lᵇ), uᵇ
    ẑ = max.(z, lᶜ)
    return _Box(lᶜ, uᶜ, _reduced_scaled_priority(lᶜ, uᶜ, i, ẑ, yI, yN))
end

Base.isempty(B::_Box) = prod(B.u[i] - B.l[i] for i in 1:length(B.u)) == 0

function _update!(
    L::Vector{Vector{_Box}},
    z::Vector{Float64},
    yI::Vector{Float64},
    yN::Vector{Float64},
)
    T = [_Box[] for _ in 1:length(L)]
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
    YN = Vector{Float64}[]
    L = [_Box[] for i in 1:n]
    k = 0
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
        MOI.set(
            model.inner,
            MOI.ObjectiveSense(),
            sense == MOI.MIN_SENSE ? MOI.MAX_SENSE : MOI.MIN_SENSE,
        )
        MOI.optimize!(model.inner)
        status = MOI.get(model.inner, MOI.TerminationStatus())
        if !_is_scalar_status_optimal(status)
            @warn(
                "Unable to solve problem using `DominguezRios()` because " *
                "objective $i does not have a finite domain.",
            )
            return status, nothing
        end
        _, Y = _compute_point(model, variables, f_i)
        yN[i] = Y
    end
    MOI.set(model.inner, MOI.ObjectiveSense(), sense)
    r = maximum(yN - yI)
    ϵ = 1 / (2 * n * (r - 1))
    push!(L[1], _Box(yI, yN, 0.0))
    iter = 1
    t_max = MOI.add_variable(model.inner)
    solutions = SolutionPoint[]
    while any(.!isempty.(L))
        i, k = _select_next_box(L, k)
        B = L[k][i]
        z = B.u
        w = 1 ./ max.(1, z - yI)
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
        obj = MOI.get(model.inner, MOI.ObjectiveValue())
        MOI.delete.(model.inner, constraints)
        if MOI.get(model.inner, MOI.TerminationStatus()) == MOI.OPTIMAL
            X, Y = _compute_point(model, variables, model.f)
            if (obj < 1) && all(yI .< z)
                push!(YN, Y)
                push!(solutions, SolutionPoint(X, Y))
                _update!(L, Y, yI, yN)
            else
                deleteat!(L[k], i)
            end
        end
        iter += 1
    end
    MOI.delete(model.inner, t_max)
    return MOI.OPTIMAL, solutions
end
