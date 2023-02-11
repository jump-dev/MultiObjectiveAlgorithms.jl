#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

"""
    DominguezRios()

`DominguezRios` implements the algorithm of Dominguez-Rios et al. (2021).
"""

mutable struct DominguezRios <: AbstractAlgorithm end

mutable struct _Box
    l::Vector{Float64}
    u::Vector{Float64}
    priority::Float64
    function _Box(l::Vector{Float64}, u::Vector{Float64}, priority::Float64)
        @assert length(l) == length(u) "Dimension mismatch between l and u"
        return new(l, u, priority)
    end
end

_Box(l::Vector{Float64}, u::Vector{Float64}) = _Box(l, u, 0.0)

function _scaled_priority(
    l::Vector{Float64},
    u::Vector{Float64},
    yI::Vector{Float64},
    yN::Vector{Float64},
)
    return prod((u - l) ./ (yN - yI))
end

function _scaled_priority(B::_Box, yI::Vector{Float64}, yN::Vector{Float64})
    return _scaled_priority(B.l, B.u, yI, yN)
end

function _reduced_scaled_priority(
    l::Vector{Float64},
    u::Vector{Float64},
    i::Int,
    z::Vector{Float64},
    yI::Vector{Float64},
    yN::Vector{Float64},
)
    if i ≠ length(z)
        return _scaled_priority(l, u, yI, yN)
    else
        return _scaled_priority(l, u, yI, yN) - _scaled_priority(l, z, yI, yN)
    end
end

function _reduced_scaled_priority(
    B::_Box,
    i::Int,
    z::Vector{Float64},
    yI::Vector{Float64},
    yN::Vector{Float64},
)
    return _reduced_scaled_priority(
        B.l,
        B.u,
        i,
        z::Vector{Float64},
        yI::Vector{Float64},
        yN::Vector{Float64},
    )
end

function _p_partition(
    B::_Box,
    z::Vector{Float64},
    yI::Vector{Float64},
    yN::Vector{Float64},
)
    l, u = B.l, B.u
    ẑ = max.(z, l)
    return [
        _Box(
            vcat(l[1:i], ẑ[i+1:end]),
            vcat(u[1:i-1], ẑ[i], u[i+1:end]),
            _reduced_scaled_priority(
                vcat(l[1:i], ẑ[i+1:end]),
                vcat(u[1:i-1], ẑ[i], u[i+1:end]),
                i,
                ẑ,
                yI,
                yN,
            ),
        ) for i in 1:length(z)
    ]
end

function _compute_r(yI::Vector{Float64}, yN::Vector{Float64})
    return maximum(yN - yI)
end

function _compute_ϵ(p::Int, r::Float64)
    return 1 / (2 * p * (r - 1))
end

function _compute_w(z::Vector{Float64}, yI::Vector{Float64})
    return 1 ./ max.(1, z - yI)
end

function _select_next_box(L::Vector{Vector{_Box}}, k::Int)
    p = length(L)
    if any(.!isempty.(L))
        k = k % p + 1
        while isempty(L[k])
            k = k % p + 1
        end
        i = argmax(B.priority for B in L[k])
    end
    return i, k
end

function Base.join(
    A::_Box,
    B::_Box,
    i::Int,
    z::Vector{Float64},
    yI::Vector{Float64},
    yN::Vector{Float64},
)
    lᵃ, uᵃ, lᵇ, uᵇ = A.l, A.u, B.l, B.u

    @assert all(uᵃ .≤ uᵇ) "`join` operation not valid. (uᵃ ≰ uᵇ)"

    lᶜ, uᶜ = min.(lᵃ, lᵇ), uᵇ

    ẑ = max.(z, lᶜ)

    return _Box(lᶜ, uᶜ, _reduced_scaled_priority(lᶜ, uᶜ, i, ẑ, yI, yN))
end

function _volume(u::Vector{Float64}, l::Vector{Float64})
    return prod(u - l)
end

function Base.isempty(B::_Box)
    return _volume(B.u, B.l) == 0
end

function _update!(
    L::Vector{Vector{_Box}},
    z::Vector{Float64},
    yI::Vector{Float64},
    yN::Vector{Float64},
)
    T = [Vector{_Box}() for _ in 1:length(L)]
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
            index_to_remove = []
            for j in i:N
                if i ≠ j
                    if all(L[k][i].u .≤ L[k][j].u)
                        L[k][i] = join(L[k][i], L[k][j], k, z, yI, yN)
                        push!(index_to_remove, j)
                    elseif all(L[k][i].u .≥ L[k][j].u)
                        L[k][i] = join(L[k][j], L[k][i], k, z, yI, yN)
                        push!(index_to_remove, j)
                    end
                end
            end
            i += 1
            N -= length(index_to_remove)
            deleteat!(L[k], index_to_remove)
        end
    end
end

function optimize_multiobjective!(algorithm::DominguezRios, model::Optimizer)
    sense = MOI.get(model.inner, MOI.ObjectiveSense())
    if sense == MOI.MAX_SENSE
        MOI.set(model, MOI.ObjectiveFunction{typeof(model.f)}(), -model.f)
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        status, solutions = optimize_multiobjective!(algorithm, model)
        MOI.set(model, MOI.ObjectiveFunction{typeof(model.f)}(), -model.f)
        MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
        return status,
        [SolutionPoint(solution.x, -solution.y) for solution in solutions]
    end
    n = MOI.output_dimension(model.f)
    solutions = SolutionPoint[]
    YN = Vector{Vector{Float64}}()
    L = fill(Vector{_Box}(), n)
    k = 0
    scalars = MOI.Utilities.scalarize(model.f)
    variables = MOI.get(model.inner, MOI.ListOfVariableIndices())
    yI, yN = zeros(n), zeros(n)
    # Ideal and Nadir point estimation
    for (i, f_i) in enumerate(scalars)
        MOI.set(model.inner, MOI.ObjectiveFunction{typeof(f_i)}(), f_i)

        MOI.set(model.inner, MOI.ObjectiveSense(), sense)
        MOI.optimize!(model.inner)
        _, Y = _compute_objective(model, variables, f_i)
        yI[i] = Y
        MOI.set(
            model.inner,
            MOI.ObjectiveSense(),
            sense == MOI.MIN_SENSE ? MOI.MAX_SENSE : MOI.MIN_SENSE,
        )
        MOI.optimize!(model.inner)
        _, Y = _compute_objective(model, variables, f_i)
        yN[i] = Y
    end
    MOI.set(model.inner, MOI.ObjectiveSense(), sense)
    r = _compute_r(yI, yN)
    ϵ = _compute_ϵ(n, r)
    push!(L[1], _Box(yI, yN, 0.0))
    iter = 1
    while any(.!isempty.(L))
        i, k = _select_next_box(L, k)
        B = L[k][i]
        z = B.u
        w = _compute_w(z, yI)
        t_max = MOI.add_variable(model.inner)
        constraints = [
            MOI.Utilities.normalize_and_add_constraint(
                model.inner,
                t_max - (w[i] * (scalars[i] - yI[i])),
                MOI.GreaterThan(0.0),
            ) for i in 1:n
        ]
        MOI.set(
            model.inner,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            t_max + ϵ * sum(w[i] * (scalars[i] - yI[i]) for i in 1:n),
        )
        # print(model.inner)
        MOI.optimize!(model.inner)
        obj = MOI.get(model.inner, MOI.ObjectiveValue())
        MOI.delete.(model.inner, constraints)
        status = MOI.get(model.inner, MOI.TerminationStatus())
        if status == MOI.OPTIMAL
            X, Y = _compute_objective(model, variables, model.f)
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
    return MOI.OPTIMAL, solutions
end