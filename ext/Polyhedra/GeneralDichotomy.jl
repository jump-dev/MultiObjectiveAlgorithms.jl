#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

mutable struct Weight
    w::Vector{Float64}   # weight vector
    z::Float64           # value of the weighted objective
    adj_bnd::Vector{Int} # weight to boundaries adjacency
    adj_sol::Vector{Int} # weight to solution adjacency
    tested::Bool         # have the weights been tested?
    removed::Bool        # weights that are no longer part of the decomposition
end

struct CustomVec
    value::Vector{Float64}
    value_int::Vector{Int}

    function CustomVec(vec::Vector{Float64}, scaling::Real)
        return new(vec, round.(Int, scaling .* vec))
    end
end

Base.:(==)(a::CustomVec, b::CustomVec) = a.value_int == b.value_int

Base.hash(a::CustomVec, h::UInt64) = hash(a.value_int, h)

function MOA.minimize_multiobjective!(
    alg::MOA.GeneralDichotomy,
    model::MOA.Optimizer,
)
    n_obj = MOI.output_dimension(model.f)
    wnorm = 100.0
    start_time = time()
    weights = Weight[]
    # Initial extreme weights.
    for i in 1:n_obj
        w = zeros(Float64, n_obj)
        w[i] = wnorm
        adj_bnd = Int[-j for j in 1:n_obj if j != i]
        push!(weights, Weight(w, NaN, adj_bnd, [1], false, false))
    end
    status, solution = MOA._solve_weighted_sum(model, alg, weights[1].w)
    if !MOA._is_scalar_status_optimal(status)
        # Return immediately if no solution nor unbounded.
        return status, nothing
    end
    solutions = MOA.SolutionPoint[solution]
    # Weight update for the new solution
    for weight in weights
        weight.z = sum(weight.w .* solutions[1].y)
    end
    weights[1].tested = true
    # Prevent solution duplicates.
    # existing_sol maps a CustomVec of the solution to the solution index.
    existing_sol = Dict(CustomVec(solution.y, alg.scaling) => 1)
    n_removed = 0
    solution_limit = MOI.get(alg, MOA.SolutionLimit())
    iteration = 0
    while !(iteration >= alg.max_iter > 0) && length(solutions) < solution_limit
        iteration += 1
        # Look for a new solution by testing the extreme weights.
        found, new_sol_ind, wind, target_weight = false, 0, 1, 0
        while wind <= length(weights) && !found
            if weights[wind].tested || weights[wind].removed
                wind += 1
                continue
            end
            status, sol = MOA._solve_weighted_sum(model, alg, weights[wind].w)
            # TODO(odow): what if this solve fails?
            weights[wind].tested = true
            sol_z = sum(sol.y .* weights[wind].w)
            if !haskey(existing_sol, CustomVec(sol.y, alg.scaling))
                push!(solutions, sol)
                # Prepare new weight index set for the new solution's adjacency.
                new_sol_ind = length(solutions)
                existing_sol[CustomVec(sol.y, alg.scaling)] = new_sol_ind
                if sol_z < weights[wind].z
                    # Triggers weight set decomp. update.
                    found = true
                    target_weight = wind
                end
            end
            wind += 1
        end
        if !found
            break  # Terminate the search when no solution can be found.
        end
        polytope_sol = Set{Int}()
        equal_weights = Dict{CustomVec,Int}()
        for (wind, weight) in enumerate(weights)
            sol_z = sum(solutions[new_sol_ind].y .* weight.w)
            if sol_z < weight.z - alg.epsilon
                # Improved weighted value.
                if length(weight.adj_bnd) < n_obj
                    weight.removed = true
                    n_removed += 1
                else
                    weight.adj_sol = Int[new_sol_ind]
                    weight.z = sol_z
                end
                union!(polytope_sol, weight.adj_sol)
            elseif sol_z <= weight.z + alg.epsilon
                # Equal weighted value.
                push!(weight.adj_sol, new_sol_ind)
                union!(polytope_sol, weight.adj_sol)
                equal_weights[CustomVec(weight.w, alg.scaling)] = wind
            end
        end
        # Construction of the weight polytope for the new solution.
        h = Polyhedra.HyperPlane(ones(n_obj), wnorm)
        for i in 1:n_obj
            vec = zeros(n_obj)
            vec[i] = -1
            h = intersect(h, Polyhedra.HalfSpace(vec, 0))
        end
        polytope_sol_vec = collect(polytope_sol)
        for other_sol_ind in polytope_sol_vec
            vec = solutions[new_sol_ind].y - solutions[other_sol_ind].y
            h = intersect(h, Polyhedra.HalfSpace(vec, 0))
        end
        poly = Polyhedra.polyhedron(h)
        # Update of the extreme weights from the new polytope vertices.
        for idx in eachindex(Polyhedra.points(poly))
            w = get(poly, idx)
            weight_ind = get(equal_weights, CustomVec(w, alg.scaling), nothing)
            if weight_ind !== nothing
                # Update an existing extreme weight.
                weights[weight_ind].z = sum(w .* solutions[new_sol_ind].y)
                weights[weight_ind].tested = true
                weights[weight_ind].adj_sol = Int[new_sol_ind]
                if length(weights[weight_ind].adj_bnd) < n_obj-1
                    if !weights[weight_ind].removed
                        weights[weight_ind].removed = true
                        n_removed += 1
                    end
                else
                    weights[weight_ind].removed = false
                end
            else
                # Insert a new extreme weight.
                incidence = Polyhedra.incidenthalfspaceindices(poly, idx)
                new_weight = Weight(
                    w,
                    sum(w .* solutions[new_sol_ind].y),
                    Int[-elt.value for elt in incidence if elt.value <= n_obj],
                    Int[
                        polytope_sol_vec[elt.value-n_obj] for
                        elt in incidence if elt.value > n_obj
                    ],
                    false,
                    false,
                )
                push!(new_weight.adj_sol, new_sol_ind)
                push!(weights, new_weight)
            end
        end
        if 3*n_removed >= length(weights)
            filter!(w -> !w.removed, weights)
            n_removed = 0
        end
    end
    return status, solutions
end
