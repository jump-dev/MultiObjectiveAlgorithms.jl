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

_round(x::Vector{Float64}; atol::Float64) = round.(Int, x ./ atol)

function MOA.minimize_multiobjective!(
    alg::MOA.GeneralDichotomy,
    model::MOA.Optimizer,
)
    # Some constants. These could be converted into algorithm options.
    # - atol: the absolute tolerance used to compare solutions in objective space
    # - wnorm: ???
    atol, wnorm = 1e-6, 1e2
    # Storage we need for the algorithm.
    weights, solutions = Weight[], MOA.SolutionPoint[]
    n_obj = MOI.output_dimension(model.f)
    # First, minimize the first objective to obtain a primal feasible point.
    w = zeros(Float64, n_obj)
    w[1] = 1.0
    status, solution = MOA._solve_weighted_sum(model, alg, w)
    if solution === nothing
        return status, nothing
    end
    push!(solutions, solution)
    # Initialize the weights. There is one weight vector for each objective, and
    # the weight is set to wnorm for each objective. We use the current solution
    # obtained by minimizing the 1st objective as the reference.
    for i in 1:n_obj
        w = zeros(Float64, n_obj)
        w[i] = wnorm
        z = w' * solution.y
        adj_bnd = Int[-j for j in 1:n_obj if j != i]
        push!(weights, Weight(w, z, adj_bnd, [1], i == 1, false))
    end
    # Prevent solution duplicates: existing_sol maps an rounded objective vector
    # to its index in `solutions::Vector{MOA.SolutionPoint}`.
    existing_sol = Dict(_round(solution.y; atol) => 1)
    n_removed = 0
    while length(solutions) < MOI.get(alg, MOA.SolutionLimit())
        # Look for a new solution by testing the extreme weights.
        improving_solution = false
        for (i, weight) in enumerate(weights)
            if weight.tested || weight.removed
                continue
            end
            status, sol = MOA._solve_weighted_sum(model, alg, weight.w)
            # TODO(odow): what if this solve fails?
            weight.tested = true
            if !haskey(existing_sol, _round(sol.y; atol))
                push!(solutions, sol)
                # Prepare new weight index set for the new solution's adjacency.
                existing_sol[_round(sol.y; atol)] = length(solutions)
                if weight.w' * sol.y < weight.z
                    improving_solution = true
                    break
                end
            end
        end
        if !improving_solution
            break  # Terminate the search when no new solution can be found.
        end
        new_sol, new_sol_ind = last(solutions), length(solutions)
        polytope_sol, equal_weights = Set{Int}(), Dict{Vector{Int},Int}()
        for (i, weight) in enumerate(weights)
            sol_z = weight.w' * new_sol.y
            if sol_z < weight.z - atol
                # The new solution is strictly better than the previous.
                if length(weight.adj_bnd) < n_obj
                    weight.removed = true
                    n_removed += 1
                else
                    weight.adj_sol = Int[new_sol_ind]
                    weight.z = sol_z
                end
                union!(polytope_sol, weight.adj_sol)
            elseif sol_z <= weight.z + atol
                # The new solution is equal in value to the previous.
                push!(weight.adj_sol, new_sol_ind)
                union!(polytope_sol, weight.adj_sol)
                equal_weights[_round(weight.w; atol)] = i
            end
        end
        # Construction of the weight polytope for the new solution.
        h = Polyhedra.HyperPlane(ones(n_obj), wnorm)
        for i in 1:n_obj
            vec = zeros(Float64, n_obj)
            vec[i] = -1.0
            h = intersect(h, Polyhedra.HalfSpace(vec, 0))
        end
        # Convert the set of polytope solutions into a vector. It's important
        # that iteration is ordered because we're going to rely on this later.
        polytope_sol_vec = collect(polytope_sol)
        for i in polytope_sol_vec
            h = intersect(h, Polyhedra.HalfSpace(new_sol.y - solutions[i].y, 0))
        end
        poly = Polyhedra.polyhedron(h)
        # Update of the extreme weights from the new polytope vertices.
        for idx in eachindex(Polyhedra.points(poly))
            w = get(poly, idx)
            z = w' * new_sol.y
            if (i = get(equal_weights, _round(w; atol), nothing)) !== nothing
                # Update an existing extreme weight.
                weights[i].z = z
                weights[i].tested = true
                weights[i].adj_sol = Int[new_sol_ind]
                if length(weights[i].adj_bnd) < n_obj - 1
                    if !weights[i].removed
                        weights[i].removed = true
                        n_removed += 1
                    end
                else
                    weights[i].removed = false
                end
            else
                # Insert a new extreme weight.
                adj_bnd, adj_sol = Int[], Int[]
                for elt in Polyhedra.incidenthalfspaceindices(poly, idx)
                    if elt.value <= n_obj
                        push!(adj_bnd, -elt.value)
                    else
                        push!(adj_sol, polytope_sol_vec[elt.value-n_obj])
                    end
                end
                push!(adj_sol, new_sol_ind)
                push!(weights, Weight(w, z, adj_bnd, adj_sol, false, false))
            end
        end
        # This is a heuristic: filter the weights if approximately 1/3 of them
        # have been removed.
        if n_removed >= length(weights) / 3
            filter!(w -> !w.removed, weights)
            n_removed = 0
        end
    end
    return status, solutions
end
