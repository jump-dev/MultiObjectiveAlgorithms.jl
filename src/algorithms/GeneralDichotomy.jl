#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

import Polyhedra

# handle floating point equality
struct CustomVec
    value::Vector{Float64}
    value_int::Vector{Int64}
    function CustomVec(vec::Vector{Float64}, scaling::Real)
        return new(vec, round.(Int64, scaling .* vec))
    end
end
Base.:(==)(a::CustomVec, b::CustomVec) = a.value_int == b.value_int
Base.hash(a::CustomVec, h::UInt64) = hash(a.value_int, h)

# data structure to store all information on the extreme weights
mutable struct Weight
    w::Vector{Float64} # weight vector
    z::Float64 # value of the weighted objective
    adj_bnd::Vector{Int64} # weight to boundaries adjacency
    adj_sol::Vector{Int64} # weight to solution adjacency
    tested::Bool # have the weights been tested ?
    removed::Bool # weights that are no longer part of the decomposition
    Weight() = new()
end

"""
    GeneralDichotomy()
    
- preprint: Samuel Buchet, Marianne Defresne. Efficient Enumeration of Supported Solutions for General Multi-Objective Optimization Problems. 2026. ⟨hal-05514317⟩

- original repository: https://forge.inrae.com/opteam/generaldichotomy

## Supported problem classes

This algorithm supports all problem classes.

## Supported optimizer attributes

 * `MOI.TimeLimitSec()`: terminate if the time limit is exceeded and return the
   list of current solutions.

 * `MOA.SolutionLimit()`: terminate once this many solutions have been found.

"""
mutable struct GeneralDichotomy <: AbstractAlgorithm
    solution_limit::Union{Nothing,Int}
    max_iter::Int64
    verbose::Int64
    weights::Array{Weight}
    epsilon::Float64
    scaling::Float64
    n_interm_weights::Int64
    n_call_solve::Int64
    GeneralDichotomy(precision::Int64) = new(nothing, 0, 0, Array{Weight}([]), 10.0^-precision, 10^precision, 0)
    GeneralDichotomy() = new(nothing, 0, 0, Array{Weight}([]), 10.0^-3, 10^3, 0) # default precision = 3
end

MOI.supports(::GeneralDichotomy, ::SolutionLimit) = true

function MOI.get(alg::GeneralDichotomy, attr::SolutionLimit)
    return something(alg.solution_limit, _default(alg, attr))
end

function MOI.set(alg::GeneralDichotomy, ::SolutionLimit, value)
    alg.solution_limit = value
    return
end

function _solve_weighted_sum(
    model::Optimizer,
    ::GeneralDichotomy,
    weight::Vector{Float64},
)
    f = _scalarise(model.f, weight)
    MOI.set(model.inner, MOI.ObjectiveFunction{typeof(f)}(), f)
    optimize_inner!(model)
    status = MOI.get(model.inner, MOI.TerminationStatus())
    if !_is_scalar_status_optimal(status)
        return status, nothing
    end
    variables = MOI.get(model.inner, MOI.ListOfVariableIndices())
    X, Y = _compute_point(model, variables, model.f)
    _log_subproblem_solve(model, Y)
    return status, SolutionPoint(X, Y)
end

function optimize_multiobjective!(
    alg::GeneralDichotomy,
    model::Optimizer
)

    if alg.verbose > 0
        println("starting the general dichotomy")
    end

    mul_sense = 1. # the weighted sum comparison is reversed when problems are maximized
    if MOI.get(model.inner, MOI.ObjectiveSense()) == MOI.MAX_SENSE
        mul_sense = -1.
    end

    n_obj = MOI.output_dimension(model.f)
    wnorm = 100.
    alg.n_call_solve = 0
    start_time = time() 
    
    alg.weights = Array{Weight}([])

    # initial extreme weights
    for i in 1:n_obj
        weight = Weight()
        weight.w = zeros(Float64, n_obj) 
        weight.w[i] = wnorm
        # weight to solution/boundaries adjacency
        weight.adj_bnd = Vector{Int64}([-j for j in 1:n_obj if j != i])
        weight.adj_sol = Vector{Int64}([1])
        weight.tested = false
        weight.removed = false
        push!(alg.weights, weight)
    end

    if alg.verbose > 0
        println("Initial weight tested:")
        println(alg.weights[1].w)
    end

    # initial solution
    status, solution = _solve_weighted_sum(model, alg, alg.weights[1].w)
    if !_is_scalar_status_optimal(status) # return immediately if no solution nor unbounded
        return status, nothing
    end
    solutions = [solution]

    if alg.verbose > 0
        println("Initial solution:")
        println(solution.y)
    end

    # weight update for the new solution
    for weight in alg.weights
        weight.z = sum(weight.w.*solutions[1].y)*mul_sense
    end
    alg.weights[1].tested = true

    # prevent solution duplicates
    existing_sol = Dict([CustomVec(solution.y, alg.scaling) => 1])

    n_removed = 0
    stop = false
    limit = MOI.get(alg, SolutionLimit())

    # list of solutions to consider when enumerating polytope vertices
    polytope_sol = Set{Int64}()

    iter_ind = 0

    # main loop
    while !stop

        if alg.verbose > 0
            println("\n\n")
            println("####################################################")
            println("                  Iteration ", iter_ind)
            println("####################################################")
            println("\n\n")
        end

        if (alg.max_iter > 0 && iter_ind >= alg.max_iter) || length(solutions) >= limit # early termination
            stop = true
            break
        end

        iter_ind += 1

        # look for a new solution by testing the extreme weights
        found = false
        new_sol_ind = 0
        wind = 1
        target_weight = 0
        while wind <= size(alg.weights, 1) && !found
            if alg.weights[wind].tested || alg.weights[wind].removed
                wind += 1
                continue
            end
            if alg.verbose > 0
                println("\n\noptimizing with weight ", alg.weights[wind].w)
            end
            status, sol = _solve_weighted_sum(model, alg, alg.weights[wind].w)
            alg.weights[wind].tested = true # this one has been tested
            sol_z = sum(sol.y .* alg.weights[wind].w)*mul_sense
            alg.n_call_solve += 1

            if alg.verbose > 0
                println("solution found: ", sol.y)
                println("comp ", sol_z, " ", alg.weights[wind].z)
            end

            # add the new solution if it was not previously discovered
            sol_ind = get(existing_sol, CustomVec(sol.y, alg.scaling), 0)
            if sol_ind == 0
                push!(solutions, sol)
                # prepare new weight index set for the new solution's adjacency
                new_sol_ind = solutions.size[1]
                push!(existing_sol, CustomVec(sol.y, alg.scaling) => new_sol_ind)
                if sol_z < alg.weights[wind].z # triggers weight set decomp. update
                    found = true
                    target_weight = wind
                    if alg.verbose > 0
                        println("found new improving solution")
                    end
                end
            elseif alg.verbose > 0
                println("the solutions was already discovered!")
            end
            wind += 1
        end

        # terminate the search when no solution can be found
        if !found
            stop = true
            break
        end

        empty!(polytope_sol)

        # record weights with the same weighted sum to prevent duplicates
        equal_weights = Dict{CustomVec, Int64}()

        # collect adjacent solutions for constructing the new polytope
        for wind in 1:alg.weights.size[1]
            weight = alg.weights[wind]
            sol_z = sum(solutions[new_sol_ind].y .* weight.w)*mul_sense
            if alg.verbose > 0
                println(" cmp ", sol_z, " vs ", weight.z)
            end
            if sol_z < weight.z-alg.epsilon
                if alg.verbose > 0
                    println("updating extreme weights", wind, weight.w)
                end
                if weight.adj_bnd.size[1] < n_obj # improved weighted value
                    weight.removed = true
                else
                    weight.adj_sol = Vector{Int64}([new_sol_ind])
                    weight.z = sol_z
                end
                union!(polytope_sol, weight.adj_sol)
            elseif sol_z <= weight.z+alg.epsilon # equal weighted value
                if alg.verbose > 0
                    println("equal weight found", wind, weight.w)
                end
                push!(weight.adj_sol, new_sol_ind)
                union!(polytope_sol, weight.adj_sol)
                equal_weights[CustomVec(weight.w, alg.scaling)] = wind # add equal weight
            end
        end

        if alg.verbose > 0
            println("\n\n")
            println("Adjacent solutions for the new polytope: ", polytope_sol)
        end

        # construction of the weight polytope for the new solution
        h = Polyhedra.HyperPlane( ones(n_obj), wnorm) # normalized weights
        for i in 1:n_obj # non-negativity
            vec = zeros(n_obj)
            vec[i] = -1
            h = h ∩ Polyhedra.HalfSpace(vec, 0)
        end
        polytope_sol = collect(polytope_sol)
        for other_sol_ind in polytope_sol # scalarizations inequality
            vec = (solutions[new_sol_ind].y - solutions[other_sol_ind].y).*mul_sense
            h = h ∩ Polyhedra.HalfSpace(vec, 0)
        end
        poly = Polyhedra.polyhedron(h)

        if alg.verbose > 1
            println("Polyhedron:")
            println(poly)
            println("\n\n")
        end

        if alg.verbose > 1
            println("Polytope extreme weights:")
            for idx in eachindex(Polyhedra.points(poly))
                p = get(poly, idx)
                println(p)
            end
            println("\n")
        end

        # update of the extreme weights from the new polytope vertices
        for idx in eachindex(Polyhedra.points(poly))
            w = get(poly, idx)
            weight_ind = get(equal_weights, CustomVec(w, alg.scaling), 0)
            if weight_ind > 0 # update an existing extreme weight
                alg.weights[weight_ind].z = sum(w .* solutions[new_sol_ind].y)*mul_sense
                alg.weights[weight_ind].tested = true
                alg.weights[weight_ind].adj_sol = Vector{Int64}([new_sol_ind])
                if alg.weights[weight_ind].adj_bnd.size[1] < n_obj-1
                    if !alg.weights[weight_ind].removed
                        alg.weights[weight_ind].removed = true
                        n_removed += 1
                    end
                else
                    alg.weights[weight_ind].removed = false
                end
            else # insert a new extreme weight
                weight_ind = alg.weights.size[1]+1
                # push!(existing_weights, p => weight_ind)
                new_weight = Weight()
                new_weight.tested = false
                new_weight.removed = false
                new_weight.w = w
                new_weight.z = sum(w .* solutions[new_sol_ind].y)*mul_sense
                incidence = Polyhedra.incidenthalfspaceindices(poly, idx)
                new_weight.adj_bnd = Vector{Int64}([-elt.value for elt in incidence if elt.value <= n_obj])
                new_weight.adj_sol = Vector{Int64}([ polytope_sol[elt.value-n_obj] for elt in incidence if elt.value > n_obj])
                push!(new_weight.adj_sol, new_sol_ind)
                push!(alg.weights, new_weight)
            end
        end

        # iteration summary
        if alg.verbose > 0
            println("Weights: ")
            w_ind = 0
            for weight in alg.weights
                print("w^", w_ind, ": ")
                print(weight.w)
                print(" z=", weight.z)
                print(", ", weight.adj_sol)
                print(", ", weight.adj_bnd)
                print(" -> ", weight.tested)
                println(" ", weight.removed)
                w_ind += 1
            end
            println("Solutions:")
            for sol in solutions
                println(sol.y)
            end
        end

        # clean removed weights
        if 3*n_removed >= alg.weights.size[1]
            alg.n_interm_weights += n_removed
            alg.weights = Array{Weight}([weight for weight in alg.weights if !weight.removed])
            n_removed = 0
        end
    end

    # final cleaning
    alg.n_interm_weights += n_removed
    alg.weights = Array{Weight}([weight for weight in alg.weights if !weight.removed])

    return status, solutions
end

