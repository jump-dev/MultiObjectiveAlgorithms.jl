#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

module MultiObjectiveAlgorithmsPolyhedraExt

import MathOptInterface as MOI
import MultiObjectiveAlgorithms as MOA
import Polyhedra

function _halfspaces(IPS::Vector{Vector{Float64}})
    V = Polyhedra.vrep(IPS)
    H = Polyhedra.halfspaces(Polyhedra.doubledescription(V))
    return [(-H_i.a, -H_i.β) for H_i in H]
end

function _compute_anchors(model::MOA.Optimizer)
    anchors = Dict{Vector{Float64},Dict{MOI.VariableIndex,Float64}}()
    n = MOI.output_dimension(model.f)
    scalars = MOI.Utilities.scalarize(model.f)
    variables = MOI.get(model.inner, MOI.ListOfVariableIndices())
    yI, yUB = zeros(n), zeros(n)
    for (i, f_i) in enumerate(scalars)
        MOI.set(model.inner, MOI.ObjectiveFunction{typeof(f_i)}(), f_i)
        MOI.optimize!(model.inner)
        # status check
        X, Y = MOA._compute_point(model, variables, model.f)
        model.ideal_point[i] = Y[i]
        yI[i] = Y[i]
        anchors[Y] = X
        MOI.set(model.inner, MOI.ObjectiveSense(), MOI.MAX_SENSE)
        MOI.optimize!(model.inner)
        # status check
        _, Y = MOA._compute_point(model, variables, f_i)
        yUB[i] = Y
        MOI.set(model.inner, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    end

    return yI, yUB, anchors
end

function _distance(w̄, b̄, OPS, model)
    n = MOI.output_dimension(model.f)
    optimizer = typeof(model.inner.optimizer)
    δ_optimizer = optimizer()
    MOI.set(δ_optimizer, MOI.Silent(), true)
    x = MOI.add_variables(δ_optimizer, n)
    for (w, b) in OPS
        MOI.add_constraint(
            δ_optimizer,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(w, x), 0.0),
            MOI.GreaterThan(b),
        )
    end
    MOI.set(
        δ_optimizer,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(w̄, x), 0.0),
    )
    MOI.set(δ_optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(δ_optimizer)
    return b̄ - MOI.get(δ_optimizer, MOI.ObjectiveValue())
end

function _select_next_halfspace(H, OPS, model)
    distances = [_distance(w, b, OPS, model) for (w, b) in H]
    @info "Distances: $(Dict(zip(H, distances)))"
    index = argmax(distances)
    w, b = H[index]
    return distances[index], w, b
end

function MOA.minimize_multiobjective!(
    algorithm::MOA.Sandwiching,
    model::MOA.Optimizer,
)
    @assert MOI.get(model.inner, MOI.ObjectiveSense()) == MOI.MIN_SENSE
    ε = algorithm.precision
    start_time = time()
    solutions = Dict{Vector{Float64},Dict{MOI.VariableIndex,Float64}}()
    variables = MOI.get(model.inner, MOI.ListOfVariableIndices())
    n = MOI.output_dimension(model.f)
    scalars = MOI.Utilities.scalarize(model.f)
    yI, yUB, anchors = _compute_anchors(model)
    merge!(solutions, anchors)
    @info "yI: $(yI)"
    @info "yUB: $(yUB)"
    IPS = [yUB, keys(anchors)...]
    OPS = Tuple{Vector{Float64},Float64}[]
    for i in 1:n
        e_i = Float64.(1:n .== i)
        push!(OPS, (e_i, yI[i])) # e_i' * y >= yI_i
        push!(OPS, (-e_i, -yUB[i])) # -e_i' * y >= -yUB_i ⟹ e_i' * y <= yUB_i
    end
    @info "IPS: $(IPS)"
    @info "OPS: $(OPS)"
    u = MOI.add_variables(model.inner, n)
    u_constraints = [ # u_i >= 0 for all i = 1:n
        MOI.add_constraint(model.inner, u_i, MOI.GreaterThan{Float64}(0))
        for u_i in u
    ]
    f_constraints = [ # f_i + u_i <= yUB_i for all i = 1:n
        MOI.Utilities.normalize_and_add_constraint(
            model.inner,
            scalars[i] + u[i],
            MOI.LessThan(yUB[i]),
        ) for i in 1:n
    ]
    H = _halfspaces(IPS)
    count = 0
    while !isempty(H)
        count += 1
        @info "-- Iteration #$(count) --"
        @info "HalfSpaces: $(H)"
        δ, w, b = _select_next_halfspace(H, OPS, model)
        @info "Selected halfspace: w: $(w), b: $(b)"
        @info "δ: $(δ)"
        if δ - 1e-3 <= ε # added some convergence tolerance
            break
        end
        # would not terminate when precision is set to 0
        new_f = sum(w[i] * (scalars[i] + u[i]) for i in 1:n) # w' * (f(x) + u)
        MOI.set(model.inner, MOI.ObjectiveFunction{typeof(new_f)}(), new_f)
        MOI.optimize!(model.inner)
        β̄ = MOI.get(model.inner, MOI.ObjectiveValue())
        @info "β̄: $(β̄)"
        X, Y = MOA._compute_point(model, variables, model.f)
        @info "Y: $(Y)"
        solutions[Y] = X
        push!(OPS, (w, β̄))
        @info "Added halfspace w: $(w), b: $(β̄) to OPS"
        IPS = push!(IPS, Y)
        @info "IPS: $(IPS)"
        @info "OPS: $(OPS)"
        H = _halfspaces(IPS)
        if count == 10
            break
        end
    end
    MOI.delete.(model.inner, f_constraints)
    MOI.delete.(model.inner, u_constraints)
    MOI.delete.(model.inner, u)
    return MOI.OPTIMAL, [MOA.SolutionPoint(X, Y) for (Y, X) in solutions]
end

end  # module MultiObjectiveAlgorithmsPolyhedraExt
