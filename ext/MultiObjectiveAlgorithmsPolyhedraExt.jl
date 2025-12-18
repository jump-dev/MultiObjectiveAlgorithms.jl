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

function _distance(w̄, b̄, δ_OPS_optimizer)
    y = MOI.get(δ_OPS_optimizer, MOI.ListOfVariableIndices())
    MOI.set(
        δ_OPS_optimizer,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(w̄, y), 0.0),
    )
    MOI.set(δ_OPS_optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(δ_OPS_optimizer)
    return b̄ - MOI.get(δ_OPS_optimizer, MOI.ObjectiveValue())
end

function _select_next_halfspace(H, δ_OPS_optimizer)
    distances = [_distance(w, b, δ_OPS_optimizer) for (w, b) in H]
    index = argmax(distances)
    w, b = H[index]
    return distances[index], w, b
end

function MOA.minimize_multiobjective!(
    algorithm::MOA.Sandwiching,
    model::MOA.Optimizer,
    inner::MOI.ModelLike,
    f::MOI.AbstractVectorFunction,
)
    @assert MOI.get(inner, MOI.ObjectiveSense()) == MOI.MIN_SENSE
    solutions = Dict{Vector{Float64},Dict{MOI.VariableIndex,Float64}}()
    variables = MOI.get(inner, MOI.ListOfVariableIndices())
    n = MOI.output_dimension(f)
    scalars = MOI.Utilities.scalarize(f)
    status = MOI.OPTIMAL
    δ_OPS_optimizer = MOI.instantiate(model.optimizer_factory)
    if MOI.supports(δ_OPS_optimizer, MOI.Silent())
        MOI.set(δ_OPS_optimizer, MOI.Silent(), true)
    end
    y = MOI.add_variables(δ_OPS_optimizer, n)
    anchors = Dict{Vector{Float64},Dict{MOI.VariableIndex,Float64}}()
    yI, yUB = zeros(n), zeros(n)
    for (i, f_i) in enumerate(scalars)
        MOI.set(inner, MOI.ObjectiveFunction{typeof(f_i)}(), f_i)
        MOA.optimize_inner!(model)
        status = MOI.get(inner, MOI.TerminationStatus())
        if !MOA._is_scalar_status_optimal(model)
            return status, nothing
        end
        X, Y = MOA._compute_point(model, variables, f)
        model.ideal_point[i] = Y[i]
        yI[i] = Y[i]
        anchors[Y] = X
        MOI.set(inner, MOI.ObjectiveSense(), MOI.MAX_SENSE)
        MOA.optimize_inner!(model)
        status = MOI.get(inner, MOI.TerminationStatus())
        if !MOA._is_scalar_status_optimal(model)
            MOA._warn_on_nonfinite_anti_ideal(algorithm, MOI.MIN_SENSE, i)
            return status, nothing
        end
        _, Y = MOA._compute_point(model, variables, f_i)
        yUB[i] = Y
        MOI.set(inner, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        e_i = Float64.(1:n .== i)
        MOI.add_constraint(
            δ_OPS_optimizer,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(e_i, y), 0.0),
            MOI.GreaterThan(yI[i]),
        )
        MOI.add_constraint(
            δ_OPS_optimizer,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(e_i, y), 0.0),
            MOI.LessThan(yUB[i]),
        )
    end
    IPS = [yUB, keys(anchors)...]
    merge!(solutions, anchors)
    u = MOI.add_variables(inner, n)
    u_constraints = [ # u_i >= 0 for all i = 1:n
        MOI.add_constraint(inner, u_i, MOI.GreaterThan{Float64}(0))
        for u_i in u
    ]
    f_constraints = [ # f_i + u_i <= yUB_i for all i = 1:n
        MOI.Utilities.normalize_and_add_constraint(
            inner,
            scalars[i] + u[i],
            MOI.LessThan(yUB[i]),
        ) for i in 1:n
    ]
    H = _halfspaces(IPS)
    count = 0
    while !isempty(H)
        ret = MOA._check_premature_termination(model)
        if ret !== nothing
            status = ret
            break
        end
        count += 1
        δ, w, b = _select_next_halfspace(H, δ_OPS_optimizer)
        if δ - 1e-3 <= algorithm.precision # added some convergence tolerance
            break
        end
        # would not terminate when precision is set to 0
        new_f = sum(w[i] * (scalars[i] + u[i]) for i in 1:n) # w' * (f(x) + u)
        MOI.set(inner, MOI.ObjectiveFunction{typeof(new_f)}(), new_f)
        MOA.optimize_inner!(model)
        status = MOI.get(inner, MOI.TerminationStatus())
        if !MOA._is_scalar_status_optimal(model)
            return status, nothing
        end
        β̄ = MOI.get(inner, MOI.ObjectiveValue())
        X, Y = MOA._compute_point(model, variables, f)
        solutions[Y] = X
        MOI.add_constraint(
            δ_OPS_optimizer,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(w, y), 0.0),
            MOI.GreaterThan(β̄),
        )
        IPS = push!(IPS, Y)
        H = _halfspaces(IPS)
    end
    MOI.delete.(inner, f_constraints)
    MOI.delete.(inner, u_constraints)
    MOI.delete.(inner, u)
    return status, [MOA.SolutionPoint(X, Y) for (Y, X) in solutions]
end

end  # module MultiObjectiveAlgorithmsPolyhedraExt
