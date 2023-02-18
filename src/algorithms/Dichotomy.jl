#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

"""
    Dichotomy()

`Dichotomy` implements the algorithm of:

...
"""
mutable struct Dichotomy <: AbstractAlgorithm end

function optimize_multiobjective!(algorithm::Dichotomy, model::Optimizer)
    if MOI.output_dimension(model.f) != 2
        error("EpsilonConstraint requires exactly two objectives")
    end
    solutions = SolutionPoint[]
    L = Tuple{Int,Int}[]
    E = Tuple{Int,Int}[]
    y1, y2 = zeros(2), zeros(2)
    variables = MOI.get(model.inner, MOI.ListOfVariableIndices())
    f1, f2 = MOI.Utilities.scalarize(model.f)
    MOI.set(model.inner, MOI.ObjectiveFunction{typeof(f1)}(), f1)
    MOI.optimize!(model.inner)
    status = MOI.get(model.inner, MOI.TerminationStatus())
    if !_is_scalar_status_optimal(status)
        return status, nothing
    end
    _, y1[1] = _compute_point(model, variables, f1)
    MOI.set(model.inner, MOI.ObjectiveFunction{typeof(f2)}(), f2)
    y1_constraint = MOI.add_constraint(model.inner, f1, MOI.EqualTo(y1[1]))
    MOI.optimize!(model.inner)
    x1, y1[2] = _compute_point(model, variables, f2)
    MOI.delete(model.inner, y1_constraint)
    k = 1
    MOI.set(model.inner, MOI.ObjectiveFunction{typeof(f2)}(), f2)
    MOI.optimize!(model.inner)
    status = MOI.get(model.inner, MOI.TerminationStatus())
    if !_is_scalar_status_optimal(status)
        return status, nothing
    end
    _, y2[2] = _compute_point(model, variables, f2)
    MOI.set(model.inner, MOI.ObjectiveFunction{typeof(f1)}(), f1)
    y2_constraint = MOI.add_constraint(model.inner, f2, MOI.EqualTo(y2[2]))
    MOI.optimize!(model.inner)
    x2, y2[1] = _compute_point(model, variables, f1)
    MOI.delete(model.inner, y2_constraint)
    if y1 == y2
        return MOI.OPTIMAL, [SolutionPoint(x1, y1)]
    else
        push!(solutions, SolutionPoint(x1, y1), SolutionPoint(x2, y2))
        push!(L, (1, 2))
        k += 1
        while !isempty(L)
            r, s = pop!(L)
            yr, ys = solutions[r].y, solutions[s].y
            a1 = abs(ys[2] - yr[2])
            a2 = abs(ys[1] - yr[1])
            MOI.set(
                model.inner,
                MOI.ObjectiveFunction{typeof(f1 + f2)}(),
                a1 * f1 + a2 * f2,
            )
            MOI.optimize!(model.inner)
            x_new, y_new = _compute_point(model, variables, model.f)
            if all(isapprox.(y_new, yr; atol = 1e-6)) ||
               all(isapprox.(y_new, ys; atol = 1e-6))
                push!(E, (r, s))
            else
                push!(solutions, SolutionPoint(x_new, y_new))
                k += 1
                append!(L, [(r, k), (k, s)])
            end
        end
    end
    return MOI.OPTIMAL, solutions
end
