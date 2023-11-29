#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

"""
    Chalmet()

`Chalmet` implements the algorithm of:

Chalmet, L.G., and Lemonidis, L., and Elzinga, D.J. (1986). An algorithm for the
bi-criterion integer programming problem. European Journal of Operational
Research. 25(2), 292-300

## Supported optimizer attributes

 * `MOI.TimeLimitSec()`: terminate if the time limit is exceeded and return the
   list of current solutions.
"""
mutable struct Chalmet <: AbstractAlgorithm end

function _solve_constrained_model(
    model::Optimizer,
    ::Chalmet,
    rhs::Vector{Float64},
)
    f = MOI.Utilities.scalarize(model.f)
    g = sum(1.0 * fi for fi in f)
    MOI.set(model.inner, MOI.ObjectiveFunction{typeof(g)}(), g)
    sets = MOI.LessThan.(rhs .- 1)
    c = MOI.Utilities.normalize_and_add_constraint.(model.inner, f, sets)
    MOI.optimize!(model.inner)
    MOI.delete.(model, c)
    status = MOI.get(model.inner, MOI.TerminationStatus())
    if !_is_scalar_status_optimal(status)
        return status, nothing
    end
    variables = MOI.get(model.inner, MOI.ListOfVariableIndices())
    X, Y = _compute_point(model, variables, model.f)
    return status, SolutionPoint(X, Y)
end

function optimize_multiobjective!(algorithm::Chalmet, model::Optimizer)
    start_time = time()
    if MOI.output_dimension(model.f) != 2
        error("Chalmet requires exactly two objectives")
    end
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
    solutions = SolutionPoint[]
    E = Tuple{Int,Int}[]
    Q = Tuple{Int,Int}[]
    variables = MOI.get(model.inner, MOI.ListOfVariableIndices())
    f1, f2 = MOI.Utilities.scalarize(model.f)
    y1, y2 = zeros(2), zeros(2)
    MOI.set(model.inner, MOI.ObjectiveFunction{typeof(f2)}(), f2)
    MOI.optimize!(model.inner)
    status = MOI.get(model.inner, MOI.TerminationStatus())
    if !_is_scalar_status_optimal(status)
        return status, nothing
    end
    _, y1[2] = _compute_point(model, variables, f2)
    MOI.set(model.inner, MOI.ObjectiveFunction{typeof(f1)}(), f1)
    y1_constraint = MOI.Utilities.normalize_and_add_constraint(
        model.inner,
        f2,
        MOI.LessThan(y1[2]),
    )
    MOI.optimize!(model.inner)
    x1, y1[1] = _compute_point(model, variables, f1)
    MOI.delete(model.inner, y1_constraint)
    push!(solutions, SolutionPoint(x1, y1))
    MOI.set(model.inner, MOI.ObjectiveFunction{typeof(f1)}(), f1)
    MOI.optimize!(model.inner)
    status = MOI.get(model.inner, MOI.TerminationStatus())
    if !_is_scalar_status_optimal(status)
        return status, nothing
    end
    _, y2[1] = _compute_point(model, variables, f1)
    if y2[1] ≈ solutions[1].y[1]
        return MOI.OPTIMAL, [solutions]
    end
    MOI.set(model.inner, MOI.ObjectiveFunction{typeof(f2)}(), f2)
    y2_constraint = MOI.Utilities.normalize_and_add_constraint(
        model.inner,
        f1,
        MOI.LessThan(y2[1]),
    )
    MOI.optimize!(model.inner)
    x2, y2[2] = _compute_point(model, variables, f2)
    MOI.delete(model.inner, y2_constraint)
    push!(solutions, SolutionPoint(x2, y2))
    push!(Q, (1, 2))
    t = 3
    while !isempty(Q)
        if _time_limit_exceeded(model, start_time)
            return MOI.TIME_LIMIT, solutions
        end
        r, s = pop!(Q)
        yr, ys = solutions[r].y, solutions[s].y
        rhs = [max(yr[1], ys[1]), max(yr[2], ys[2])]
        status, solution = _solve_constrained_model(model, algorithm, rhs)
        if !_is_scalar_status_optimal(status)
            push!(E, (r, s))
            continue
        end
        push!(solutions, solution)
        append!(Q, [(r, t), (t, s)])
        t += 1
    end
    return MOI.OPTIMAL, solutions
end
