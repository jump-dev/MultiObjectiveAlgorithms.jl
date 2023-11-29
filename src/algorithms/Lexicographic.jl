#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

"""
    Lexicographic()

`Lexicographic()` implements a lexigographic algorithm that returns a single
point on the frontier, corresponding to solving each objective in order.

## Supported optimizer attributes

 * `MOI.TimeLimitSec()`: terminate if the time limit is exceeded and return the
   current best solutions.

 * `MOA.LexicographicAllPermutations()`: Controls whether to return the
   lexicographic solution for all permutations of the scalar objectives (when
   `true`), or only the solution corresponding to the lexicographic solution of
   the original objective function (when `false`).

 * `MOA.ObjectiveRelativeTolerance(index)`: after solving objective `index`, a
   constraint is added such that the relative degradation in the objective value
   of objective `index` is less than this tolerance.
"""
mutable struct Lexicographic <: AbstractAlgorithm
    rtol::Vector{Float64}
    all_permutations::Bool

    function Lexicographic(; all_permutations::Union{Nothing,Bool} = nothing)
        if all_permutations !== nothing
            @warn(
                "The `all_permutations` argument to `Lexicographic` was " *
                "removed in v1.0. Set the `MOA.LexicographicAllPermutations()` " *
                "option to `$all_permutations` instead.",
            )
        end
        return new(Float64[], default(LexicographicAllPermutations()))
    end
end

MOI.supports(::Lexicographic, ::ObjectiveRelativeTolerance) = true

function MOI.get(alg::Lexicographic, attr::ObjectiveRelativeTolerance)
    return get(alg.rtol, attr.index, default(alg, attr))
end

function MOI.set(alg::Lexicographic, attr::ObjectiveRelativeTolerance, value)
    for _ in (1+length(alg.rtol)):attr.index
        push!(alg.rtol, default(alg, attr))
    end
    alg.rtol[attr.index] = value
    return
end

MOI.supports(::Lexicographic, ::LexicographicAllPermutations) = true

function MOI.get(alg::Lexicographic, ::LexicographicAllPermutations)
    return alg.all_permutations
end

function MOI.set(alg::Lexicographic, ::LexicographicAllPermutations, val::Bool)
    alg.all_permutations = val
    return
end

function optimize_multiobjective!(algorithm::Lexicographic, model::Optimizer)
    start_time = time()
    sequence = 1:MOI.output_dimension(model.f)
    if !MOI.get(algorithm, LexicographicAllPermutations())
        return _solve_in_sequence(algorithm, model, sequence, start_time)
    end
    solutions = SolutionPoint[]
    status = MOI.OPTIMAL
    for sequence in Combinatorics.permutations(sequence)
        status, solution =
            _solve_in_sequence(algorithm, model, sequence, start_time)
        if !isempty(solution)
            push!(solutions, solution[1])
        end
        if !_is_scalar_status_optimal(status)
            break
        end
    end
    sense = MOI.get(model.inner, MOI.ObjectiveSense())
    return status, filter_nondominated(sense, solutions)
end

function _solve_in_sequence(
    algorithm::Lexicographic,
    model::Optimizer,
    sequence::AbstractVector{Int},
    start_time::Float64,
)
    variables = MOI.get(model.inner, MOI.ListOfVariableIndices())
    constraints = Any[]
    scalars = MOI.Utilities.eachscalar(model.f)
    solution = SolutionPoint[]
    status = MOI.OPTIMAL
    for i in sequence
        if _time_limit_exceeded(model, start_time)
            status = MOI.TIME_LIMIT
            break
        end
        f = scalars[i]
        MOI.set(model.inner, MOI.ObjectiveFunction{typeof(f)}(), f)
        MOI.optimize!(model.inner)
        status = MOI.get(model.inner, MOI.TerminationStatus())
        primal_status = MOI.get(model.inner, MOI.PrimalStatus())
        if _is_scalar_status_feasible_point(primal_status)
            X, Y = _compute_point(model, variables, model.f)
            solution = [SolutionPoint(X, Y)]
        end
        if !_is_scalar_status_optimal(status)
            break
        end
        X, Y = _compute_point(model, variables, f)
        rtol = MOI.get(algorithm, ObjectiveRelativeTolerance(i))
        set = if MOI.get(model.inner, MOI.ObjectiveSense()) == MOI.MIN_SENSE
            MOI.LessThan(Y + rtol * abs(Y))
        else
            MOI.GreaterThan(Y - rtol * abs(Y))
        end
        ci = MOI.Utilities.normalize_and_add_constraint(model, f, set)
        push!(constraints, ci)
    end
    for c in constraints
        MOI.delete(model, c)
    end
    return status, solution
end
