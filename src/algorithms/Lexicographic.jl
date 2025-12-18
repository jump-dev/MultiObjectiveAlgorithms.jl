#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

"""
    Lexicographic()

`Lexicographic()` implements a lexigographic algorithm that returns a single
point on the frontier, corresponding to solving each objective in order.

## Supported problem classes

This algorithm supports all problem classes.

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
    all_permutations::Union{Nothing,Bool}

    function Lexicographic(; all_permutations::Union{Nothing,Bool} = nothing)
        if all_permutations !== nothing
            @warn(
                "The `all_permutations` argument to `Lexicographic` was " *
                "removed in v1.0. Set the `MOA.LexicographicAllPermutations()` " *
                "option to `$all_permutations` instead.",
            )
        end
        return new(Float64[], nothing)
    end
end

MOI.supports(::Lexicographic, ::ObjectiveRelativeTolerance) = true

function MOI.get(alg::Lexicographic, attr::ObjectiveRelativeTolerance)
    return get(alg.rtol, attr.index, _default(alg, attr))
end

function MOI.set(alg::Lexicographic, attr::ObjectiveRelativeTolerance, value)
    for _ in (1+length(alg.rtol)):attr.index
        push!(alg.rtol, _default(alg, attr))
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
    return optimize_multiobjective!(algorithm, model, model.inner, model.f)
end

function optimize_multiobjective!(
    algorithm::Lexicographic,
    model::Optimizer,
    inner::MOI.ModelLike,
    f::MOI.AbstractVectorFunction,
)
    sequence = 1:MOI.output_dimension(model.f)
    perm = MOI.get(algorithm, LexicographicAllPermutations())
    if !something(perm, _default(LexicographicAllPermutations()))
        return _solve_in_sequence(algorithm, model, inner, f, sequence)
    end
    if perm === nothing && length(sequence) >= 5
        o, n = length(sequence), factorial(length(sequence))
        @warn(
            """
            The `MOA.Lexicographic` algorithm has been called with the default
            option for `MOA.LexicographicAllPermutations()`.

            Because there are $o objectives, the algorithm will solve all
            $(o)! = $n permutations of the objectives.

            To solve only a single sequence corresponding to the lexicographic
            order of the objective function, set `MOA.LexicographicAllPermutations()`
            to `false`:
            ```julia
            set_attribute(model, MOA.LexicographicAllPermutations(), false)
            ```

            To disable this warning, explicitly opt-in to all permutations by
            setting it to `true`:
            ```julia
            set_attribute(model, MOA.LexicographicAllPermutations(), true)
            ```
            """,
        )
    end
    solutions = SolutionPoint[]
    status = MOI.OPTIMAL
    for sequence in Combinatorics.permutations(sequence)
        status, solution =
            _solve_in_sequence(algorithm, model, inner, f, sequence)
        if !isempty(solution)
            push!(solutions, solution[1])
        end
        if !_is_scalar_status_optimal(status)
            break
        end
    end
    return status, solutions
end

function _solve_in_sequence(
    algorithm::Lexicographic,
    model::Optimizer,
    inner::MOI.ModelLike,
    f::MOI.AbstractVectorFunction,
    sequence::AbstractVector{Int},
)
    variables = MOI.get(inner, MOI.ListOfVariableIndices())
    constraints = Any[]
    scalars = MOI.Utilities.eachscalar(f)
    solution = SolutionPoint[]
    status = MOI.OPTIMAL
    for i in sequence
        if (ret = _check_premature_termination(model)) !== nothing
            status = ret
            break
        end
        scalar_f = scalars[i]
        MOI.set(inner, MOI.ObjectiveFunction{typeof(scalar_f)}(), scalar_f)
        optimize_inner!(model)
        status = MOI.get(inner, MOI.TerminationStatus())
        primal_status = MOI.get(inner, MOI.PrimalStatus())
        if _is_scalar_status_feasible_point(primal_status)
            X, Y = _compute_point(model, variables, f)
            _log_subproblem_solve(model, Y)
            solution = [SolutionPoint(X, Y)]
        end
        if !_is_scalar_status_optimal(status)
            break
        end
        X, Y = _compute_point(model, variables, scalar_f)
        rtol = MOI.get(algorithm, ObjectiveRelativeTolerance(i))
        set = if MOI.get(inner, MOI.ObjectiveSense()) == MOI.MIN_SENSE
            MOI.LessThan(Y + rtol * abs(Y))
        else
            MOI.GreaterThan(Y - rtol * abs(Y))
        end
        ci = MOI.Utilities.normalize_and_add_constraint(model, scalar_f, set)
        push!(constraints, ci)
    end
    for c in constraints
        MOI.delete(model, c)
    end
    return status, solution
end
