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

function _sequences(sequence, perm)
    if !something(perm, _default(LexicographicAllPermutations()))
        return [sequence]
    end
    return Combinatorics.permutations(sequence)
end

function minimize_multiobjective!(
    algorithm::Lexicographic,
    model::Optimizer,
    inner::MOI.ModelLike,
    f::MOI.AbstractVectorFunction,
)
    dim = MOI.output_dimension(model.f)
    perm = MOI.get(algorithm, LexicographicAllPermutations())
    if perm === nothing && dim >= 5
        @warn(
            """
            The `MOA.Lexicographic` algorithm has been called with the default
            option for `MOA.LexicographicAllPermutations()`.

            Because there are $dim objectives, the algorithm will solve all
            $(dim)! = $(factorial(dim)) permutations of the objectives.

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
    status, solutions = MOI.OPTIMAL, SolutionPoint[]
    variables = MOI.get(inner, MOI.ListOfVariableIndices())
    scalars = MOI.Utilities.eachscalar(f)
    set_inf = MOI.LessThan(Inf)
    constraints = map(scalars) do fi
        return MOI.Utilities.normalize_and_add_constraint(model, fi, set_inf)
    end
    constants = MOI.constant(f, Float64)
    for sequence in _sequences(1:dim, perm)
        solution = nothing
        for i in sequence
            if (ret = _check_premature_termination(model)) !== nothing
                status = ret
                break
            end
            fi = scalars[i]
            MOI.set(inner, MOI.ObjectiveFunction{typeof(fi)}(), fi)
            optimize_inner!(model)
            status = MOI.get(inner, MOI.TerminationStatus())
            primal_status = MOI.get(inner, MOI.PrimalStatus())
            if _is_scalar_status_feasible_point(primal_status)
                X, Y = _compute_point(model, variables, f)
                _log_subproblem_solve(model, Y)
                solution = SolutionPoint(X, Y)
                get_X = Base.Fix1(getindex, X)
                Yi = MOI.Utilities.eval_variables(get_X, model, fi)
                rtol = MOI.get(algorithm, ObjectiveRelativeTolerance(i))
                set = MOI.LessThan(Yi - constants[i] + rtol * abs(Yi))
                MOI.set(model, MOI.ConstraintSet(), constraints[i], set)
            end
            if !_is_scalar_status_optimal(status)
                break
            end
        end
        if solution !== nothing
            push!(solutions, solution)
        end
        if !_is_scalar_status_optimal(status)
            break
        end
        for c in constraints
            MOI.set(model, MOI.ConstraintSet(), c, set_inf)
        end
    end
    for c in constraints
        MOI.delete(model, c)
    end
    return status, solutions
end
