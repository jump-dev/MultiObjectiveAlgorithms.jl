module TestSandwiching

using Test

import HiGHS
using Polyhedra
import MultiObjectiveAlgorithms as MOA
import MultiObjectiveAlgorithms: MOI

function run_tests()
    for name in names(@__MODULE__; all = true)
        if startswith("$name", "test_")
            @testset "$name" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

# From International Doctoral School Algorithmic Decision Theory: MCDA and MOO
# Lecture 2: Multiobjective Linear Programming
# Matthias Ehrgott
# Department of Engineering Science, The University of Auckland, New Zealand
# Laboratoire d’Informatique de Nantes Atlantique, CNRS, Universit´e de Nantes, France
function test_molp()
    C = Float64[
        3 1
        -1 -2
    ]
    p = size(C, 1)
    A = Float64[
        0 1
        3 -1
    ]
    m, n = size(A)
    b = Float64[3, 6]
    model = MOI.instantiate(; with_bridge_type = Float64) do
        return MOA.Optimizer(HiGHS.Optimizer)
    end
    MOI.set(model, MOA.Algorithm(), MOA.Sandwiching(0.0))
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, n)
    MOI.add_constraint.(model, x, MOI.GreaterThan(0.0))
    for i in 1:m
        MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(A[i, j], x[j]) for j in 1:n],
                0.0,
            ),
            MOI.LessThan(b[i]),
        )
    end
    f = MOI.VectorAffineFunction(
        [
            MOI.VectorAffineTerm(i, MOI.ScalarAffineTerm(C[i, j], x[j])) for
            i in 1:p for j in 1:n
        ],
        zeros(p),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    N = MOI.get(model, MOI.ResultCount())
    solutions = reverse([
        MOI.get(model, MOI.VariablePrimal(i), x) =>
            MOI.get(model, MOI.ObjectiveValue(i)) for i in 1:N
    ])
    results = reverse([
        [0.0, 0.0] => [0.0, 0.0],
        [0.0, 3.0] => [3.0, -6.0],
        [3.0, 3.0] => [12.0, -9.0],
    ])
    @test length(solutions) == length(results)
    for (sol, res) in zip(solutions, results)
        x_sol, y_sol = sol
        x_res, y_res = res
        @test ≈(x_sol, x_res; atol = 1e-6)
        @test ≈(y_sol, y_res; atol = 1e-6)
    end
end

end

TestSandwiching.run_tests()
