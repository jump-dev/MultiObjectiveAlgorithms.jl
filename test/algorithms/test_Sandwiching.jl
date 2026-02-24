#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

module TestSandwiching

using Test

import HiGHS
import MultiObjectiveAlgorithms as MOA
import MultiObjectiveAlgorithms: MOI
import Polyhedra

include(joinpath(dirname(@__DIR__), "mock_optimizer.jl"))

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

function _test_molp(C, A, b, results, sense)
    p = size(C, 1)
    m, n = size(A)
    model = MOA.Optimizer(HiGHS.Optimizer)
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
    MOI.set(model, MOI.ObjectiveSense(), sense)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    N = MOI.get(model, MOI.ResultCount())
    solutions = sort([
        MOI.get(model, MOI.VariablePrimal(i), x) =>
            MOI.get(model, MOI.ObjectiveValue(i)) for i in 1:N
    ])
    @test N == length(results)
    @test MOI.get(model, MOA.SubproblemCount()) >= length(results)
    for (sol, res) in zip(solutions, results)
        x_sol, y_sol = sol
        x_res, y_res = res
        @test ≈(x_sol, x_res; atol = 1e-6)
        @test ≈(y_sol, y_res; atol = 1e-6)
    end
    return
end

# From International Doctoral School Algorithmic Decision Theory: MCDA and MOO
# Lecture 2: Multiobjective Linear Programming
# Matthias Ehrgott
# Department of Engineering Science, The University of Auckland, New Zealand
# Laboratoire d’Informatique de Nantes Atlantique, CNRS, Universit´e de Nantes, France
function test_molp_1()
    C = Float64[3 1; -1 -2]
    A = Float64[0 1; 3 -1]
    b = Float64[3, 6]
    results = sort([
        [0.0, 0.0] => [0.0, 0.0],
        [0.0, 3.0] => [3.0, -6.0],
        [3.0, 3.0] => [12.0, -9.0],
    ])
    sense = MOI.MIN_SENSE
    return _test_molp(C, A, b, results, sense)
end

# From Civil and Environmental Systems Engineering
# Chapter 5 Exercise 5.A.3 A graphical Interpretation of Noninferiority
function test_molp_2()
    C = Float64[3 -2; -1 2]
    A = Float64[-4 -8; 3 -6; 4 -2; 1 0; -1 3; -2 4; -6 3]
    b = Float64[-8, 6, 14, 6, 15, 18, 9]
    results = [
        [3.0, 6.0] => [-3.0, 9.0],
        [4.0, 1.0] => [10.0, -2.0],
        [6.0, 5.0] => [8.0, 4.0],
        [6.0, 7.0] => [4.0, 8.0],
    ]
    sense = MOI.MAX_SENSE
    return _test_molp(C, A, b, results, sense)
end

function test_infeasible()
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.Sandwiching(0.0))
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, 2)
    MOI.add_constraint.(model, x, MOI.GreaterThan(0.0))
    MOI.add_constraint(model, 1.0 * x[1] + 1.0 * x[2], MOI.LessThan(-1.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    f = MOI.Utilities.operate(vcat, Float64, 1.0 .* x...)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.INFEASIBLE
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.NO_SOLUTION
    @test MOI.get(model, MOI.DualStatus()) == MOI.NO_SOLUTION
    return
end

function test_unbounded()
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.Sandwiching(0.0))
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, 2)
    MOI.add_constraint.(model, x, MOI.GreaterThan(0.0))
    f = MOI.Utilities.operate(vcat, Float64, 1.0 .* x...)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.DUAL_INFEASIBLE
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.NO_SOLUTION
    @test MOI.get(model, MOI.DualStatus()) == MOI.NO_SOLUTION
    return
end

function test_no_bounding_box()
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.Sandwiching(0.0))
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, 2)
    MOI.add_constraint.(model, x, MOI.GreaterThan(0.0))
    f = MOI.Utilities.operate(vcat, Float64, 1.0 .* x...)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    @test_logs (:warn,) MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.DUAL_INFEASIBLE
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.NO_SOLUTION
    @test MOI.get(model, MOI.DualStatus()) == MOI.NO_SOLUTION
    return
end

function test_time_limit()
    p = 3
    n = 10
    W = 2137.0
    C = Float64[
        566 611 506 180 817 184 585 423 26 317
        62 84 977 979 874 54 269 93 881 563
        664 982 962 140 224 215 12 869 332 537
    ]
    w = Float64[557, 898, 148, 63, 78, 964, 246, 662, 386, 272]
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.Sandwiching(0.0))
    MOI.set(model, MOI.TimeLimitSec(), 0.0)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, n)
    MOI.add_constraint.(model, x, MOI.ZeroOne())
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(w[j], x[j]) for j in 1:n],
            0.0,
        ),
        MOI.LessThan(W),
    )
    f = MOI.VectorAffineFunction(
        [
            MOI.VectorAffineTerm(i, MOI.ScalarAffineTerm(-C[i, j], x[j]))
            for i in 1:p for j in 1:n
        ],
        fill(0.0, p),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.TIME_LIMIT
    @test MOI.get(model, MOI.ResultCount()) == size(C, 1) # anchor points are already computed when the time limit is checked
    return
end

function test_solve_failures()
    m, n = 2, 10
    p1 = [5.0 1 10 8 3 5 3 3 7 2; 10 6 1 6 8 3 2 10 6 1]
    p2 = [4.0 6 4 3 1 6 8 2 9 7; 8 8 8 2 4 8 8 1 10 1]
    w = [5.0 9 3 5 10 5 7 10 7 8; 4 8 8 6 10 8 10 7 5 1]
    b = [34.0, 33.0]
    for fail_after in 0:4
        model = MOA.Optimizer(mock_optimizer(fail_after))
        MOI.set(model, MOA.Algorithm(), MOA.Sandwiching(0.0))
        x_ = MOI.add_variables(model, m * n)
        x = reshape(x_, m, n)
        MOI.add_constraint.(model, x, MOI.Interval(0.0, 1.0))
        f = MOI.Utilities.operate(vcat, Float64, sum(p1 .* x), sum(p2 .* x))
        MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
        MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
        for i in 1:m
            f_i = sum(w[i, j] * x[i, j] for j in 1:n)
            MOI.add_constraint(model, f_i, MOI.LessThan(b[i]))
        end
        for j in 1:n
            MOI.add_constraint(model, sum(1.0 .* x[:, j]), MOI.EqualTo(1.0))
        end
        MOI.optimize!(model)
        @test MOI.get(model, MOI.TerminationStatus()) == MOI.NUMERICAL_ERROR
        @test MOI.get(model, MOI.ResultCount()) == 0
    end
    return
end

end  # TestSandwiching

TestSandwiching.run_tests()
