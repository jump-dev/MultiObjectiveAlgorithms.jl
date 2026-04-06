#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

module TestDichotomy

using Test

import HiGHS
import Ipopt
import MultiObjectiveAlgorithms as MOA
import MultiObjectiveAlgorithms: MOI

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

function test_Dichotomy_SolutionLimit()
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.Dichotomy())
    @test MOI.supports(MOA.Dichotomy(), MOA.SolutionLimit())
    @test MOI.supports(model, MOA.SolutionLimit())
    @test MOI.get(model, MOA.SolutionLimit()) ==
          MOA._default(MOA.SolutionLimit())
    MOI.set(model, MOA.SolutionLimit(), 1)
    @test MOI.get(model, MOA.SolutionLimit()) == 1
    return
end

function test_moi_bolp_1()
    f = MOI.OptimizerWithAttributes(
        () -> MOA.Optimizer(HiGHS.Optimizer),
        MOA.Algorithm() => MOA.Dichotomy(),
    )
    model = MOI.instantiate(f)
    MOI.set(model, MOI.Silent(), true)
    MOI.Utilities.loadfromstring!(
        model,
        """
        variables: x, y
        minobjective: [2 * x + y + 1, x + 3 * y]
        c1: x + y >= 1.0
        c2: 0.5 * x + y >= 0.75
        c3: x >= 0.0
        c4: y >= 0.25
        """,
    )
    x = MOI.get(model, MOI.VariableIndex, "x")
    y = MOI.get(model, MOI.VariableIndex, "y")
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(model, MOI.ResultCount()) == 3
    X = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.25]]
    Y = [[2.0, 3.0], [2.5, 2.0], [3.25, 1.75]]
    for i in 1:3
        @test MOI.get(model, MOI.PrimalStatus(i)) == MOI.FEASIBLE_POINT
        @test MOI.get(model, MOI.DualStatus(i)) == MOI.NO_SOLUTION
        @test MOI.get(model, MOI.ObjectiveValue(i)) == Y[i]
        @test MOI.get(model, MOI.VariablePrimal(i), x) == X[i][1]
        @test MOI.get(model, MOI.VariablePrimal(i), y) == X[i][2]
    end
    @test MOI.get(model, MOI.ObjectiveBound()) == [2.0, 1.75]
    return
end

function test_moi_bolp_1_maximize()
    f = MOI.OptimizerWithAttributes(
        () -> MOA.Optimizer(HiGHS.Optimizer),
        MOA.Algorithm() => MOA.Dichotomy(),
    )
    model = MOI.instantiate(f)
    MOI.set(model, MOI.Silent(), true)
    MOI.Utilities.loadfromstring!(
        model,
        """
        variables: x, y
        maxobjective: [-2.0 * x + -1.0 * y, -1.0 * x + -3.0 * y + 0.5]
        c1: x + y >= 1.0
        c2: 0.5 * x + y >= 0.75
        c3: x >= 0.0
        c4: y >= 0.25
        """,
    )
    x = MOI.get(model, MOI.VariableIndex, "x")
    y = MOI.get(model, MOI.VariableIndex, "y")
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(model, MOI.ResultCount()) == 3
    X = [[1.0, 0.25], [0.5, 0.5], [0.0, 1.0]]
    Y = [[-2.25, -1.25], [-1.5, -1.5], [-1.0, -2.5]]
    reverse!(X)
    reverse!(Y)
    for i in 1:3
        @test MOI.get(model, MOI.PrimalStatus(i)) == MOI.FEASIBLE_POINT
        @test MOI.get(model, MOI.DualStatus(i)) == MOI.NO_SOLUTION
        @test MOI.get(model, MOI.ObjectiveValue(i)) == Y[i]
        @test MOI.get(model, MOI.VariablePrimal(i), x) == X[i][1]
        @test MOI.get(model, MOI.VariablePrimal(i), y) == X[i][2]
    end
    @test MOI.get(model, MOI.ObjectiveBound()) == -[1.0, 1.25]
    return
end

function test_moi_bolp_1_reversed()
    f = MOI.OptimizerWithAttributes(
        () -> MOA.Optimizer(HiGHS.Optimizer),
        MOA.Algorithm() => MOA.Dichotomy(),
    )
    model = MOI.instantiate(f)
    MOI.set(model, MOI.Silent(), true)
    MOI.Utilities.loadfromstring!(
        model,
        """
        variables: x, y
        minobjective: [x + 3 * y, 2 * x + y]
        c1: x + y >= 1.0
        c2: 0.5 * x + y >= 0.75
        c3: x >= 0.0
        c4: y >= 0.25
        """,
    )
    x = MOI.get(model, MOI.VariableIndex, "x")
    y = MOI.get(model, MOI.VariableIndex, "y")
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(model, MOI.ResultCount()) == 3
    X = reverse([[0.0, 1.0], [0.5, 0.5], [1.0, 0.25]])
    Y = reverse([[1.0, 3.0], [1.5, 2.0], [2.25, 1.75]])
    for i in 1:3
        @test MOI.get(model, MOI.PrimalStatus(i)) == MOI.FEASIBLE_POINT
        @test MOI.get(model, MOI.DualStatus(i)) == MOI.NO_SOLUTION
        @test MOI.get(model, MOI.ObjectiveValue(i)) == reverse(Y[i])
        @test MOI.get(model, MOI.VariablePrimal(i), x) == X[i][1]
        @test MOI.get(model, MOI.VariablePrimal(i), y) == X[i][2]
    end
    @test MOI.get(model, MOI.ObjectiveBound()) == reverse([1.0, 1.75])
    return
end

function test_moi_bolp_1_scalar()
    f = MOI.OptimizerWithAttributes(
        () -> MOA.Optimizer(HiGHS.Optimizer),
        MOA.Algorithm() => MOA.Dichotomy(),
    )
    model = MOI.instantiate(f)
    MOI.set(model, MOI.Silent(), true)
    MOI.Utilities.loadfromstring!(
        model,
        """
        variables: x, y
        minobjective: [2 * x + y, x + 3 * y]
        c1: x + y >= 1.0
        c2: 0.5 * x + y >= 0.75
        c3: x >= 0.0
        c4: y >= 0.25
        """,
    )
    x = MOI.get(model, MOI.VariableIndex, "x")
    y = MOI.get(model, MOI.VariableIndex, "y")
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(model, MOI.ResultCount()) == 3
    X = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.25]]
    Y = [[1.0, 3.0], [1.5, 2.0], [2.25, 1.75]]
    for i in 1:3
        @test MOI.get(model, MOI.PrimalStatus(i)) == MOI.FEASIBLE_POINT
        @test MOI.get(model, MOI.DualStatus(i)) == MOI.NO_SOLUTION
        @test MOI.get(model, MOI.ObjectiveValue(i)) == Y[i]
        @test MOI.get(model, MOI.VariablePrimal(i), x) == X[i][1]
        @test MOI.get(model, MOI.VariablePrimal(i), y) == X[i][2]
    end
    @test MOI.get(model, MOI.ObjectiveBound()) == [1.0, 1.75]
    f = MOI.Utilities.operate(vcat, Float64, 2.0 * x + 1.0 * y)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(model, MOI.ResultCount()) == 1
    X = [[0.0, 1.0]]
    Y = [[1.0]]
    for i in 1:1
        @test MOI.get(model, MOI.PrimalStatus(i)) == MOI.FEASIBLE_POINT
        @test MOI.get(model, MOI.DualStatus(i)) == MOI.NO_SOLUTION
        @test MOI.get(model, MOI.ObjectiveValue(i)) == Y[i]
        @test MOI.get(model, MOI.VariablePrimal(i), x) == X[i][1]
        @test MOI.get(model, MOI.VariablePrimal(i), y) == X[i][2]
    end
    @test MOI.get(model, MOI.ObjectiveBound()) == [1.0]
    return
end

function test_biobjective_knapsack()
    p1 = [77, 94, 71, 63, 96, 82, 85, 75, 72, 91, 99, 63, 84, 87, 79, 94, 90]
    p2 = [65, 90, 90, 77, 95, 84, 70, 94, 66, 92, 74, 97, 60, 60, 65, 97, 93]
    w = [80, 87, 68, 72, 66, 77, 99, 85, 70, 93, 98, 72, 100, 89, 67, 86, 91]
    f = MOI.OptimizerWithAttributes(
        () -> MOA.Optimizer(HiGHS.Optimizer),
        MOA.Algorithm() => MOA.Dichotomy(),
    )
    model = MOI.instantiate(f)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, length(w))
    MOI.add_constraint.(model, x, MOI.ZeroOne())
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    f = MOI.Utilities.operate(
        vcat,
        Float64,
        [sum(1.0 * p[i] * x[i] for i in 1:length(w)) for p in [p1, p2]]...,
    )
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.add_constraint(
        model,
        sum(1.0 * w[i] * x[i] for i in 1:length(w)),
        MOI.LessThan(900.0),
    )
    MOI.optimize!(model)
    results = [
        [918.0, 983.0] => [2, 3, 4, 5, 6, 8, 10, 11, 12, 16, 17],
        [934.0, 971.0] => [2, 3, 5, 6, 8, 10, 11, 12, 15, 16, 17],
        [948.0, 939.0] => [1, 2, 3, 5, 6, 8, 10, 11, 15, 16, 17],
        [955.0, 906.0] => [2, 3, 5, 6, 9, 10, 11, 14, 15, 16, 17],
    ]
    reverse!(results)
    for i in 1:MOI.get(model, MOI.ResultCount())
        x_sol = MOI.get(model, MOI.VariablePrimal(i), x)
        @test results[i][2] == findall(elt -> elt > 0.9, x_sol)
        @test results[i][1] ≈ MOI.get(model, MOI.ObjectiveValue(i))
    end
    return
end

function test_time_limit()
    p1 = [77, 94, 71, 63, 96, 82, 85, 75, 72, 91, 99, 63, 84, 87, 79, 94, 90]
    p2 = [65, 90, 90, 77, 95, 84, 70, 94, 66, 92, 74, 97, 60, 60, 65, 97, 93]
    w = [80, 87, 68, 72, 66, 77, 99, 85, 70, 93, 98, 72, 100, 89, 67, 86, 91]
    f = MOI.OptimizerWithAttributes(
        () -> MOA.Optimizer(HiGHS.Optimizer),
        MOA.Algorithm() => MOA.Dichotomy(),
    )
    model = MOI.instantiate(f)
    MOI.set(model, MOI.Silent(), true)
    MOI.set(model, MOI.TimeLimitSec(), 0.0)
    x = MOI.add_variables(model, length(w))
    MOI.add_constraint.(model, x, MOI.ZeroOne())
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    f = MOI.Utilities.operate(
        vcat,
        Float64,
        [sum(1.0 * p[i] * x[i] for i in 1:length(w)) for p in [p1, p2]]...,
    )
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.add_constraint(
        model,
        sum(1.0 * w[i] * x[i] for i in 1:length(w)),
        MOI.LessThan(900.0),
    )
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.TIME_LIMIT
    @test MOI.get(model, MOI.ResultCount()) == 2
    return
end

function test_infeasible()
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.Dichotomy())
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, 2)
    MOI.add_constraint.(model, x, MOI.GreaterThan(0.0))
    MOI.add_constraint(model, 1.0 * x[1] + 1.0 * x[2], MOI.LessThan(-1.0))
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
    MOI.set(model, MOA.Algorithm(), MOA.Dichotomy())
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

function test_bicriteria_transportation_nise()
    m, n = 3, 4
    c = Float64[1 2 7 7; 1 9 3 4; 8 9 4 6]
    d = Float64[4 4 3 4; 5 8 9 10; 6 2 5 1]
    a = Float64[11, 3, 14, 16]
    b = Float64[8, 19, 17]
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.Dichotomy())
    MOI.set(model, MOI.Silent(), true)
    x = [MOI.add_variable(model) for i in 1:m, j in 1:n]
    MOI.add_constraint.(model, x, MOI.GreaterThan(0.0))
    for j in 1:n
        terms = [MOI.ScalarAffineTerm(1.0, x[i, j]) for i in 1:m]
        MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(terms, 0.0),
            MOI.EqualTo(a[j]),
        )
    end
    for i in 1:m
        terms = [MOI.ScalarAffineTerm(1.0, x[i, j]) for j in 1:n]
        MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(terms, 0.0),
            MOI.EqualTo(b[i]),
        )
    end
    f = MOI.Utilities.vectorize([
        sum(c[i, j] * x[i, j] for i in 1:m, j in 1:n),
        sum(d[i, j] * x[i, j] for i in 1:m, j in 1:n),
    ])
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(model)
    N = MOI.get(model, MOI.ResultCount())
    y_sol = hcat(MOI.get.(model, MOI.ObjectiveValue.(1:N))...)
    Y_N = Float64[143 156 176 186 208; 265 200 175 171 167]
    @test isapprox(y_sol, Y_N; atol = 1e-6)
    return
end

function test_deprecated()
    nise = MOA.NISE()
    dichotomy = MOA.Dichotomy()
    @test nise isa typeof(dichotomy)
    @test nise.solution_limit === dichotomy.solution_limit
    return
end

function test_three_objective()
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.Dichotomy())
    MOI.set(model, MOI.Silent(), true)
    MOI.Utilities.loadfromstring!(
        model,
        """
        variables: x
        maxobjective: [1.0 * x, -1.0 * x, 2.0 * x + 2.0]
        """,
    )
    @test_throws(
        ErrorException("Only scalar or bi-objective problems supported."),
        MOI.optimize!(model),
    )
    return
end

function test_quadratic()
    μ = [0.05470748600000001, 0.18257110599999998]
    Q = [0.00076204 0.00051972; 0.00051972 0.00546173]
    N = 2
    model = MOA.Optimizer(Ipopt.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.Dichotomy())
    MOI.set(model, MOA.SolutionLimit(), 10)
    MOI.set(model, MOI.Silent(), true)
    w = MOI.add_variables(model, N)
    MOI.add_constraint.(model, w, MOI.GreaterThan(0.0))
    MOI.add_constraint.(model, w, MOI.LessThan(1.0))
    MOI.add_constraint(model, sum(1.0 * w[i] for i in 1:N), MOI.EqualTo(1.0))
    var = sum(Q[i, j] * w[i] * w[j] for i in 1:N, j in 1:N)
    mean = sum(-μ[i] * w[i] for i in 1:N)
    f = MOI.Utilities.operate(vcat, Float64, var, mean)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.ResultCount()) == 10
    for i in 1:MOI.get(model, MOI.ResultCount())
        w_sol = MOI.get(model, MOI.VariablePrimal(i), w)
        y = MOI.get(model, MOI.ObjectiveValue(i))
        @test y ≈ [w_sol' * Q * w_sol, -μ' * w_sol]
    end
    return
end

function test_vector_of_variables_objective()
    model = MOI.instantiate(; with_bridge_type = Float64) do
        return MOA.Optimizer(HiGHS.Optimizer)
    end
    MOI.set(model, MOA.Algorithm(), MOA.Dichotomy())
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, 2)
    MOI.add_constraint.(model, x, MOI.ZeroOne())
    f = MOI.VectorOfVariables(x)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.add_constraint(model, sum(1.0 * xi for xi in x), MOI.GreaterThan(1.0))
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(model, MOA.SubproblemCount()) >= 1
    return
end

function test_solve_failures()
    m, n = 2, 10
    p1 = [5.0 1 10 8 3 5 3 3 7 2; 10 6 1 6 8 3 2 10 6 1]
    p2 = [4.0 6 4 3 1 6 8 2 9 7; 8 8 8 2 4 8 8 1 10 1]
    w = [5.0 9 3 5 10 5 7 10 7 8; 4 8 8 6 10 8 10 7 5 1]
    b = [34.0, 33.0]
    for fail_after in 0:3
        model = MOA.Optimizer(mock_optimizer(fail_after))
        MOI.set(model, MOA.Algorithm(), MOA.Dichotomy())
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
        @test MOI.get(model, MOI.ResultCount()) ==
              (fail_after < 2 ? 0 : fail_after)
    end
    return
end

function test_scalar_time_limit()
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.Dichotomy())
    MOI.set(model, MOI.Silent(), true)
    MOI.set(model, MOI.TimeLimitSec(), 0.0)
    MOI.Utilities.loadfromstring!(
        model,
        """
        variables: x
        minobjective: [2 * x]
        x >= 0.0
        """,
    )
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.TIME_LIMIT
    return
end

function test_dichotomy_issue_191()
    #!format:off
    C = Float64[
        31 42 34 29 28 24 93 81 60 65 70 26 73 78 70 6 53 17 75 53 37 5 68 84 18 78 14 14 33 13 77 60 34 12 87 47 48 47 22 38 22 46 16 12 40 82 42 51 51 10 100 40 81 86 29 95 97 72 89 87 77 86 40 80 2 64 66 76 24 36 82 68 9 75 94 87 35 36 57 34 10 12 57 46 42 53 22 55 8 38 53 73 77 91 61 98 52 75 47 39 10 30 65 68 65 24 57 70 57 11 77 49 76 14 56 81 77 90 97 70 84 22 41 5 3 50 98 20 23 94 46 39 25 44 63 91 64 8 75 15 11 14 64 72 56 80 79 61 75 83 14 21 92 58 77 92 74 82 96 54 19 15 23 12 89 44 37 29 15 41 61 73 16 86 81 47 74 83 4 20 67 87 9 52 87 21 4 52 16 87 86 35 21 3 97 65 73 8 65 75
        38 45 53 27 44 71 84 35 66 100 69 95 13 24 90 5 95 28 13 76 65 68 16 42 41 58 31 53 74 91 24 83 43 46 44 26 22 4 47 41 49 56 20 44 23 96 35 94 84 1 20 84 18 99 52 7 64 27 81 70 87 1 72 45 96 35 31 25 42 80 90 70 3 6 96 76 80 67 78 99 66 68 10 91 57 81 65 67 18 49 13 17 7 64 33 16 94 52 55 28 58 74 85 43 8 77 33 57 64 67 66 15 65 8 14 38 12 87 68 8 67 12 2 37 94 32 53 99 63 8 90 25 38 27 68 47 79 43 40 75 4 25 10 58 6 79 18 78 71 11 45 96 77 69 20 57 58 81 6 63 17 82 97 63 21 10 39 71 83 57 50 59 83 47 31 81 33 42 12 46 30 60 41 54 78 92 24 96 74 46 19 92 68 91 39 100 38 2 41 59
    ]
    w = Float64[2, 9, 32, 24, 39, 8, 57, 44, 98, 35, 54, 9, 37, 60, 52, 40, 94, 92, 8, 52, 4, 91, 10, 57, 75, 20, 32, 19, 53, 59, 96, 84, 32, 86, 13, 84, 47, 70, 37, 33, 40, 40, 63, 59, 27, 51, 51, 67, 42, 50, 40, 73, 7, 74, 94, 30, 40, 29, 85, 37, 62, 51, 76, 32, 72, 53, 11, 61, 17, 84, 44, 76, 41, 70, 71, 53, 7, 65, 12, 84, 93, 47, 57, 83, 97, 23, 79, 94, 55, 84, 94, 28, 54, 29, 26, 82, 70, 12, 34, 87, 96, 77, 55, 57, 70, 65, 54, 8, 47, 77, 94, 59, 82, 31, 80, 69, 13, 56, 14, 75, 52, 89, 11, 28, 10, 82, 17, 41, 67, 51, 97, 75, 29, 54, 98, 34, 96, 79, 78, 42, 98, 26, 89, 22, 40, 63, 73, 23, 72, 23, 35, 26, 24, 68, 93, 89, 84, 39, 3, 84, 33, 39, 12, 85, 11, 5, 91, 26, 10, 71, 25, 30, 66, 36, 67, 94, 74, 9, 81, 34, 41, 86, 7, 3, 75, 94, 84, 95, 34, 47, 26, 90, 17, 43, 88, 94, 35, 10, 34, 13.0]
    #!format:on
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    MOI.set(model, MOA.Algorithm(), MOA.Dichotomy())
    x = MOI.add_variables(model, 200)
    MOI.add_constraint.(model, x, MOI.ZeroOne())
    MOI.add_constraint(model, w' * x, MOI.LessThan(5207.5))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    f = MOI.Utilities.vectorize(C * x)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test 38 <= MOI.get(model, MOI.ResultCount()) <= 39
    @test 70 <= MOI.get(model, MOA.SubproblemCount()) <= 80
    return
end

end  # module TestDichotomy

TestDichotomy.run_tests()
