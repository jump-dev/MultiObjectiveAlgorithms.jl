#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

module TestDichotomy

using Test

import HiGHS
import Ipopt
import MultiObjectiveAlgorithms as MOA

const MOI = MOA.MOI

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
          MOA.default(MOA.SolutionLimit())
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
    X = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.25]]
    Y = [-[1.0, 2.5], -[1.5, 1.5], -[2.25, 1.25]]
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
    results = Dict(
        [955.0, 906.0] => [2, 3, 5, 6, 9, 10, 11, 14, 15, 16, 17],
        [948.0, 939.0] => [1, 2, 3, 5, 6, 8, 10, 11, 15, 16, 17],
        [934.0, 971.0] => [2, 3, 5, 6, 8, 10, 11, 12, 15, 16, 17],
        [918.0, 983.0] => [2, 3, 4, 5, 6, 8, 10, 11, 12, 16, 17],
    )
    for i in 1:MOI.get(model, MOI.ResultCount())
        x_sol = MOI.get(model, MOI.VariablePrimal(i), x)
        X = findall(elt -> elt > 0.9, x_sol)
        Y = MOI.get(model, MOI.ObjectiveValue(i))
        @test results[Y] == X
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
    @test MOI.get(model, MOI.ResultCount()) > 0
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
    MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    return
end

end

TestDichotomy.run_tests()
