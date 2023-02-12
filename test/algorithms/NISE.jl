#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

module TestNISE

using Test

import HiGHS
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

function test_NISE_SolutionLimit()
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.NISE())
    @test MOI.get(model, MOA.SolutionLimit()) ==
          MOA.default(MOA.SolutionLimit())
    MOI.set(model, MOA.SolutionLimit(), 1)
    @test MOI.get(model, MOA.SolutionLimit()) == 1
    return
end

function test_moi_bolp_1()
    f = MOI.OptimizerWithAttributes(
        () -> MOA.Optimizer(HiGHS.Optimizer),
        MOA.Algorithm() => MOA.NISE(),
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
    return
end

function test_moi_bolp_1_maximize()
    f = MOI.OptimizerWithAttributes(
        () -> MOA.Optimizer(HiGHS.Optimizer),
        MOA.Algorithm() => MOA.NISE(),
    )
    model = MOI.instantiate(f)
    MOI.set(model, MOI.Silent(), true)
    MOI.Utilities.loadfromstring!(
        model,
        """
variables: x, y
maxobjective: [-2.0 * x + -1.0 * y, -1.0 * x + -3.0 * y]
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
    Y = [-[1.0, 3.0], -[1.5, 2.0], -[2.25, 1.75]]
    for i in 1:3
        @test MOI.get(model, MOI.PrimalStatus(i)) == MOI.FEASIBLE_POINT
        @test MOI.get(model, MOI.DualStatus(i)) == MOI.NO_SOLUTION
        @test MOI.get(model, MOI.ObjectiveValue(i)) == Y[i]
        @test MOI.get(model, MOI.VariablePrimal(i), x) == X[i][1]
        @test MOI.get(model, MOI.VariablePrimal(i), y) == X[i][2]
    end
    @test MOI.get(model, MOI.ObjectiveBound()) == -[1.0, 1.75]
    return
end

function test_moi_bolp_1_reversed()
    f = MOI.OptimizerWithAttributes(
        () -> MOA.Optimizer(HiGHS.Optimizer),
        MOA.Algorithm() => MOA.NISE(),
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
        MOA.Algorithm() => MOA.NISE(),
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
        MOA.Algorithm() => MOA.NISE(),
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

end

TestNISE.run_tests()
