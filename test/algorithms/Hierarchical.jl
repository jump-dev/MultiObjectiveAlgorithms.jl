#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

module TestHierarchical

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

function test_sorted_priorities()
    @test MOA._sorted_priorities([0, 0, 0]) == [[1, 2, 3]]
    @test MOA._sorted_priorities([1, 0, 0]) == [[1], [2, 3]]
    @test MOA._sorted_priorities([0, 1, 0]) == [[2], [1, 3]]
    @test MOA._sorted_priorities([0, 0, 1]) == [[3], [1, 2]]
    @test MOA._sorted_priorities([0, 1, 1]) == [[2, 3], [1]]
    @test MOA._sorted_priorities([0, 2, 1]) == [[2], [3], [1]]
    return
end

function test_knapsack()
    P = Float64[1 0 0 0; 0 1 1 0; 0 0 1 1; 0 1 0 0]
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.Hierarchical())
    MOI.set.(model, MOA.ObjectivePriority.(1:4), [2, 1, 1, 0])
    MOI.set.(model, MOA.ObjectiveWeight.(1:4), [1, 0.5, 0.5, 1])
    MOI.set(model, MOA.ObjectiveRelativeTolerance(1), 0.1)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, 4)
    MOI.add_constraint.(model, x, MOI.GreaterThan(0.0))
    MOI.add_constraint.(model, x, MOI.LessThan(1.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    f = MOI.Utilities.operate(vcat, Float64, P * x...)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.add_constraint(model, sum(1.0 * x[i] for i in 1:4), MOI.LessThan(2.0))
    MOI.optimize!(model)
    x_sol = MOI.get(model, MOI.VariablePrimal(), x)
    @test ≈(x_sol, [0.9, 0, 0.9, 0.2]; atol = 1e-3)
    return
end

function test_knapsack_min()
    P = Float64[1 0 0 0; 0 1 1 0; 0 0 1 1; 0 1 0 0]
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.Hierarchical())
    MOI.set.(model, MOA.ObjectivePriority.(1:4), [2, 1, 1, 0])
    MOI.set.(model, MOA.ObjectiveWeight.(1:4), [1, 0.5, 0.5, 1])
    MOI.set(model, MOA.ObjectiveRelativeTolerance(1), 0.1)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, 4)
    MOI.add_constraint.(model, x, MOI.GreaterThan(0.0))
    MOI.add_constraint.(model, x, MOI.LessThan(1.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    f = MOI.Utilities.operate(vcat, Float64, -P * x...)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.add_constraint(model, sum(1.0 * x[i] for i in 1:4), MOI.LessThan(2.0))
    MOI.optimize!(model)
    x_sol = MOI.get(model, MOI.VariablePrimal(), x)
    @test ≈(x_sol, [0.9, 0, 0.9, 0.2]; atol = 1e-3)
    return
end

function test_infeasible()
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.Hierarchical())
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, 2)
    MOI.add_constraint.(model, x, MOI.GreaterThan(0.0))
    MOI.add_constraint(model, 1.0 * x[1] + 1.0 * x[2], MOI.LessThan(-1.0))
    f = MOI.Utilities.operate(vcat, Float64, 1.0 .* x...)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OTHER_ERROR
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.NO_SOLUTION
    @test MOI.get(model, MOI.DualStatus()) == MOI.NO_SOLUTION
    return
end

function test_unbounded()
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.Hierarchical())
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, 2)
    MOI.add_constraint.(model, x, MOI.GreaterThan(0.0))
    f = MOI.Utilities.operate(vcat, Float64, 1.0 .* x...)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OTHER_ERROR
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.NO_SOLUTION
    @test MOI.get(model, MOI.DualStatus()) == MOI.NO_SOLUTION
    return
end

end

TestHierarchical.run_tests()
