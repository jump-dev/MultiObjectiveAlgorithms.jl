#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

module TestEpsilonConstraint

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

function test_biobjective_knapsack()
    p1 = [77, 94, 71, 63, 96, 82, 85, 75, 72, 91, 99, 63, 84, 87, 79, 94, 90]
    p2 = [65, 90, 90, 77, 95, 84, 70, 94, 66, 92, 74, 97, 60, 60, 65, 97, 93]
    w = [80, 87, 68, 72, 66, 77, 99, 85, 70, 93, 98, 72, 100, 89, 67, 86, 91]
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.EpsilonConstraint())
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
        [955, 906] => [2, 3, 5, 6, 9, 10, 11, 14, 15, 16, 17],
        [949, 915] => [1, 2, 5, 6, 8, 9, 10, 11, 15, 16, 17],
        [948, 939] => [1, 2, 3, 5, 6, 8, 10, 11, 15, 16, 17],
        [943, 940] => [2, 3, 5, 6, 8, 9, 10, 11, 15, 16, 17],
        [936, 942] => [1, 2, 3, 5, 6, 10, 11, 12, 15, 16, 17],
        [935, 947] => [2, 5, 6, 8, 9, 10, 11, 12, 15, 16, 17],
        [934, 971] => [2, 3, 5, 6, 8, 10, 11, 12, 15, 16, 17],
        [927, 972] => [2, 3, 5, 6, 8, 9, 10, 11, 12, 16, 17],
        [918, 983] => [2, 3, 4, 5, 6, 8, 10, 11, 12, 16, 17],
    )
    @test MOI.get(model, MOI.ResultCount()) == 9
    for i in 1:MOI.get(model, MOI.ResultCount())
        x_sol = MOI.get(model, MOI.VariablePrimal(i), x)
        X = findall(elt -> elt > 0.9, x_sol)
        Y = MOI.get(model, MOI.ObjectiveValue(i))
        @test results[round.(Int, Y)] == X
    end
    return
end

function test_biobjective_knapsack_atol()
    p1 = [77, 94, 71, 63, 96, 82, 85, 75, 72, 91, 99, 63, 84, 87, 79, 94, 90]
    p2 = [65, 90, 90, 77, 95, 84, 70, 94, 66, 92, 74, 97, 60, 60, 65, 97, 93]
    w = [80, 87, 68, 72, 66, 77, 99, 85, 70, 93, 98, 72, 100, 89, 67, 86, 91]
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.EpsilonConstraint())
    MOI.set(model, MOA.ObjectiveAbsoluteTolerance(1), 1.0)
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
        [955, 906] => [2, 3, 5, 6, 9, 10, 11, 14, 15, 16, 17],
        [949, 915] => [1, 2, 5, 6, 8, 9, 10, 11, 15, 16, 17],
        [948, 939] => [1, 2, 3, 5, 6, 8, 10, 11, 15, 16, 17],
        [943, 940] => [2, 3, 5, 6, 8, 9, 10, 11, 15, 16, 17],
        [936, 942] => [1, 2, 3, 5, 6, 10, 11, 12, 15, 16, 17],
        [935, 947] => [2, 5, 6, 8, 9, 10, 11, 12, 15, 16, 17],
        [934, 971] => [2, 3, 5, 6, 8, 10, 11, 12, 15, 16, 17],
        [927, 972] => [2, 3, 5, 6, 8, 9, 10, 11, 12, 16, 17],
        [918, 983] => [2, 3, 4, 5, 6, 8, 10, 11, 12, 16, 17],
    )
    @test MOI.get(model, MOI.ResultCount()) == 9
    for i in 1:MOI.get(model, MOI.ResultCount())
        x_sol = MOI.get(model, MOI.VariablePrimal(i), x)
        X = findall(elt -> elt > 0.9, x_sol)
        Y = MOI.get(model, MOI.ObjectiveValue(i))
        @test results[round.(Int, Y)] == X
    end
    return
end

function test_biobjective_knapsack_min()
    p1 = [77, 94, 71, 63, 96, 82, 85, 75, 72, 91, 99, 63, 84, 87, 79, 94, 90]
    p2 = [65, 90, 90, 77, 95, 84, 70, 94, 66, 92, 74, 97, 60, 60, 65, 97, 93]
    w = [80, 87, 68, 72, 66, 77, 99, 85, 70, 93, 98, 72, 100, 89, 67, 86, 91]
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.EpsilonConstraint())
    MOI.set(model, MOA.SolutionLimit(), 100)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, length(w))
    MOI.add_constraint.(model, x, MOI.ZeroOne())
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    f = MOI.Utilities.operate(
        vcat,
        Float64,
        [sum(-1.0 * p[i] * x[i] for i in 1:length(w)) for p in [p1, p2]]...,
    )
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.add_constraint(
        model,
        sum(1.0 * w[i] * x[i] for i in 1:length(w)),
        MOI.LessThan(900.0),
    )
    MOI.optimize!(model)
    results = Dict(
        [955, 906] => [2, 3, 5, 6, 9, 10, 11, 14, 15, 16, 17],
        [949, 915] => [1, 2, 5, 6, 8, 9, 10, 11, 15, 16, 17],
        [948, 939] => [1, 2, 3, 5, 6, 8, 10, 11, 15, 16, 17],
        [943, 940] => [2, 3, 5, 6, 8, 9, 10, 11, 15, 16, 17],
        [936, 942] => [1, 2, 3, 5, 6, 10, 11, 12, 15, 16, 17],
        [935, 947] => [2, 5, 6, 8, 9, 10, 11, 12, 15, 16, 17],
        [934, 971] => [2, 3, 5, 6, 8, 10, 11, 12, 15, 16, 17],
        [927, 972] => [2, 3, 5, 6, 8, 9, 10, 11, 12, 16, 17],
        [918, 983] => [2, 3, 4, 5, 6, 8, 10, 11, 12, 16, 17],
    )
    @test MOI.get(model, MOI.ResultCount()) == 9
    for i in 1:MOI.get(model, MOI.ResultCount())
        x_sol = MOI.get(model, MOI.VariablePrimal(i), x)
        X = findall(elt -> elt > 0.9, x_sol)
        Y = MOI.get(model, MOI.ObjectiveValue(i))
        @test results[-round.(Int, Y)] == X
    end
    return
end

function test_biobjective_knapsack_min_solution_limit()
    p1 = [77, 94, 71, 63, 96, 82, 85, 75, 72, 91, 99, 63, 84, 87, 79, 94, 90]
    p2 = [65, 90, 90, 77, 95, 84, 70, 94, 66, 92, 74, 97, 60, 60, 65, 97, 93]
    w = [80, 87, 68, 72, 66, 77, 99, 85, 70, 93, 98, 72, 100, 89, 67, 86, 91]
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.EpsilonConstraint())
    MOI.set(model, MOA.SolutionLimit(), 3)
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
        [955, 906] => [2, 3, 5, 6, 9, 10, 11, 14, 15, 16, 17],
        [949, 915] => [1, 2, 5, 6, 8, 9, 10, 11, 15, 16, 17],
        [948, 939] => [1, 2, 3, 5, 6, 8, 10, 11, 15, 16, 17],
        [943, 940] => [2, 3, 5, 6, 8, 9, 10, 11, 15, 16, 17],
        [936, 942] => [1, 2, 3, 5, 6, 10, 11, 12, 15, 16, 17],
        [935, 947] => [2, 5, 6, 8, 9, 10, 11, 12, 15, 16, 17],
        [934, 971] => [2, 3, 5, 6, 8, 10, 11, 12, 15, 16, 17],
        [927, 972] => [2, 3, 5, 6, 8, 9, 10, 11, 12, 16, 17],
        [918, 983] => [2, 3, 4, 5, 6, 8, 10, 11, 12, 16, 17],
    )
    @test MOI.get(model, MOI.ResultCount()) == 3
    for i in 1:MOI.get(model, MOI.ResultCount())
        x_sol = MOI.get(model, MOI.VariablePrimal(i), x)
        X = findall(elt -> elt > 0.9, x_sol)
        Y = MOI.get(model, MOI.ObjectiveValue(i))
        @test results[round.(Int, Y)] == X
    end
    return
end

end

TestEpsilonConstraint.run_tests()
