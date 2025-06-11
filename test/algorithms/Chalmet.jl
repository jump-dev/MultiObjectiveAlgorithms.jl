#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

module TestChalmet

using Test

import HiGHS
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

function test_knapsack_min()
    n = 10
    W = 2137.0
    C = Float64[
        566 611 506 180 817 184 585 423 26 317
        62 84 977 979 874 54 269 93 881 563
    ]
    w = Float64[557, 898, 148, 63, 78, 964, 246, 662, 386, 272]
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.Chalmet())
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
            for i in 1:2 for j in 1:n
        ],
        [0.0, 0.0],
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    results = [
        [1, 0, 1, 1, 1, 0, 1, 1, 0, 1] => [-3394, -3817],
        [0, 1, 1, 1, 1, 0, 1, 0, 1, 1] => [-3042, -4627],
        [0, 0, 1, 1, 1, 0, 1, 1, 1, 1] => [-2854, -4636],
    ]
    @test MOI.get(model, MOI.ResultCount()) == length(results)
    for (i, (x_sol, y_sol)) in enumerate(results)
        @test ≈(x_sol, MOI.get(model, MOI.VariablePrimal(i), x); atol = 1e-6)
        @test ≈(y_sol, MOI.get(model, MOI.ObjectiveValue(i)); atol = 1e-6)
    end
    @test MOI.get(model, MOI.ObjectiveBound()) ≈ [-3394, -4636]
    return
end

function test_knapsack_max()
    n = 10
    W = 2137.0
    C = Float64[
        566 611 506 180 817 184 585 423 26 317
        62 84 977 979 874 54 269 93 881 563
    ]
    w = Float64[557, 898, 148, 63, 78, 964, 246, 662, 386, 272]
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.Chalmet())
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
            MOI.VectorAffineTerm(i, MOI.ScalarAffineTerm(C[i, j], x[j])) for
            i in 1:2 for j in 1:n
        ],
        [1.0, 0.0],
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    results = [
        [0, 0, 1, 1, 1, 0, 1, 1, 1, 1] => [2855, 4636],
        [0, 1, 1, 1, 1, 0, 1, 0, 1, 1] => [3043, 4627],
        [1, 0, 1, 1, 1, 0, 1, 1, 0, 1] => [3395, 3817],
    ]
    @test MOI.get(model, MOI.ResultCount()) == length(results)
    for (i, (x_sol, y_sol)) in enumerate(results)
        @test ≈(x_sol, MOI.get(model, MOI.VariablePrimal(i), x); atol = 1e-6)
        @test ≈(y_sol, MOI.get(model, MOI.ObjectiveValue(i)); atol = 1e-6)
    end
    @test MOI.get(model, MOI.ObjectiveBound()) ≈ [3395, 4636]
    return
end

function test_time_limit()
    n = 10
    W = 2137.0
    C = Float64[
        566 611 506 180 817 184 585 423 26 317
        62 84 977 979 874 54 269 93 881 563
    ]
    w = Float64[557, 898, 148, 63, 78, 964, 246, 662, 386, 272]
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.Chalmet())
    MOI.set(model, MOI.Silent(), true)
    MOI.set(model, MOI.TimeLimitSec(), 0.0)
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
            MOI.VectorAffineTerm(i, MOI.ScalarAffineTerm(C[i, j], x[j])) for
            i in 1:2 for j in 1:n
        ],
        [0.0, 0.0],
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.TIME_LIMIT
    @test MOI.get(model, MOI.ResultCount()) > 0
    return
end

function test_unbounded()
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.Chalmet())
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

function test_infeasible()
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.Chalmet())
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

function test_vector_of_variables_objective()
    model = MOI.instantiate(; with_bridge_type = Float64) do
        return MOA.Optimizer(HiGHS.Optimizer)
    end
    MOI.set(model, MOA.Algorithm(), MOA.Chalmet())
    MOI.set(model, MOI.Silent(), true)
    MOI.set(model, MOA.ComputeIdealPoint(), false)
    x = MOI.add_variables(model, 2)
    MOI.add_constraint.(model, x, MOI.ZeroOne())
    f = MOI.VectorOfVariables(x)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.add_constraint(model, sum(1.0 * xi for xi in x), MOI.GreaterThan(1.0))
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    ideal_point = MOI.get(model, MOI.ObjectiveBound())
    @test length(ideal_point) == 2 && all(isnan, ideal_point)
    return
end

function test_too_many_objectives()
    P = Float64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.Chalmet())
    x = MOI.add_variables(model, 4)
    MOI.add_constraint.(model, x, MOI.GreaterThan(0.0))
    MOI.add_constraint.(model, x, MOI.LessThan(1.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    f = MOI.Utilities.operate(vcat, Float64, P * x...)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    @test_throws(
        ErrorException("Chalmet requires exactly two objectives"),
        MOI.optimize!(model),
    )
    return
end

end  # module TestChalmet

TestChalmet.run_tests()
