#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

module TestTambyVanderpooten

using Test

import HiGHS
import MultiObjectiveAlgorithms as MOA
import MultiObjectiveAlgorithms: MOI

include(joinpath(dirname(@__DIR__), "mock_optimizer.jl"))
include(joinpath(dirname(@__DIR__), "problems.jl"))
include(joinpath(dirname(@__DIR__), "vOptLib.jl"))

function run_tests()
    is_test(name) = startswith("$name", "test_")
    @testset "$name" for name in filter(is_test, names(@__MODULE__; all = true))
        getfield(@__MODULE__, name)()
    end
    return
end

function test_problems()
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.TambyVanderpooten())
    MOI.set(model, MOI.Silent(), true)
    Problems.run_tests(model)
    return
end

function test_vOptLib_runtests()
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.TambyVanderpooten())
    MOI.set(model, MOI.Silent(), true)
    vOptLib.run_tests(model)
    return
end

function test_infeasible()
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.TambyVanderpooten())
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
    MOI.set(model, MOA.Algorithm(), MOA.TambyVanderpooten())
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
    MOI.set(model, MOA.Algorithm(), MOA.TambyVanderpooten())
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
    MOI.set(model, MOA.Algorithm(), MOA.TambyVanderpooten())
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
    @test MOI.get(model, MOI.ResultCount()) == 0
    return
end

function test_vector_of_variables_objective()
    model = MOI.instantiate(; with_bridge_type = Float64) do
        return MOA.Optimizer(HiGHS.Optimizer)
    end
    MOI.set(model, MOA.Algorithm(), MOA.TambyVanderpooten())
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, 2)
    MOI.add_constraint.(model, x, MOI.ZeroOne())
    f = MOI.VectorOfVariables(x)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.add_constraint(model, sum(1.0 * xi for xi in x), MOI.GreaterThan(1.0))
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    return
end

function test_solve_failures()
    m, n = 2, 10
    p1 = [5.0 1 10 8 3 5 3 3 7 2; 10 6 1 6 8 3 2 10 6 1]
    p2 = [4.0 6 4 3 1 6 8 2 9 7; 8 8 8 2 4 8 8 1 10 1]
    w = [5.0 9 3 5 10 5 7 10 7 8; 4 8 8 6 10 8 10 7 5 1]
    b = [34.0, 33.0]
    for fail_after in 0:5
        model = MOA.Optimizer(mock_optimizer(fail_after))
        MOI.set(model, MOA.Algorithm(), MOA.TambyVanderpooten())
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

function _solve_noisy_mock(mock, counter::Ref{Int})
    highs = HiGHS.Optimizer()
    MOI.set(highs, MOI.Silent(), true)
    index_map = MOI.copy_to(highs, mock)
    MOI.optimize!(highs)
    x = [index_map[xi] for xi in MOI.get(mock, MOI.ListOfVariableIndices())]
    # Perturb the solution by a deterministic amount that is within the
    # tolerances that a typical MIP solver might return.
    noise() = 1e-6 * sin(counter[] += 1)
    x_primal = [xi + noise() for xi in MOI.get(highs, MOI.VariablePrimal(), x)]
    status = MOI.get(highs, MOI.TerminationStatus())
    MOI.Utilities.mock_optimize!(mock, status, x_primal)
    obj = MOI.get(highs, MOI.ObjectiveValue())
    MOI.set(mock, MOI.ObjectiveValue(), obj + noise())
    return
end

function test_noisy_solutions()
    C = [
        [3 13 3 20 9; 4 13 3 4 9; 18 9 3 9 17; 1 16 16 11 3; 18 4 7 13 11],
        [1 12 4 6 20; 18 5 13 20 7; 9 3 3 2 15; 11 18 1 7 13; 6 7 3 15 18],
        [13 13 13 14 15; 16 3 20 7 13; 12 17 9 8 3; 6 10 4 18 5; 4 9 7 14 11],
    ]
    n = size(C[1], 1)
    solutions = Dict{Bool,Set{Vector{Int}}}()
    for noisy in (false, true)
        counter = Ref(0)
        model = MOA.Optimizer() do
            mock = MOI.Utilities.MockOptimizer(MOI.Utilities.Model{Float64}())
            MOI.Utilities.set_mock_optimize!(
                mock,
                m -> noisy ? _solve_noisy_mock(m, counter) : _solve_mock(m),
            )
            return mock
        end
        MOI.set(model, MOA.Algorithm(), MOA.TambyVanderpooten())
        MOI.set(model, MOI.Silent(), true)
        MOI.set(model, MOI.TimeLimitSec(), 60.0)
        x = reshape(MOI.add_variables(model, n * n), n, n)
        MOI.add_constraint.(model, x, MOI.ZeroOne())
        for i in 1:n
            MOI.add_constraint(model, sum(1.0 .* x[i, :]), MOI.EqualTo(1.0))
            MOI.add_constraint(model, sum(1.0 .* x[:, i]), MOI.EqualTo(1.0))
        end
        f_k = [sum(1.0 * C[k] .* x) for k in 1:3]
        f = MOI.Utilities.operate(vcat, Float64, f_k...)
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
        MOI.optimize!(model)
        @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
        N = MOI.get(model, MOI.ResultCount())
        Y_float = [MOI.get(model, MOI.ObjectiveValue(i)) for i in 1:N]
        # The algorithm rounds internally, but we return the solver's values.
        @test any(y -> any(!isinteger, y), Y_float) == noisy
        Y = [round.(Int, y) for y in Y_float]
        @test allunique(Y)
        solutions[noisy] = Set(Y)
    end
    @test solutions[false] == solutions[true]
    return
end

end  # module TestTambyVanderpooten

TestTambyVanderpooten.run_tests()
