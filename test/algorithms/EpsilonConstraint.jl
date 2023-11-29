#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

module TestEpsilonConstraint

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

function test_biobjective_knapsack()
    p1 = [77, 94, 71, 63, 96, 82, 85, 75, 72, 91, 99, 63, 84, 87, 79, 94, 90]
    p2 = [65, 90, 90, 77, 95, 84, 70, 94, 66, 92, 74, 97, 60, 60, 65, 97, 93]
    w = [80, 87, 68, 72, 66, 77, 99, 85, 70, 93, 98, 72, 100, 89, 67, 86, 91]
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.EpsilonConstraint())
    MOI.set(model, MOA.SolutionLimit(), 100)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, length(w))
    MOI.add_constraint.(model, x, MOI.ZeroOne())
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    f = MOI.Utilities.operate(
        vcat,
        Float64,
        [sum(1.0 * p[i] * x[i] for i in 1:length(w)) for p in [p1, p2]]...,
    )
    f.constants[1] = 1.0
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.add_constraint(
        model,
        sum(1.0 * w[i] * x[i] for i in 1:length(w)),
        MOI.LessThan(900.0),
    )
    MOI.optimize!(model)
    results = Dict(
        [956, 906] => [2, 3, 5, 6, 9, 10, 11, 14, 15, 16, 17],
        [950, 915] => [1, 2, 5, 6, 8, 9, 10, 11, 15, 16, 17],
        [949, 939] => [1, 2, 3, 5, 6, 8, 10, 11, 15, 16, 17],
        [944, 940] => [2, 3, 5, 6, 8, 9, 10, 11, 15, 16, 17],
        [937, 942] => [1, 2, 3, 5, 6, 10, 11, 12, 15, 16, 17],
        [936, 947] => [2, 5, 6, 8, 9, 10, 11, 12, 15, 16, 17],
        [935, 971] => [2, 3, 5, 6, 8, 10, 11, 12, 15, 16, 17],
        [928, 972] => [2, 3, 5, 6, 8, 9, 10, 11, 12, 16, 17],
        [919, 983] => [2, 3, 4, 5, 6, 8, 10, 11, 12, 16, 17],
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

function test_biobjective_knapsack_atol_large()
    p1 = [77, 94, 71, 63, 96, 82, 85, 75, 72, 91, 99, 63, 84, 87, 79, 94, 90]
    p2 = [65, 90, 90, 77, 95, 84, 70, 94, 66, 92, 74, 97, 60, 60, 65, 97, 93]
    w = [80, 87, 68, 72, 66, 77, 99, 85, 70, 93, 98, 72, 100, 89, 67, 86, 91]
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.EpsilonConstraint())
    @test MOI.supports(model, MOA.EpsilonConstraintStep())
    MOI.set(model, MOA.EpsilonConstraintStep(), 10.0)
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
        [948, 939] => [1, 2, 3, 5, 6, 8, 10, 11, 15, 16, 17],
        [934, 971] => [2, 3, 5, 6, 8, 10, 11, 12, 15, 16, 17],
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
    @test MOI.supports(model, MOA.SolutionLimit())
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
        [943, 940] => [2, 3, 5, 6, 8, 9, 10, 11, 15, 16, 17],
        [918, 983] => [2, 3, 4, 5, 6, 8, 10, 11, 12, 16, 17],
    )
    @test MOI.get(model, MOI.ResultCount()) == 2
    for i in 1:MOI.get(model, MOI.ResultCount())
        x_sol = MOI.get(model, MOI.VariablePrimal(i), x)
        X = findall(elt -> elt > 0.9, x_sol)
        Y = MOI.get(model, MOI.ObjectiveValue(i))
        @test results[round.(Int, Y)] == X
    end
    return
end

function test_infeasible()
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.EpsilonConstraint())
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
    MOI.set(model, MOA.Algorithm(), MOA.EpsilonConstraint())
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

function test_unbounded_second()
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.EpsilonConstraint())
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, 2)
    MOI.add_constraint.(model, x, MOI.GreaterThan(0.0))
    MOI.add_constraint(model, x[1], MOI.LessThan(1.0))
    f = MOI.Utilities.operate(vcat, Float64, 1.0 .* x...)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.DUAL_INFEASIBLE
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.NO_SOLUTION
    @test MOI.get(model, MOI.DualStatus()) == MOI.NO_SOLUTION
    return
end

function test_deprecated()
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.EpsilonConstraint())
    @test MOI.supports(model, MOA.ObjectiveAbsoluteTolerance(1))
    @test_logs (:warn,) MOI.set(model, MOA.ObjectiveAbsoluteTolerance(1), 1.0)
    @test_logs (:warn,) MOI.get(model, MOA.ObjectiveAbsoluteTolerance(1))
    return
end

function test_quadratic()
    μ = [0.05470748600000001, 0.18257110599999998]
    Q = [0.00076204 0.00051972; 0.00051972 0.00546173]
    N = 2
    model = MOA.Optimizer(Ipopt.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.EpsilonConstraint())
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
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    return
end

function test_poor_numerics()
    μ = [0.006898463772627643, -0.02972609131603086]
    Q = [0.030446 0.00393731; 0.00393731 0.00713285]
    N = 2
    model = MOA.Optimizer(Ipopt.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.EpsilonConstraint())
    MOI.set(model, MOA.SolutionLimit(), 10)
    MOI.set(model, MOI.Silent(), true)
    w = MOI.add_variables(model, N)
    sharpe = MOI.add_variable(model)
    MOI.add_constraint.(model, w, MOI.GreaterThan(0.0))
    MOI.add_constraint.(model, w, MOI.LessThan(1.0))
    MOI.add_constraint(model, sum(1.0 * w[i] for i in 1:N), MOI.EqualTo(1.0))
    variance = Expr(:call, :+)
    for i in 1:N, j in 1:N
        push!(variance.args, Expr(:call, :*, Q[i, j], w[i], w[j]))
    end
    nlp = MOI.Nonlinear.Model()
    MOI.Nonlinear.add_constraint(
        nlp,
        :(($(μ[1]) * $(w[1]) + $(μ[2]) * $(w[2])) / sqrt($variance) - $sharpe),
        MOI.EqualTo(0.0),
    )
    evaluator = MOI.Nonlinear.Evaluator(
        nlp,
        MOI.Nonlinear.SparseReverseMode(),
        [w; sharpe],
    )
    MOI.set(model, MOI.NLPBlock(), MOI.NLPBlockData(evaluator))
    f = MOI.Utilities.operate(vcat, Float64, μ' * w, sharpe)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.ResultCount()) == 1
    for i in 1:MOI.get(model, MOI.ResultCount())
        w_sol = MOI.get(model, MOI.VariablePrimal(i), w)
        sharpe_sol = MOI.get(model, MOI.VariablePrimal(i), sharpe)
        y = MOI.get(model, MOI.ObjectiveValue(i))
        @test y ≈ [μ' * w_sol, sharpe_sol]
    end
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    return
end

function test_vectornonlinearfunction()
    μ = [0.006898463772627643, -0.02972609131603086]
    Q = [0.030446 0.00393731; 0.00393731 0.00713285]
    N = 2
    model = MOA.Optimizer(Ipopt.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.EpsilonConstraint())
    MOI.set(model, MOA.SolutionLimit(), 10)
    MOI.set(model, MOI.Silent(), true)
    w = MOI.add_variables(model, N)
    MOI.add_constraint.(model, w, MOI.GreaterThan(0.0))
    MOI.add_constraint.(model, w, MOI.LessThan(1.0))
    MOI.add_constraint(model, sum(1.0 * w[i] for i in 1:N), MOI.EqualTo(1.0))
    f = MOI.VectorNonlinearFunction([
        μ' * w,
        MOI.ScalarNonlinearFunction(
            :/,
            Any[μ'*w, MOI.ScalarNonlinearFunction(:sqrt, Any[w'*Q*w])],
        ),
    ])
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.ResultCount()) >= 1
    for i in 1:MOI.get(model, MOI.ResultCount())
        w_sol = MOI.get(model, MOI.VariablePrimal(i), w)
        y = MOI.get(model, MOI.ObjectiveValue(i))
        @test y ≈ [μ' * w_sol, (μ' * w_sol) / sqrt(w_sol' * Q * w_sol)]
    end
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    return
end

function test_time_limit()
    p1 = [77, 94, 71, 63, 96, 82, 85, 75, 72, 91, 99, 63, 84, 87, 79, 94, 90]
    p2 = [65, 90, 90, 77, 95, 84, 70, 94, 66, 92, 74, 97, 60, 60, 65, 97, 93]
    w = [80, 87, 68, 72, 66, 77, 99, 85, 70, 93, 98, 72, 100, 89, 67, 86, 91]
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.EpsilonConstraint())
    MOI.set(model, MOI.TimeLimitSec(), 0.0)
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
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.TIME_LIMIT
    @test MOI.get(model, MOI.ResultCount()) == 0
    return
end

function test_time_limit_large()
    p1 = [77, 94, 71, 63, 96, 82, 85, 75, 72, 91, 99, 63, 84, 87, 79, 94, 90]
    p2 = [65, 90, 90, 77, 95, 84, 70, 94, 66, 92, 74, 97, 60, 60, 65, 97, 93]
    w = [80, 87, 68, 72, 66, 77, 99, 85, 70, 93, 98, 72, 100, 89, 67, 86, 91]
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.EpsilonConstraint())
    MOI.set(model, MOI.TimeLimitSec(), 1.0)
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
    @test MOI.get(model, MOI.ResultCount()) >= 0
    return
end

function test_vector_of_variables_objective()
    model = MOI.instantiate(; with_bridge_type = Float64) do
        return MOA.Optimizer(HiGHS.Optimizer)
    end
    MOI.set(model, MOA.Algorithm(), MOA.EpsilonConstraint())
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

TestEpsilonConstraint.run_tests()
