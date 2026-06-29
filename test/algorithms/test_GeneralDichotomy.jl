
module TestGeneralDichotomy

using Test

import HiGHS
import MultiObjectiveAlgorithms as MOA
import MultiObjectiveAlgorithms: MOI

import Polyhedra

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

function test_Dichotomy_SolutionLimit() # from the Dichotomy test set
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.GeneralDichotomy())
    @test MOI.supports(MOA.GeneralDichotomy(), MOA.SolutionLimit())
    @test MOI.supports(model, MOA.SolutionLimit())
    @test MOI.get(model, MOA.SolutionLimit()) ==
          MOA._default(MOA.SolutionLimit())
    MOI.set(model, MOA.SolutionLimit(), 1)
    @test MOI.get(model, MOA.SolutionLimit()) == 1
    return
end

function test_lap() # toy instance from (Przybylski et al., 2019)
    costs = zeros(Float64, (3, 4, 4))
    costs[1, :, :] = [3 6 4 5; 2 3 5 4; 3 5 4 2; 4 5 3 6]
    costs[2, :, :] = [2 3 5 4; 5 3 4 3; 5 2 6 4; 4 5 2 5]
    costs[3, :, :] = [4 2 4 2; 4 2 4 6; 4 2 6 3; 2 4 5 3]
    n = costs.size[2] # 4 variables
    d = costs.size[1] # 3 objectives
    model = MOA.Optimizer(HiGHS.Optimizer)
    x_ = MOI.add_variables(model, n * n)
    x = reshape(x_, n, n)
    MOI.add_constraints(
        model,
        MOI.ScalarAffineFunction.(
            [MOI.ScalarAffineTerm.(ones(n), x[i, :]) for i in 1:n],
            0.0,
        ),
        MOI.EqualTo(1.0),
    )
    MOI.add_constraints(
        model,
        MOI.ScalarAffineFunction.(
            [MOI.ScalarAffineTerm.(ones(n), x[:, j]) for j in 1:n],
            0.0,
        ),
        MOI.EqualTo(1.0),
    )
    MOI.add_constraint.(model, x, MOI.ZeroOne())
    f = MOI.Utilities.operate(
        vcat,
        Float64,
        sum(costs[1, :, :] .* x),
        sum(costs[2, :, :] .* x),
        sum(costs[3, :, :] .* x),
    )
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.Silent(), true)
    precision = 3
    algorithm = MOA.GeneralDichotomy(precision)
    MOI.set(model, MOA.Algorithm(), algorithm)
    MOI.optimize!(model)
    solve_time = MOI.get(model, MOI.SolveTimeSec())
    @test MOI.get(model, MOI.ResultCount()) == 4
end

function test_lap_2() # testing weight set cleaning on a random lap instance
    costs = zeros(Float64, (4, 5, 5))
    costs[1, :, :] = [
        77 89 53 70 92;
        89 89 13 4 41;
        7 7 34 45 87;
        65 0 99 23 93;
        93 58 93 18 17
    ]
    costs[2, :, :] = [
        26 20 71 34 57;
        58 42 66 22 87;
        93 85 87 0 42;
        26 2 38 57 2;
        62 39 99 42 85
    ]
    costs[3, :, :] = [
        36 48 65 49 13;
        13 39 17 75 83;
        10 17 82 99 73;
        13 95 62 26 6;
        21 10 19 36 27
    ]
    costs[4, :, :] = [
        89 78 20 24 55;
        59 20 64 1 19;
        7 27 63 91 1;
        20 26 64 80 60;
        21 21 13 92 73
    ]
    n = costs.size[2] # 5 variables
    d = costs.size[1] # 4 objectives
    model = MOA.Optimizer(HiGHS.Optimizer)
    x_ = MOI.add_variables(model, n * n)
    x = reshape(x_, n, n)
    MOI.add_constraints(
        model,
        MOI.ScalarAffineFunction.(
            [MOI.ScalarAffineTerm.(ones(n), x[i, :]) for i in 1:n],
            0.0,
        ),
        MOI.EqualTo(1.0),
    )
    MOI.add_constraints(
        model,
        MOI.ScalarAffineFunction.(
            [MOI.ScalarAffineTerm.(ones(n), x[:, j]) for j in 1:n],
            0.0,
        ),
        MOI.EqualTo(1.0),
    )
    MOI.add_constraint.(model, x, MOI.ZeroOne())
    f = MOI.Utilities.operate(
        vcat,
        Float64,
        sum(costs[1, :, :] .* x),
        sum(costs[2, :, :] .* x),
        sum(costs[3, :, :] .* x),
    )
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.Silent(), true)
    MOI.set(model, MOA.Algorithm(), MOA.GeneralDichotomy())
    MOI.optimize!(model)
    solve_time = MOI.get(model, MOI.SolveTimeSec())
    @test MOI.get(model, MOI.ResultCount()) == 9
end

function test_vlp() # test instance from Bensolve (http://www.bensolve.org/)
    P = [1.0 0.0 1.0; 1.0 1.0 0.0; 0.0 1.0 1.0]
    B = [1.0 1.0 1.0; 1.0 2.0 2.0; 2.0 2.0 1.0; 2 1.0 2.0]
    a = [1.0, 3/2, 3/2, 3/2]

    # println("Test Vector linear program")
    # println("P")
    # println(P)
    # println("B")
    # println(B)
    # println("a")
    # println(a)

    model = MOA.Optimizer(HiGHS.Optimizer)
    x = MOI.add_variables(model, 3)

    MOI.add_constraints(
        model,
        MOI.ScalarAffineFunction.(
            [MOI.ScalarAffineTerm.(B[i, :], x) for i in 1:4],
            0.0,
        ),
        MOI.GreaterThan.(a),
    )
    MOI.add_constraint.(model, x, MOI.GreaterThan(0.0))

    f = MOI.Utilities.vectorize(P' * x)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)

    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.Silent(), true)

    # println("vlp model:")
    # println(model)

    precision = 3
    algorithm = MOA.GeneralDichotomy(precision)
    algorithm.verbose = 0
    MOI.set(model, MOA.Algorithm(), algorithm)
    return MOI.optimize!(model)

    # for i in 1:MOI.get(model, MOI.ResultCount())
    #     println(MOI.get(model, MOI.ObjectiveValue(i)))
    # end

    # solve_time = MOI.get(model, MOI.SolveTimeSec())
    # @test  MOI.get(model, MOI.ResultCount()) == 4 
end

function test_biobjective_knapsack() # from the Dichotomy test set
    p1 = [77, 94, 71, 63, 96, 82, 85, 75, 72, 91, 99, 63, 84, 87, 79, 94, 90]
    p2 = [65, 90, 90, 77, 95, 84, 70, 94, 66, 92, 74, 97, 60, 60, 65, 97, 93]
    w = [80, 87, 68, 72, 66, 77, 99, 85, 70, 93, 98, 72, 100, 89, 67, 86, 91]
    precision = 3
    general_dichotomy = MOA.GeneralDichotomy(precision)
    general_dichotomy.verbose = 0
    f = MOI.OptimizerWithAttributes(
        () -> MOA.Optimizer(HiGHS.Optimizer),
        MOA.Algorithm() => general_dichotomy,
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

function test_infeasible() # from the Dichotomy test set
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.GeneralDichotomy(3))
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, 6)
    MOI.add_constraint.(model, x, MOI.GreaterThan(0.0))
    MOI.add_constraint(model, 1.0 * x[1] + 1.0 * x[2], MOI.LessThan(-1.0))
    f = MOI.Utilities.operate(vcat, Float64, 1.0 .* x...)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.INFEASIBLE
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.NO_SOLUTION
    @test MOI.get(model, MOI.DualStatus()) == MOI.NO_SOLUTION
    return
end

function test_unbounded() # from the Dichotomy test set
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.GeneralDichotomy(3))
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, 3)
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

end  # module TestGeneralDichotomy

TestGeneralDichotomy.run_tests()
