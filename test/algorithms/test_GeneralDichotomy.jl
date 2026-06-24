
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


function test_lap() # toy instance from (Przybylski et al., 2019)
    costs = zeros(Float64, (3, 4, 4))
    costs[1,:,:] = [3 6 4 5; 2 3 5 4; 3 5 4 2; 4 5 3 6]
    costs[2,:,:] = [2 3 5 4; 5 3 4 3; 5 2 6 4; 4 5 2 5]
    costs[3,:,:] = [4 2 4 2; 4 2 4 6; 4 2 6 3; 2 4 5 3]
    n = costs.size[2] # 4 variables
    d = costs.size[1] # 3 objectives
    model = MOA.Optimizer(HiGHS.Optimizer)
    x_ = MOI.add_variables(model, n * n)
    x = reshape(x_, n, n)
    MOI.add_constraints(model, MOI.ScalarAffineFunction.([MOI.ScalarAffineTerm.(ones(n), x[i,:]) for i in 1:n], 0.0), MOI.EqualTo(1.0))
    MOI.add_constraints(model, MOI.ScalarAffineFunction.([MOI.ScalarAffineTerm.(ones(n), x[:,j]) for j in 1:n], 0.0), MOI.EqualTo(1.0))
    MOI.add_constraint.(model, x, MOI.ZeroOne())
    f = MOI.Utilities.operate(vcat, Float64, sum(costs[1,:,:] .* x), sum(costs[2,:,:] .* x), sum(costs[3,:,:] .* x))
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    precision = 6
    algorithm = MOA.GeneralDichotomy(precision)
    MOI.set(model, MOA.Algorithm(), algorithm)
    MOI.optimize!(model)
    solve_time = MOI.get(model, MOI.SolveTimeSec())
    @test  MOI.get(model, MOI.ResultCount()) == 4 
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

end  # module TestGeneralDichotomy

TestGeneralDichotomy.run_tests()

