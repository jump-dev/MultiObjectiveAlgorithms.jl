
module TestGeneralDichotomy

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


function test_lap()

    # toy instance from ?
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
    f = MOI.Utilities.operate(vcat, Float64, sum(costs[1,:,:] .* x), sum(costs[2,:,:] .* x), sum(costs[3,:,:] .* x))
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)


    println("model: ")
    println(model)

    precision = 3

    algorithm = MOA.GeneralDichotomy(precision)
    algorithm.verbose = 1

    # algorithm = MOA.RandomWeighting()

    MOI.set(model, MOA.Algorithm(), algorithm)

    # MOI.set(optimizer, MOA.SolutionLimit(), 3)

    # costs = load_instance(model, fname)

    println("Costs size: ", costs.size)


    # disable model log
    # set_attribute(model, MOI.Silent(), true)

    MOI.optimize!(model)

    solve_time = MOI.get(model, MOI.SolveTimeSec())

    println("Number of solutions: ", MOI.get(model, MOI.ResultCount()))
    println("solve time: ", solve_time)
    println("solve time inner: ", optimizer.solve_time_inner)

    generaldichotomy_time = optimizer.solve_time-optimizer.solve_time_inner
    println("GeneralDichotomy time: ", generaldichotomy_time, " s")

    println("n weights: ", general_dichotomy.weights.size)
    println("n intermediate weights: ", general_dichotomy.n_interm_weights)
    println("N solved scalarization: ", general_dichotomy.n_call_solve)

end

# test on knapsack

# test on bi-objective problem

# test on min and max

end  # module TestGeneralDichotomy

TestGeneralDichotomy.run_tests()

