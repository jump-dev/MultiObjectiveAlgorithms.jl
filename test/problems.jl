#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

module Problems

using Test
import MathOptInterface as MOI
import MultiObjectiveAlgorithms as MOA

function run_tests(model::MOI.ModelLike)
    for name in names(@__MODULE__; all = true)
        if startswith("$name", "test_")
            @testset "$name" begin
                MOI.empty!(model)
                getfield(@__MODULE__, name)(model)
            end
        end
    end
    return
end

function test_problem_knapsack_min_p3(model)
    p = 3
    n = 10
    W = 2137.0
    C = Float64[
        566 611 506 180 817 184 585 423 26 317
        62 84 977 979 874 54 269 93 881 563
        664 982 962 140 224 215 12 869 332 537
    ]
    w = Float64[557, 898, 148, 63, 78, 964, 246, 662, 386, 272]
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
        ones(p),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    results = [
        [1, 0, 1, 1, 1, 0, 1, 1, 0, 1] => [-3394, -3817, -3408],
        [0, 1, 1, 1, 1, 0, 1, 0, 1, 1] => [-3042, -4627, -3189],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 1] => [-2997, -3539, -3509],
        [0, 0, 1, 1, 1, 0, 1, 1, 1, 1] => [-2854, -4636, -3076],
        [0, 1, 1, 1, 1, 0, 0, 1, 0, 1] => [-2854, -3570, -3714],
        [1, 1, 1, 1, 1, 0, 0, 0, 1, 0] => [-2706, -3857, -3304],
        [1, 0, 1, 1, 1, 0, 0, 1, 1, 0] => [-2518, -3866, -3191],
    ]
    @assert MOI.get(model, MOI.ResultCount()) == length(results)
    for i in 1:length(results)
        X = MOI.get(model, MOI.VariablePrimal(i), x)
        Y = MOI.get(model, MOI.ObjectiveValue(i))
        @test isapprox(results[i][1], X; atol = 1e-6)
        @test isapprox(results[i][2] .+ 1, Y; atol = 1e-6)
    end
    @test ≈(
        MOI.get(model, MOI.ObjectiveBound()),
        vec(minimum(mapreduce(last, hcat, results); dims = 2)) .+ 1,
    )
    return
end

function test_problem_knapsack_max_p3(model)
    p = 3
    n = 10
    W = 2137.0
    C = Float64[
        566 611 506 180 817 184 585 423 26 317
        62 84 977 979 874 54 269 93 881 563
        664 982 962 140 224 215 12 869 332 537
    ]
    w = Float64[557, 898, 148, 63, 78, 964, 246, 662, 386, 272]
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
            i in 1:p for j in 1:n
        ],
        fill(0.0, p),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    results = [
        [1, 0, 1, 1, 1, 0, 0, 1, 1, 0] => [2518, 3866, 3191],
        [1, 1, 1, 1, 1, 0, 0, 0, 1, 0] => [2706, 3857, 3304],
        [0, 1, 1, 1, 1, 0, 0, 1, 0, 1] => [2854, 3570, 3714],
        [0, 0, 1, 1, 1, 0, 1, 1, 1, 1] => [2854, 4636, 3076],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 1] => [2997, 3539, 3509],
        [0, 1, 1, 1, 1, 0, 1, 0, 1, 1] => [3042, 4627, 3189],
        [1, 0, 1, 1, 1, 0, 1, 1, 0, 1] => [3394, 3817, 3408],
    ]
    reverse!(results)
    N = MOI.get(model, MOI.ResultCount())
    @assert N == length(results)
    for i in 1:length(results)
        X = MOI.get(model, MOI.VariablePrimal(i), x)
        Y = MOI.get(model, MOI.ObjectiveValue(i))
        @test isapprox(results[i][1], X; atol = 1e-6)
        @test isapprox(results[i][2], Y; atol = 1e-6)
    end
    @test ≈(
        MOI.get(model, MOI.ObjectiveBound()),
        vec(maximum(mapreduce(last, hcat, results); dims = 2)),
    )
    return
end

function test_problem_knapsack_min_p4(model)
    p = 4
    n = 10
    W = 2653.0
    C = Float64[
        566 611 506 180 817 184 585 423 26 317
        62 84 977 979 874 54 269 93 881 563
        664 982 962 140 224 215 12 869 332 537
        557 898 148 63 78 964 246 662 386 272
    ]
    w = Float64[979 448 355 955 426 229 9 695 322 889]
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
    results = reverse([
        [0, 0, 1, 1, 0, 1, 1, 1, 1, 0] => [1904, 3253, 2530, 2469],
        [0, 1, 1, 1, 0, 1, 1, 0, 1, 0] => [2092, 3244, 2643, 2705],
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 1] => [2146, 1944, 2947, 3428],
        [0, 0, 1, 1, 1, 1, 1, 0, 1, 0] => [2298, 4034, 1885, 1885],
        [0, 0, 1, 0, 1, 1, 1, 0, 1, 1] => [2435, 3618, 2282, 2094],
        [0, 1, 1, 1, 1, 0, 1, 0, 1, 0] => [2725, 4064, 2652, 1819],
        [0, 1, 1, 0, 1, 0, 1, 0, 1, 1] => [2862, 3648, 3049, 2028],
        [0, 1, 1, 1, 1, 1, 1, 0, 0, 0] => [2883, 3237, 2535, 2397],
        [0, 1, 1, 0, 1, 1, 1, 1, 1, 0] => [3152, 3232, 3596, 3382],
        [1, 1, 1, 0, 1, 1, 1, 0, 0, 0] => [3269, 2320, 3059, 2891],
    ])
    @test MOI.get(model, MOI.ResultCount()) == length(results)
    for (i, (x_sol, y_sol)) in enumerate(results)
        @test ≈(x_sol, MOI.get(model, MOI.VariablePrimal(i), x); atol = 1e-6)
        @test ≈(-y_sol, MOI.get(model, MOI.ObjectiveValue(i)); atol = 1e-6)
    end
    @test MOI.get(model, MOI.ObjectiveBound()) ≈ -[3269, 4064, 3596, 3428]
    return
end

function test_problem_knapsack_max_p4(model)
    p = 4
    n = 10
    W = 2653.0
    C = Float64[
        566 611 506 180 817 184 585 423 26 317
        62 84 977 979 874 54 269 93 881 563
        664 982 962 140 224 215 12 869 332 537
        557 898 148 63 78 964 246 662 386 272
    ]
    w = Float64[979 448 355 955 426 229 9 695 322 889]
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
            i in 1:p for j in 1:n
        ],
        fill(0.0, p),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    results = [
        [0, 0, 1, 1, 0, 1, 1, 1, 1, 0] => [1904, 3253, 2530, 2469],
        [0, 1, 1, 1, 0, 1, 1, 0, 1, 0] => [2092, 3244, 2643, 2705],
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 1] => [2146, 1944, 2947, 3428],
        [0, 0, 1, 1, 1, 1, 1, 0, 1, 0] => [2298, 4034, 1885, 1885],
        [0, 0, 1, 0, 1, 1, 1, 0, 1, 1] => [2435, 3618, 2282, 2094],
        [0, 1, 1, 1, 1, 0, 1, 0, 1, 0] => [2725, 4064, 2652, 1819],
        [0, 1, 1, 0, 1, 0, 1, 0, 1, 1] => [2862, 3648, 3049, 2028],
        [0, 1, 1, 1, 1, 1, 1, 0, 0, 0] => [2883, 3237, 2535, 2397],
        [0, 1, 1, 0, 1, 1, 1, 1, 1, 0] => [3152, 3232, 3596, 3382],
        [1, 1, 1, 0, 1, 1, 1, 0, 0, 0] => [3269, 2320, 3059, 2891],
    ]
    reverse!(results)
    @test MOI.get(model, MOI.ResultCount()) == length(results)
    for (i, (x_sol, y_sol)) in enumerate(results)
        @test ≈(x_sol, MOI.get(model, MOI.VariablePrimal(i), x); atol = 1e-6)
        @test ≈(y_sol, MOI.get(model, MOI.ObjectiveValue(i)); atol = 1e-6)
    end
    @test MOI.get(model, MOI.ObjectiveBound()) ≈ [3269, 4064, 3596, 3428]
    return
end

function test_problem_assignment_min_p3(model)
    p = 3
    n = 5
    C = Float64[
        6 1 20 2 3
        2 6 9 10 18
        1 6 20 5 9
        6 8 6 9 6
        7 10 10 6 2
        17 20 8 8 20
        10 13 1 10 15
        4 11 1 13 1
        19 13 7 18 17
        15 3 5 1 11
        10 7 1 19 12
        2 15 12 10 3
        11 20 16 12 9
        10 15 20 11 7
        1 9 20 7 6
    ]
    C = permutedims(reshape(C, (n, p, n)), [2, 1, 3])
    x = [MOI.add_variable(model) for i in 1:n, j in 1:n]
    MOI.add_constraint.(model, x, MOI.ZeroOne())
    for i in 1:n
        MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, x[i, j]) for j in 1:n],
                0.0,
            ),
            MOI.EqualTo(1.0),
        )
    end
    for j in 1:n
        MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, x[i, j]) for i in 1:n],
                0.0,
            ),
            MOI.EqualTo(1.0),
        )
    end
    f = MOI.VectorAffineFunction(
        [
            MOI.VectorAffineTerm(k, MOI.ScalarAffineTerm(C[k, i, j], x[i, j])) for k in 1:p for i in 1:n for j in 1:n
        ],
        fill(0.0, p),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    results = [
        [0 1 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 1] => [16, 61, 47],
        [0 0 1 0 0 0 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1] => [17, 43, 71],
        [0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1] => [18, 47, 67],
        [0 0 1 0 0 1 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 1] => [20, 52, 54],
        [0 0 1 0 0 0 0 0 1 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 1] => [22, 37, 63],
        [0 0 1 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 1] => [22, 54, 47],
        [0 0 1 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 1 0] => [23, 43, 44],
        [0 1 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 1 0 0] => [24, 39, 45],
        [0 0 1 0 0 0 0 0 0 1 0 1 0 0 0 1 0 0 0 0 0 0 0 1 0] => [28, 33, 58],
        [0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0] => [28, 66, 39],
        [0 1 0 0 0 0 0 0 0 1 0 0 0 1 0 1 0 0 0 0 0 0 1 0 0] => [29, 29, 59],
        [0 1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1] => [34, 60, 42],
        [0 0 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 0 0 0 0 1 0 0] => [35, 38, 56],
        [0 1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 1 0] => [35, 49, 39],
        [0 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1] => [37, 55, 36],
        [1 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 1 0 0 1 0 0] => [38, 33, 53],
        [0 0 1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 0] => [39, 43, 41],
        [0 1 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 0] => [40, 47, 37],
        [0 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 0 0 0 0 0 1 0] => [43, 51, 31],
        [0 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1 0 0 1 0 0] => [45, 33, 34],
        [0 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 0 0 0 1 0 0] => [50, 40, 32],
    ]
    @test MOI.get(model, MOI.ResultCount()) == length(results)
    for (i, (x_sol, y_sol)) in enumerate(results)
        x_primal = MOI.get(model, MOI.VariablePrimal(i), vec(x))
        @test ≈(vec(x_sol), x_primal; atol = 1e-6)
        @test ≈(y_sol, MOI.get(model, MOI.ObjectiveValue(i)); atol = 1e-6)
    end
    return
end

function test_problem_assignment_max_p3(model)
    p = 3
    n = 5
    C = Float64[
        6 1 20 2 3
        2 6 9 10 18
        1 6 20 5 9
        6 8 6 9 6
        7 10 10 6 2
        17 20 8 8 20
        10 13 1 10 15
        4 11 1 13 1
        19 13 7 18 17
        15 3 5 1 11
        10 7 1 19 12
        2 15 12 10 3
        11 20 16 12 9
        10 15 20 11 7
        1 9 20 7 6
    ]
    C = permutedims(reshape(C, (n, p, n)), [2, 1, 3])
    x = [MOI.add_variable(model) for i in 1:n, j in 1:n]
    MOI.add_constraint.(model, x, MOI.ZeroOne())
    for i in 1:n
        MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, x[i, j]) for j in 1:n],
                0.0,
            ),
            MOI.EqualTo(1.0),
        )
    end
    for j in 1:n
        MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, x[i, j]) for i in 1:n],
                0.0,
            ),
            MOI.EqualTo(1.0),
        )
    end
    f = MOI.VectorAffineFunction(
        [
            MOI.VectorAffineTerm(k, MOI.ScalarAffineTerm(-C[k, i, j], x[i, j])) for k in 1:p for i in 1:n for j in 1:n
        ],
        fill(0.0, p),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    results = [
        [0 1 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 1] => [16, 61, 47],
        [0 0 1 0 0 0 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1] => [17, 43, 71],
        [0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1] => [18, 47, 67],
        [0 0 1 0 0 1 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 1] => [20, 52, 54],
        [0 0 1 0 0 0 0 0 1 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 1] => [22, 37, 63],
        [0 0 1 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 1] => [22, 54, 47],
        [0 0 1 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 1 0] => [23, 43, 44],
        [0 1 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 1 0 0] => [24, 39, 45],
        [0 0 1 0 0 0 0 0 0 1 0 1 0 0 0 1 0 0 0 0 0 0 0 1 0] => [28, 33, 58],
        [0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0] => [28, 66, 39],
        [0 1 0 0 0 0 0 0 0 1 0 0 0 1 0 1 0 0 0 0 0 0 1 0 0] => [29, 29, 59],
        [0 1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1] => [34, 60, 42],
        [0 0 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 0 0 0 0 1 0 0] => [35, 38, 56],
        [0 1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 1 0] => [35, 49, 39],
        [0 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1] => [37, 55, 36],
        [1 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 1 0 0 1 0 0] => [38, 33, 53],
        [0 0 1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 0] => [39, 43, 41],
        [0 1 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 0] => [40, 47, 37],
        [0 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 0 0 0 0 0 1 0] => [43, 51, 31],
        [0 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1 0 0 1 0 0] => [45, 33, 34],
        [0 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 0 0 0 1 0 0] => [50, 40, 32],
    ]
    @test MOI.get(model, MOI.ResultCount()) == length(results)
    @test MOI.get(model, MOA.SubproblemCount()) >= length(results)
    for (i, (x_sol, y_sol)) in enumerate(results)
        x_primal = MOI.get(model, MOI.VariablePrimal(i), vec(x))
        @test ≈(vec(x_sol), x_primal; atol = 1e-6)
        @test ≈(-y_sol, MOI.get(model, MOI.ObjectiveValue(i)); atol = 1e-6)
    end
    return
end

function test_problem_issue_105(model)
    cost = [100.0, 120.0, 150.0, 110.0, 200.0, 170.0]
    time = [8.0, 3.0, 4.0, 2.0, 5.0, 4.0]
    capacity = [10.0, 8.0]
    demand = [5.0, 8.0, 5.0]
    m, n = 2, 3
    x = MOI.add_variables(model, m * n)
    MOI.add_constraint.(model, x, MOI.GreaterThan(0.0))
    MOI.add_constraint.(model, x, MOI.Integer())
    X = reshape(x, m, n)
    for i in 1:m
        f_i = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, X[i, :]), 0.0)
        MOI.add_constraint(model, f_i, MOI.LessThan(capacity[i]))
    end
    for j in 1:n
        f_j = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, X[:, j]), 0.0)
        MOI.add_constraint(model, f_j, MOI.EqualTo(demand[j]))
    end
    f = MOI.Utilities.operate(
        vcat,
        Float64,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(cost, x), 0.0),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(time, x), 0.0),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, x), 0.0),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.ResultCount()) == 6
    @test MOI.get(model, MOA.SubproblemCount()) >= 6
    for (i, y) in enumerate([
        [2380.0, 81.0, 18.0],
        [2440.0, 78.0, 18.0],
        [2500.0, 75.0, 18.0],
        [2560.0, 72.0, 18.0],
        [2620.0, 69.0, 18.0],
        [2680.0, 66.0, 18.0],
    ])
        @test ≈(y, MOI.get(model, MOI.ObjectiveValue(i)); atol = 1e-6)
    end
    return
end

function test_issue_122(model)
    m, n = 3, 10
    p1 = [5.0 1 10 8 3 5 3 3 7 2; 10 6 1 6 8 3 2 10 6 1; 2 3 1 6 9 7 1 5 4 8]
    p2 = [4.0 6 4 3 1 6 8 2 9 7; 8 8 8 2 4 8 8 1 10 1; 8 7 8 5 9 2 2 7 10 10]
    p3 = [4.0 3 6 4 7 5 9 5 8 4; 8 6 2 2 6 8 5 2 2 3; 2 8 10 3 5 7 5 9 5 5]
    w = [5.0 9 3 5 10 5 7 10 7 8; 4 8 8 6 10 8 10 7 5 1; 10 7 5 8 8 2 8 1 10 3]
    b = [34.0, 33.0, 31.0]
    x_ = MOI.add_variables(model, m * n)
    x = reshape(x_, m, n)
    MOI.add_constraint.(model, x, MOI.ZeroOne())
    f = MOI.Utilities.operate(
        vcat,
        Float64,
        sum(p1 .* x),
        sum(p2 .* x),
        sum(p3 .* x),
    )
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
    @test MOI.get(model, MOI.ResultCount()) == 42
    @test MOI.get(model, MOA.SubproblemCount()) >= 42
    return
end

function test_issue_133(model)
    #!format: off
    p = Float64[
        33 90 96 75 1 69 100 50 63 61 59 95 58 10 77 30 86 89 82 51 38 33 73 54 91 89 95 82 48 67
        55 36 80 58 20 96 75 57 24 68 37 58 8 85 27 25 71 53 47 72 57 64 1 8 12 68 3 80 20 90
        22 40 50 73 44 65 12 26 13 77 14 68 71 35 54 98 45 95 98 19 18 38 14 51 37 48 35 97 95 36
    ]
    w = Float64[
        22, 13, 10, 25, 4, 15, 17, 15, 15, 28, 14, 13, 2, 23, 6, 22, 18, 6, 23,
        21, 7, 7, 14, 4, 3, 27, 10, 5, 9, 10
    ]
    #!format: on
    x = MOI.add_variables(model, length(w))
    MOI.add_constraint.(model, x, MOI.ZeroOne())
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    f = MOI.Utilities.vectorize(p * x)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.add_constraint(model, w' * x, MOI.LessThan(204.0))
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(model, MOI.ResultCount()) == 95
    @test MOI.get(model, MOA.SubproblemCount()) >= 95
    return
end

end  # module Problems
