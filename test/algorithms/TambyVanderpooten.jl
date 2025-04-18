#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

module TestTambyVanderpooten

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

function test_knapsack_min_p3()
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
        ones(p),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    X_E = Float64[
        1 0 1 1 1 0 1 1 0 1
        0 1 1 1 1 0 1 0 1 1
        1 1 1 1 1 0 0 0 0 1
        0 1 1 1 1 0 0 1 0 1
        1 1 1 1 1 0 0 0 1 0
        1 0 1 1 1 0 0 1 1 0
        0 0 1 1 1 0 1 1 1 1
    ]
    Y_N = Float64[
        -3394 -3817 -3408
        -3042 -4627 -3189
        -2997 -3539 -3509
        -2854 -3570 -3714
        -2706 -3857 -3304
        -2518 -3866 -3191
        -2854 -4636 -3076
    ]
    Y_N .+= 1
    N = MOI.get(model, MOI.ResultCount())
    x_sol = hcat([MOI.get(model, MOI.VariablePrimal(i), x) for i in 1:N]...)'
    y_sol = hcat([MOI.get(model, MOI.ObjectiveValue(i)) for i in 1:N]...)'
    y_sol, x_sol = y_sol[sortperm(collect(eachrow(y_sol))), :],
    x_sol[sortperm(collect(eachrow(y_sol))), :]
    Y_N, X_E = Y_N[sortperm(collect(eachrow(Y_N))), :],
    X_E[sortperm(collect(eachrow(Y_N))), :]
    @test isapprox(x_sol, X_E; atol = 1e-6)
    @test isapprox(y_sol, Y_N; atol = 1e-6)
    @test MOI.get(model, MOI.ObjectiveBound()) ≈ vec(minimum(Y_N; dims = 1))
    return
end

function test_knapsack_max_p3()
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
            i in 1:p for j in 1:n
        ],
        fill(0.0, p),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    X_E = Float64[
        1 0 1 1 1 0 1 1 0 1
        0 1 1 1 1 0 1 0 1 1
        1 1 1 1 1 0 0 0 0 1
        0 1 1 1 1 0 0 1 0 1
        1 1 1 1 1 0 0 0 1 0
        1 0 1 1 1 0 0 1 1 0
        0 0 1 1 1 0 1 1 1 1
    ]
    Y_N = Float64[
        3394 3817 3408
        3042 4627 3189
        2997 3539 3509
        2854 3570 3714
        2706 3857 3304
        2518 3866 3191
        2854 4636 3076
    ]
    N = MOI.get(model, MOI.ResultCount())
    x_sol = hcat([MOI.get(model, MOI.VariablePrimal(i), x) for i in 1:N]...)'
    y_sol = hcat([MOI.get(model, MOI.ObjectiveValue(i)) for i in 1:N]...)'
    y_sol, x_sol = y_sol[sortperm(collect(eachrow(y_sol))), :],
    x_sol[sortperm(collect(eachrow(y_sol))), :]
    Y_N, X_E = Y_N[sortperm(collect(eachrow(Y_N))), :],
    X_E[sortperm(collect(eachrow(Y_N))), :]
    @test isapprox(x_sol, X_E; atol = 1e-6)
    @test isapprox(y_sol, Y_N; atol = 1e-6)
    @test MOI.get(model, MOI.ObjectiveBound()) ≈ vec(maximum(Y_N; dims = 1))
    return
end

function test_knapsack_min_p4()
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
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.TambyVanderpooten())
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
    X_E = Float64[
        1 1 1 0 1 1 1 0 0 0
        0 1 1 0 1 1 1 1 1 0
        0 1 1 1 1 1 1 0 0 0
        0 1 1 0 1 0 1 0 1 1
        0 1 1 1 1 0 1 0 1 0
        0 0 1 0 1 1 1 0 1 1
        0 1 1 1 0 1 1 0 1 0
        0 0 1 1 0 1 1 1 1 0
        0 0 1 1 1 1 1 0 1 0
        0 1 0 0 0 1 1 1 1 1
    ]
    Y_N = Float64[
        -3269 -2320 -3059 -2891
        -3152 -3232 -3596 -3382
        -2883 -3237 -2535 -2397
        -2862 -3648 -3049 -2028
        -2725 -4064 -2652 -1819
        -2435 -3618 -2282 -2094
        -2092 -3244 -2643 -2705
        -1904 -3253 -2530 -2469
        -2298 -4034 -1885 -1885
        -2146 -1944 -2947 -3428
    ]
    N = MOI.get(model, MOI.ResultCount())
    x_sol = hcat([MOI.get(model, MOI.VariablePrimal(i), x) for i in 1:N]...)'
    y_sol = hcat([MOI.get(model, MOI.ObjectiveValue(i)) for i in 1:N]...)'
    y_sol, x_sol = y_sol[sortperm(collect(eachrow(y_sol))), :],
    x_sol[sortperm(collect(eachrow(y_sol))), :]
    Y_N, X_E = Y_N[sortperm(collect(eachrow(Y_N))), :],
    X_E[sortperm(collect(eachrow(Y_N))), :]
    @test isapprox(x_sol, X_E; atol = 1e-6)
    @test isapprox(y_sol, Y_N; atol = 1e-6)
    @test MOI.get(model, MOI.ObjectiveBound()) ≈ vec(minimum(Y_N; dims = 1))
    return
end

function test_knapsack_max_p4()
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
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.TambyVanderpooten())
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
            i in 1:p for j in 1:n
        ],
        fill(0.0, p),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    X_E = Float64[
        1 1 1 0 1 1 1 0 0 0
        0 1 1 0 1 1 1 1 1 0
        0 1 1 1 1 1 1 0 0 0
        0 1 1 0 1 0 1 0 1 1
        0 1 1 1 1 0 1 0 1 0
        0 0 1 0 1 1 1 0 1 1
        0 1 1 1 0 1 1 0 1 0
        0 0 1 1 0 1 1 1 1 0
        0 0 1 1 1 1 1 0 1 0
        0 1 0 0 0 1 1 1 1 1
    ]
    Y_N = Float64[
        3269 2320 3059 2891
        3152 3232 3596 3382
        2883 3237 2535 2397
        2862 3648 3049 2028
        2725 4064 2652 1819
        2435 3618 2282 2094
        2092 3244 2643 2705
        1904 3253 2530 2469
        2298 4034 1885 1885
        2146 1944 2947 3428
    ]
    N = MOI.get(model, MOI.ResultCount())
    x_sol = hcat([MOI.get(model, MOI.VariablePrimal(i), x) for i in 1:N]...)'
    y_sol = hcat([MOI.get(model, MOI.ObjectiveValue(i)) for i in 1:N]...)'
    y_sol, x_sol = y_sol[sortperm(collect(eachrow(y_sol))), :],
    x_sol[sortperm(collect(eachrow(y_sol))), :]
    Y_N, X_E = Y_N[sortperm(collect(eachrow(Y_N))), :],
    X_E[sortperm(collect(eachrow(Y_N))), :]
    @test isapprox(x_sol, X_E; atol = 1e-6)
    @test isapprox(y_sol, Y_N; atol = 1e-6)
    @test MOI.get(model, MOI.ObjectiveBound()) ≈ vec(maximum(Y_N; dims = 1))
    return
end

function test_assignment_min_p3()
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
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.TambyVanderpooten())
    MOI.set(model, MOI.Silent(), true)
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
    X_E = Float64[
        0 1 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 1
        0 0 1 0 0 0 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1
        0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1
        0 0 1 0 0 1 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 1
        0 0 1 0 0 0 0 0 1 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 1
        0 0 1 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 1 0
        0 0 1 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 1
        0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0
        0 1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1
        0 1 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 1 0 0
        0 1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 1 0
        0 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1
        0 0 1 0 0 0 0 0 0 1 0 1 0 0 0 1 0 0 0 0 0 0 0 1 0
        0 0 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 0 0 0 0 1 0 0
        0 0 1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 0
        1 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 1 0 0 1 0 0
        0 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1 0 0 1 0 0
        0 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 0 0 0 0 0 1 0
        0 1 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 0
        0 1 0 0 0 0 0 0 0 1 0 0 0 1 0 1 0 0 0 0 0 0 1 0 0
        0 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 0 0 0 1 0 0
    ]
    Y_N = Float64[
        16 61 47
        17 43 71
        18 47 67
        20 52 54
        22 37 63
        23 43 44
        22 54 47
        28 66 39
        34 60 42
        24 39 45
        35 49 39
        37 55 36
        28 33 58
        35 38 56
        39 43 41
        38 33 53
        45 33 34
        43 51 31
        40 47 37
        29 29 59
        50 40 32
    ]
    N = MOI.get(model, MOI.ResultCount())
    x_sol =
        hcat([MOI.get(model, MOI.VariablePrimal(i), vec(x)) for i in 1:N]...)'
    y_sol = hcat([MOI.get(model, MOI.ObjectiveValue(i)) for i in 1:N]...)'
    y_sol, x_sol = y_sol[sortperm(collect(eachrow(y_sol))), :],
    x_sol[sortperm(collect(eachrow(y_sol))), :]
    Y_N, X_E = Y_N[sortperm(collect(eachrow(Y_N))), :],
    X_E[sortperm(collect(eachrow(Y_N))), :]
    @test isapprox(x_sol, X_E; atol = 1e-6)
    @test isapprox(y_sol, Y_N; atol = 1e-6)
    return
end

function test_assignment_max_p3()
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
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.TambyVanderpooten())
    MOI.set(model, MOI.Silent(), true)
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
    X_E = Float64[
        0 1 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 1
        0 0 1 0 0 0 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1
        0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1
        0 0 1 0 0 1 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 1
        0 0 1 0 0 0 0 0 1 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 1
        0 0 1 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 1 0
        0 0 1 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 1
        0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0
        0 1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1
        0 1 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 1 0 0
        0 1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 1 0
        0 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1
        0 0 1 0 0 0 0 0 0 1 0 1 0 0 0 1 0 0 0 0 0 0 0 1 0
        0 0 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 0 0 0 0 1 0 0
        0 0 1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 0
        1 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 1 0 0 1 0 0
        0 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1 0 0 1 0 0
        0 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 0 0 0 0 0 1 0
        0 1 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 0
        0 1 0 0 0 0 0 0 0 1 0 0 0 1 0 1 0 0 0 0 0 0 1 0 0
        0 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 0 0 0 1 0 0
    ]
    Y_N = Float64[
        -16 -61 -47
        -17 -43 -71
        -18 -47 -67
        -20 -52 -54
        -22 -37 -63
        -23 -43 -44
        -22 -54 -47
        -28 -66 -39
        -34 -60 -42
        -24 -39 -45
        -35 -49 -39
        -37 -55 -36
        -28 -33 -58
        -35 -38 -56
        -39 -43 -41
        -38 -33 -53
        -45 -33 -34
        -43 -51 -31
        -40 -47 -37
        -29 -29 -59
        -50 -40 -32
    ]
    N = MOI.get(model, MOI.ResultCount())
    x_sol =
        hcat([MOI.get(model, MOI.VariablePrimal(i), vec(x)) for i in 1:N]...)'
    y_sol = hcat([MOI.get(model, MOI.ObjectiveValue(i)) for i in 1:N]...)'
    y_sol, x_sol = y_sol[sortperm(collect(eachrow(y_sol))), :],
    x_sol[sortperm(collect(eachrow(y_sol))), :]
    Y_N, X_E = Y_N[sortperm(collect(eachrow(Y_N))), :],
    X_E[sortperm(collect(eachrow(Y_N))), :]
    @test isapprox(x_sol, X_E; atol = 1e-6)
    @test isapprox(y_sol, Y_N; atol = 1e-6)
    return
end

function test_infeasible()
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.TambyVanderpooten())
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
    MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    return
end

end

TestTambyVanderpooten.run_tests()
