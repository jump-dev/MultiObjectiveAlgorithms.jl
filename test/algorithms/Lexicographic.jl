#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

module TestLexicographic

using Test
using JuMP

import HiGHS
import MOO

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

function test_knapsack()
    P = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
    model = Model(() -> MOO.Optimizer(HiGHS.Optimizer))
    set_optimizer_attribute(model, MOO.Algorithm(), MOO.Lexicographic())
    set_optimizer_attribute(model, MOO.ObjectiveRelativeTolerance(1), 0.1)
    set_silent(model)
    @variable(model, 0 <= x[1:4] <= 1)
    @objective(model, Max, P * x)
    @constraint(model, sum(x) <= 2)
    optimize!(model)
    @test ≈(value.(x), [0.9, 1, 0, 0.1]; atol = 1e-3)
    return
end

function test_knapsack_min()
    P = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
    model = Model(() -> MOO.Optimizer(HiGHS.Optimizer))
    set_optimizer_attribute(model, MOO.Algorithm(), MOO.Lexicographic())
    set_optimizer_attribute(model, MOO.ObjectiveRelativeTolerance(1), 0.1)
    set_silent(model)
    @variable(model, 0 <= x[1:4] <= 1)
    @objective(model, Min, -P * x)
    @constraint(model, sum(x) <= 2)
    optimize!(model)
    @test ≈(value.(x), [0.9, 1, 0, 0.1]; atol = 1e-3)
    return
end

function test_knapsack_default()
    P = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
    model = Model(() -> MOO.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, 0 <= x[1:4] <= 1)
    @objective(model, Max, P * x)
    @constraint(model, sum(x) <= 2)
    optimize!(model)
    @test raw_status(model) == "Solve complete. Found 1 solution(s)"
    @test ≈(value.(x), [1, 1, 0, 0]; atol = 1e-3)
    return
end

end

TestLexicographic.run_tests()
