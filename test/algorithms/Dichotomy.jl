#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

module TestDicho

using Test

import HiGHS
import MultiObjectiveAlgorithms as MOA

const MOI = MOA.MOI

function run_tests()
    if Sys.WORD_SIZE == 32
        return  # Skip on 32-bit because HiGHS fails
    end
    for name in names(@__MODULE__; all = true)
        if startswith("$name", "test_")
            @testset "$name" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

function test_bicriteria_transportation()
    m = 3
    n = 4
    c = Float64[
        1 2 7 7
        1 9 3 4
        8 9 4 6
    ]
    d = Float64[
        4 4 3 4
        5 8 9 10
        6 2 5 1
    ]
    a = Float64[11, 3, 14, 16]
    b = Float64[8, 19, 17]
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOA.Algorithm(), MOA.Dichotomy())
    MOI.set(model, MOI.Silent(), true)
    x = [MOI.add_variable(model) for i in 1:m, j in 1:n]
    MOI.add_constraint.(model, x, MOI.GreaterThan(0.0))
    for j in 1:n
        MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, x[i, j]) for i in 1:m],
                0.0,
            ),
            MOI.EqualTo(a[j]),
        )
    end
    for i in 1:m
        MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(1.0, x[i, j]) for j in 1:n],
                0.0,
            ),
            MOI.EqualTo(b[i]),
        )
    end
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.VectorAffineFunction{Float64}}(),
        MOI.Utilities.vectorize(
            [
                MOI.ScalarAffineFunction(
                    [
                        MOI.ScalarAffineTerm(c[i, j], x[i, j]) for i in 1:m for
                        j in 1:n
                    ],
                    0.0,
                )
                MOI.ScalarAffineFunction(
                    [
                        MOI.ScalarAffineTerm(d[i, j], x[i, j]) for i in 1:m for
                        j in 1:n
                    ],
                    0.0,
                )
            ],
        ),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(model)
    N = MOI.get(model, MOI.ResultCount())
    y_sol = hcat([MOI.get(model, MOI.ObjectiveValue(i)) for i in 1:N]...)
    Y_N = Float64[
        143 265
        208 167
        156 200
        176 175
        186 171
    ]
    @test isapprox(y_sol, Y_N'; atol = 1e-6)
end

end

TestDicho.run_tests()
