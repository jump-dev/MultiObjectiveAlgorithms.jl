#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

module TestNISE

using Test

import HiGHS
import MOO

const MOI = MOO.MOI

include("../mo_models.jl")

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

function test_NISE()
    TestModels.run_tests(
        MOI.OptimizerWithAttributes(
            () -> MOO.Optimizer(HiGHS.Optimizer),
            MOO.Algorithm() => MOO.NISE(),
        ),
    )
    return
end

function test_NISE_SolutionLimit()
    model = MOO.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOO.Algorithm(), MOO.NISE())
    @test MOI.get(model, MOO.SolutionLimit()) == typemax(Int)
    MOI.set(model, MOO.SolutionLimit(), 1)
    @test MOI.get(model, MOO.SolutionLimit()) == 1
    return
end

end

TestNISE.run_tests()
