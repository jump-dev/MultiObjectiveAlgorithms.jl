#  Copyright 2019, Oscar Dowson. This Source Code Form is subject to the terms
#  of the Mozilla Public License, v.2.0. If a copy of the MPL was not
#  distributed with this file, You can obtain one at
#  http://mozilla.org/MPL/2.0/.

import Pkg
Pkg.pkg"add MathOptInterface#od/vector-optimization"
Pkg.pkg"add JuMP#od/vector-optimization"

using MOO
using Test

import HiGHS

const MOI = MOO.MOI

include("mo_models.jl")

@testset "$name" for (name, alg) in Dict("NISE" => MOO.NISE())
    TestModels.run_tests() do
        return MOI.OptimizerWithAttributes(
            MOO.Optimizer(HiGHS.Optimizer),
            "algorithm" => alg,
        )
    end
end

@testset "test_NISE_basics" begin
    model = MOO.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOI.RawOptimizerAttribute("algorithm"), MOO.NISE())
    @test MOI.get(model, MOI.RawOptimizerAttribute("solution_limit")) ==
          typemax(Int)
    MOI.set(model, MOI.RawOptimizerAttribute("solution_limit"), 1)
    @test MOI.get(model, MOI.RawOptimizerAttribute("solution_limit")) == 1
    @test_throws(
        MOI.UnsupportedAttribute,
        MOI.get(model, MOI.RawOptimizerAttribute("bad_options"))
    )
    @test_throws(
        MOI.UnsupportedAttribute,
        MOI.set(model, MOI.RawOptimizerAttribute("bad_options"), 1)
    )
end
