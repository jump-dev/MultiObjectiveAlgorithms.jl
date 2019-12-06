#  Copyright 2019, Oscar Dowson. This Source Code Form is subject to the terms
#  of the Mozilla Public License, v.2.0. If a copy of the MPL was not
#  distributed with this file, You can obtain one at
#  http://mozilla.org/MPL/2.0/.

@testset "NISE" begin
    model = MOO.NISE(GLPK.Optimizer())
    bolp_1(model)
    @test MOI.get(model, MOI.RawParameter("solution_limit")) == typemax(Int)
    MOI.set(model, MOI.RawParameter("solution_limit"), 1)
    @test MOI.get(model, MOI.RawParameter("solution_limit")) == 1
    @test_throws(
        MOI.UnsupportedAttribute,
        MOI.get(model, MOI.RawParameter("bad_options"))
    )
    @test_throws(
        MOI.UnsupportedAttribute,
        MOI.set(model, MOI.RawParameter("bad_options"), 1)
    )
    @test_throws(
        ErrorException(
            "Expected type $(Int) when setting parameter solution_limit. " *
            "Got Symbol."
        ),
        MOI.set(model, MOI.RawParameter("solution_limit"), :a)
    )
end
