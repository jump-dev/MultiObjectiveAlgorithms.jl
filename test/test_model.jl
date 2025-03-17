#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

module TestModel

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

function _mock_optimizer()
    return MOI.Utilities.MockOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
    )
end

function test_moi_runtests()
    MOI.Test.runtests(
        MOA.Optimizer(_mock_optimizer),
        MOI.Test.Config(; exclude = Any[MOI.optimize!]);
        exclude = String[
            # Skipped beause of UniversalFallback in _mock_optimizer
            "test_attribute_Silent",
            "test_attribute_after_empty",
            "test_model_copy_to_UnsupportedAttribute",
            "test_model_copy_to_UnsupportedConstraint",
            "test_model_supports_constraint_ScalarAffineFunction_EqualTo",
            "test_model_supports_constraint_VariableIndex_EqualTo",
            "test_model_supports_constraint_VectorOfVariables_Nonnegatives",
        ],
    )
    return
end

function test_infeasible()
    model = MOA.Optimizer(HiGHS.Optimizer)
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
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, 2)
    MOI.add_constraint.(model, x, MOI.GreaterThan(0.0))
    f = MOI.Utilities.operate(vcat, Float64, 1.0 .* x...)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.DUAL_INFEASIBLE
    return
end

function test_time_limit()
    model = MOA.Optimizer(HiGHS.Optimizer)
    @test MOI.supports(model, MOI.TimeLimitSec())
    @test MOI.get(model, MOI.TimeLimitSec()) === nothing
    MOI.set(model, MOI.TimeLimitSec(), 2)
    @test MOI.get(model, MOI.TimeLimitSec()) === 2.0
    MOI.set(model, MOI.TimeLimitSec(), nothing)
    @test MOI.get(model, MOI.TimeLimitSec()) === nothing
    return
end

function test_solve_time()
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, 2)
    MOI.add_constraint.(model, x, MOI.GreaterThan(0.0))
    f = MOI.Utilities.operate(vcat, Float64, 1.0 .* x...)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    @test isnan(MOI.get(model, MOI.SolveTimeSec()))
    MOI.optimize!(model)
    @test MOI.get(model, MOI.SolveTimeSec()) >= 0
    return
end

function test_unnsupported_attributes()
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, 2)
    c = MOI.add_constraint.(model, x, MOI.GreaterThan(0.0))
    f = MOI.Utilities.operate(vcat, Float64, 1.0 .* x...)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.optimize!(model)
    @test_throws(
        MOI.GetAttributeNotAllowed{MOI.RelativeGap},
        MOI.get(model, MOI.RelativeGap()),
    )
    @test_throws(
        MOI.GetAttributeNotAllowed{MOI.DualObjectiveValue},
        MOI.get(model, MOI.DualObjectiveValue()),
    )
    @test_throws(
        MOI.GetAttributeNotAllowed{MOI.ConstraintDual},
        MOI.get(model, MOI.ConstraintDual(), c),
    )
    return
end

function test_invalid_model()
    model = MOA.Optimizer(HiGHS.Optimizer)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.INVALID_MODEL
    return
end

function test_raw_optimizer_attribuute()
    model = MOA.Optimizer(HiGHS.Optimizer)
    attr = MOI.RawOptimizerAttribute("presolve")
    @test MOI.supports(model, attr)
    @test MOI.get(model, attr) == "choose"
    MOI.set(model, attr, "off")
    @test MOI.get(model, attr) == "off"
    return
end

function test_algorithm()
    model = MOA.Optimizer(HiGHS.Optimizer)
    @test MOI.supports(model, MOA.Algorithm())
    @test MOI.get(model, MOA.Algorithm()) == nothing
    MOI.set(model, MOA.Algorithm(), MOA.Chalmet())
    @test MOI.get(model, MOA.Algorithm()) == MOA.Chalmet()
    return
end

function test_copy_to()
    src = MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}())
    MOI.set(src, MOA.Algorithm(), MOA.Chalmet())
    x = MOI.add_variables(src, 2)
    MOI.add_constraint.(src, x, MOI.GreaterThan(0.0))
    f = MOI.Utilities.operate(vcat, Float64, 1.0 .* x...)
    MOI.set(src, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.set(src, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    dest = MOA.Optimizer(HiGHS.Optimizer)
    index_map = MOI.copy_to(dest, src)
    MOI.set(dest, MOI.Silent(), true)
    MOI.optimize!(dest)
    @test MOI.get(dest, MOI.NumberOfVariables()) == 2
    return
end

function test_scalarise()
    x = MOI.VariableIndex.(1:2)
    f = MOI.VectorOfVariables(x)
    g = MOA._scalarise(f, [0.2, 0.8])
    @test isapprox(g, 0.2 * x[1] + 0.8 * x[2])
    return
end

function test_ideal_point()
    for (flag, result) in (true => [0.0, -9.0], false => [NaN, NaN])
        model = MOA.Optimizer(HiGHS.Optimizer)
        @test MOI.supports(model, MOA.ComputeIdealPoint())
        @test MOI.get(model, MOA.ComputeIdealPoint())
        @test MOI.set(model, MOA.ComputeIdealPoint(), flag) === nothing
        @test MOI.get(model, MOA.ComputeIdealPoint()) == flag
        # Test that MOI.empty! does not override ComputeIdealPoint
        MOI.empty!(model)
        @test MOI.get(model, MOA.ComputeIdealPoint()) == flag
        MOI.set(model, MOI.Silent(), true)
        x = MOI.add_variables(model, 2)
        MOI.add_constraint.(model, x, MOI.GreaterThan(0.0))
        MOI.add_constraint(model, x[2], MOI.LessThan(3.0))
        MOI.add_constraint(model, 3.0 * x[1] - 1.0 * x[2], MOI.LessThan(6.0))
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        f = MOI.Utilities.vectorize([
            3.0 * x[1] + x[2],
            -1.0 * x[1] - 2.0 * x[2],
        ])
        MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
        MOI.optimize!(model)
        point = MOI.get(model, MOI.ObjectiveBound())
        @test length(point) == 2
        if flag
            @test point ≈ result
        else
            @test all(isnan, point)
        end
    end
    return
end

end  # module

TestModel.run_tests()
