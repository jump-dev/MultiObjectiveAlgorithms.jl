#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

module TestModel

using Test

import HiGHS
import MultiObjectiveAlgorithms as MOA

const MOI = MOA.MOI

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
        warn_unsupported = true,
        exclude = String[
            # Skipped beause of UniversalFallback in _mock_optimizer
            "test_attribute_Silent",
            "test_model_copy_to_UnsupportedAttribute",
            "test_model_copy_to_UnsupportedConstraint",
            "test_model_supports_constraint_ScalarAffineFunction_EqualTo",
            "test_model_supports_constraint_VariableIndex_EqualTo",
            "test_model_supports_constraint_VectorOfVariables_Nonnegatives",
        ],
    )
    return
end

end

TestModel.run_tests()
