#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

function _solve_mock(mock)
    highs = HiGHS.Optimizer()
    MOI.set(highs, MOI.Silent(), true)
    index_map = MOI.copy_to(highs, mock)
    MOI.optimize!(highs)
    x = [index_map[xi] for xi in MOI.get(mock, MOI.ListOfVariableIndices())]
    MOI.Utilities.mock_optimize!(
        mock,
        MOI.get(highs, MOI.TerminationStatus()),
        MOI.get(highs, MOI.VariablePrimal(), x),
    )
    obj = MOI.get(highs, MOI.ObjectiveValue())
    MOI.set(mock, MOI.ObjectiveValue(), obj)
    return
end

function mock_optimizer(fail_after::Int)
    return () -> begin
        model = MOI.Utilities.MockOptimizer(MOI.Utilities.Model{Float64}())
        MOI.Utilities.set_mock_optimize!(
            model,
            ntuple(i -> _solve_mock, fail_after)...,
            mock -> MOI.Utilities.mock_optimize!(mock, MOI.NUMERICAL_ERROR),
        )
        return model
    end
end
