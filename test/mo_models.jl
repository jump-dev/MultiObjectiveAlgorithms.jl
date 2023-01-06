#  Copyright 2019, Oscar Dowson. This Source Code Form is subject to the terms
#  of the Mozilla Public License, v.2.0. If a copy of the MPL was not
#  distributed with this file, You can obtain one at
#  http://mozilla.org/MPL/2.0/.

module TestModels

using Test
using JuMP

import MathOptInterface

const MOI = MathOptInterface

function run_tests(f)
    for name in names(@__MODULE__; all = true)
        if startswith("$name", "example_")
            @testset "$name" begin
                getfield(@__MODULE__, name)(f)
            end
        end
    end
    return
end

function example_moi_bolp_1(f)
    model = MOI.instantiate(f)
    MOI.set(model, MOI.Silent(), true)
    MOI.Utilities.loadfromstring!(
        model,
        """
variables: x, y
minobjective: [2 * x + y, x + 3 * y]
c1: x + y >= 1.0
c2: 0.5 * x + y >= 0.75
c3: x >= 0.0
c4: y >= 0.25
""",
    )
    x = MOI.get(model, MOI.VariableIndex, "x")
    y = MOI.get(model, MOI.VariableIndex, "y")
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(model, MOI.ResultCount()) == 3
    X = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.25]]
    Y = [[1.0, 3.0], [1.5, 2.0], [2.25, 1.75]]
    for i in 1:3
        @test MOI.get(model, MOI.PrimalStatus(i)) == MOI.FEASIBLE_POINT
        @test MOI.get(model, MOI.DualStatus(i)) == MOI.NO_SOLUTION
        @test MOI.get(model, MOI.ObjectiveValue(i)) == Y[i]
        @test MOI.get(model, MOI.VariablePrimal(i), x) == X[i][1]
        @test MOI.get(model, MOI.VariablePrimal(i), y) == X[i][2]
    end
    @test MOI.get(model, MOI.ObjectiveBound()) == [1.0, 1.75]
    return
end

function example_jump_bolp_1(f)
    model = Model(f; add_bridges = false)
    set_silent(model)
    @variable(model, x >= 0.0)
    @variable(model, y >= 0.25)
    @constraint(model, x + y >= 1.0)
    @constraint(model, 0.5 * x + y >= 0.75)
    @objective(model, Min, [2 * x + y, x + 3 * y])
    optimize!(model)
    print(solution_summary(model))
    @test termination_status(model) == OPTIMAL
    @test result_count(model) == 3
    X = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.25]]
    Y = [[1.0, 3.0], [1.5, 2.0], [2.25, 1.75]]
    for i in 1:3
        @test primal_status(model; result = i) == MOI.FEASIBLE_POINT
        @test dual_status(model; result = i) == MOI.NO_SOLUTION
        @test objective_value(model; result = i) == Y[i]
        @test value(x; result = i) == X[i][1]
        @test value(y; result = i) == X[i][2]
    end
    @test objective_bound(model) == [1.0, 1.75]
    return
end

function example_jump_bolp_1_maximize(f)
    model = Model(f; add_bridges = false)
    set_silent(model)
    @variable(model, x >= 0.0)
    @variable(model, y >= 0.25)
    @constraint(model, x + y >= 1.0)
    @constraint(model, 0.5 * x + y >= 0.75)
    @objective(model, Max, -[2 * x + y, x + 3 * y])
    optimize!(model)
    @test termination_status(model) == OPTIMAL
    @test result_count(model) == 3
    X = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.25]]
    Y = [-[1.0, 3.0], -[1.5, 2.0], -[2.25, 1.75]]
    for i in 1:3
        @test primal_status(model; result = i) == MOI.FEASIBLE_POINT
        @test dual_status(model; result = i) == MOI.NO_SOLUTION
        @test objective_value(model; result = i) == Y[i]
        @test value(x; result = i) == X[i][1]
        @test value(y; result = i) == X[i][2]
    end
    @test objective_bound(model) == -[1.0, 1.75]
    return
end

function example_jump_bolp_1_min_min(f)
    model = Model(f; add_bridges = false)
    set_silent(model)
    @variable(model, x >= 0.0)
    @variable(model, y >= 0.25)
    @constraint(model, x + y >= 1.0)
    @constraint(model, 0.5 * x + y >= 0.75)
    @expression(model, obj1, 2 * x + y)
    @expression(model, obj2, x + 3 * y)
    @objective(model, Min, [obj1, obj2])
    optimize!(model)
    @test termination_status(model) == OPTIMAL
    @test result_count(model) == 3
    X = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.25]]
    Y = [[1.0, 3.0], [1.5, 2.0], [2.25, 1.75]]
    for i in 1:3
        @test primal_status(model; result = i) == MOI.FEASIBLE_POINT
        @test dual_status(model; result = i) == MOI.NO_SOLUTION
        @test objective_value(model; result = i) == Y[i]
        @test value(x; result = i) == X[i][1]
        @test value(y; result = i) == X[i][2]
    end
    @test objective_bound(model) == [1.0, 1.75]
    return
end

function example_jump_bolp_1_min_min_reversed(f)
    model = Model(f; add_bridges = false)
    set_silent(model)
    @variable(model, x >= 0.0)
    @variable(model, y >= 0.25)
    @constraint(model, x + y >= 1.0)
    @constraint(model, 0.5 * x + y >= 0.75)
    @expression(model, obj1, 2 * x + y)
    @expression(model, obj2, x + 3 * y)
    # Here the objectives are reversed!
    @objective(model, Min, [obj2, obj1])
    optimize!(model)
    @test termination_status(model) == OPTIMAL
    @test result_count(model) == 3
    # Solution vectors are reversed as well
    X = reverse([[0.0, 1.0], [0.5, 0.5], [1.0, 0.25]])
    Y = reverse([[1.0, 3.0], [1.5, 2.0], [2.25, 1.75]])
    for i in 1:3
        @test primal_status(model; result = i) == MOI.FEASIBLE_POINT
        @test dual_status(model; result = i) == MOI.NO_SOLUTION
        @test objective_value(model; result = i) == reverse(Y[i])
        @test value(x; result = i) == X[i][1]
        @test value(y; result = i) == X[i][2]
    end
    @test objective_bound(model) == reverse([1.0, 1.75])
    return
end

function example_jump_bolp_1_min_max(f)
    model = Model(f; add_bridges = false)
    set_silent(model)
    @variable(model, x >= 0.0)
    @variable(model, y >= 0.25)
    @constraint(model, x + y >= 1.0)
    @constraint(model, 0.5 * x + y >= 0.75)
    @expression(model, obj1, 2 * x + y)
    @expression(model, obj2, -x - 3 * y)
    # We want
    #   min obj1
    #   max obj2
    # But you must pick a single sense.
    @objective(model, Min, [obj1, -obj2])
    optimize!(model)
    @test termination_status(model) == OPTIMAL
    @test result_count(model) == 3
    X = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.25]]
    Y = [[1.0, 3.0], [1.5, 2.0], [2.25, 1.75]]
    for i in 1:3
        @test primal_status(model; result = i) == MOI.FEASIBLE_POINT
        @test dual_status(model; result = i) == MOI.NO_SOLUTION
        @test objective_value(model; result = i) == Y[i]
        @test value(x; result = i) == X[i][1]
        @test value(y; result = i) == X[i][2]
    end
    @test objective_bound(model) == [1.0, 1.75]
    return
end


end  # module
