#  Copyright 2019, Oscar Dowson. This Source Code Form is subject to the terms
#  of the Mozilla Public License, v.2.0. If a copy of the MPL was not
#  distributed with this file, You can obtain one at
#  http://mozilla.org/MPL/2.0/.

@testset "NISE" begin
    model = MOO.NISE(GLPK.Optimizer())
    x = MOI.add_variables(model, 2)
    MOI.add_constraint(model, MOI.SingleVariable(x[1]), MOI.GreaterThan(0.0))
    MOI.add_constraint(model, MOI.SingleVariable(x[2]), MOI.GreaterThan(0.25))
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 1.0], x), 0.0),
        MOI.GreaterThan(1.0)
    )
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([0.5, 1.0], x), 0.0),
        MOI.GreaterThan(0.75)
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.VectorAffineFunction{Float64}}(),
        MOI.VectorAffineFunction(
            MOI.VectorAffineTerm{Float64}[
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(2.0, x[1])),
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[2])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[1])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(3.0, x[2]))
            ],
            [0.0, 0.0]
        )
    )
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(model, MOI.ResultCount()) == 3
    X = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.25]]
    Y = [[1.0, 3.0], [1.5, 2.0], [2.25, 1.75]]
    for i = 1:3
        @test MOI.get(model, MOI.PrimalStatus(i)) == MOI.FEASIBLE_POINT
        @test MOI.get(model, MOI.DualStatus(i)) == MOI.NO_SOLUTION
        @test MOI.get(model, MOI.ObjectiveValue(i)) == Y[i]
        @test MOI.get.(model, MOI.VariablePrimal(i), x) == X[i]
    end
    @test MOI.get(model, MOI.ObjectiveBound()) == [1.0, 1.75]
end
