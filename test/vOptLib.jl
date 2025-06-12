#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

module vOptLib

using Test
import JSON
import MathOptInterface as MOI

function run_tests(model::MOI.ModelLike)
    for name in names(@__MODULE__; all = true)
        if startswith("$name", "test_")
            @testset "$name" begin
                MOI.empty!(model)
                getfield(@__MODULE__, name)(model)
            end
        end
    end
    return
end

function _test_vOptLib_instance(model, instance)
    root = joinpath(dirname(@__DIR__), "instances")
    src = MOI.FileFormats.MOF.Model()
    MOI.read_from_file(src, joinpath(root, "models", instance * ".mof.json"))
    MOI.copy_to(model, src)
    MOI.optimize!(model)
    x = MOI.get(model, MOI.ListOfVariableIndices())
    sol_list = JSON.parsefile(joinpath(root, "solutions", instance * ".json"))
    # solutions[Y] => [X...]
    solutions = Dict{Vector{Int},Vector{Vector{Int}}}()
    for sol in sol_list
        Y = convert(Vector{Int}, sol["Y"])
        if !haskey(solutions, Y)
            solutions[Y] = Vector{Int}[]
        end
        push!(solutions[Y], convert(Vector{Int}, sol["X"]))
    end
    @test MOI.get(model, MOI.ResultCount()) >= length(solutions)
    for i in 1:MOI.get(model, MOI.ResultCount())
        Y = round.(Int, MOI.get(model, MOI.ObjectiveValue(i)))
        @test haskey(solutions, Y)
        X = round.(Int, MOI.get(model, MOI.VariablePrimal(i), x))
        @test X in solutions[Y]
    end
    return
end

test_vOptLib_2KP50_11(model) = _test_vOptLib_instance(model, "2KP50-11")

test_vOptLib_2KP50_50(model) = _test_vOptLib_instance(model, "2KP50-50")

test_vOptLib_2KP50_92(model) = _test_vOptLib_instance(model, "2KP50-92")

test_vOptLib_2KP100_50(model) = _test_vOptLib_instance(model, "2KP100-50")

end  # module vOptLib
