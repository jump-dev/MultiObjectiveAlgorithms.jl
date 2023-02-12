#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

module TestUtilities

using Test

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

function test_filter_nondominated()
    x = Dict{MOI.VariableIndex,Float64}()
    solutions = [MOA.SolutionPoint(x, [0, 1]), MOA.SolutionPoint(x, [1, 0])]
    @test MOA.filter_nondominated(MOI.MIN_SENSE, solutions) == solutions
    @test MOA.filter_nondominated(MOI.MAX_SENSE, solutions) == solutions
    return
end

function test_filter_nondominated_sort_in_order()
    x = Dict{MOI.VariableIndex,Float64}()
    solutions = [MOA.SolutionPoint(x, [0, 1]), MOA.SolutionPoint(x, [1, 0])]
    r_solutions = reverse(solutions)
    @test MOA.filter_nondominated(MOI.MIN_SENSE, r_solutions) == solutions
    @test MOA.filter_nondominated(MOI.MAX_SENSE, r_solutions) == solutions
    return
end

function test_filter_nondominated_remove_duplicates()
    x = Dict{MOI.VariableIndex,Float64}()
    solutions = [MOA.SolutionPoint(x, [0, 1]), MOA.SolutionPoint(x, [1, 0])]
    trial = solutions[[1, 1]]
    @test MOA.filter_nondominated(MOI.MIN_SENSE, trial) == [solutions[1]]
    @test MOA.filter_nondominated(MOI.MAX_SENSE, trial) == [solutions[1]]
    return
end

function test_filter_nondominated_weakly_dominated()
    x = Dict{MOI.VariableIndex,Float64}()
    solutions = [
        MOA.SolutionPoint(x, [0, 1]),
        MOA.SolutionPoint(x, [0.5, 1]),
        MOA.SolutionPoint(x, [1, 0]),
    ]
    @test MOA.filter_nondominated(MOI.MIN_SENSE, solutions) == solutions[[1, 3]]
    @test MOA.filter_nondominated(MOI.MAX_SENSE, solutions) == solutions[[2, 3]]
    solutions = [
        MOA.SolutionPoint(x, [0, 1]),
        MOA.SolutionPoint(x, [0.5, 1]),
        MOA.SolutionPoint(x, [0.75, 1]),
        MOA.SolutionPoint(x, [0.8, 0.5]),
        MOA.SolutionPoint(x, [0.9, 0.5]),
        MOA.SolutionPoint(x, [1, 0]),
    ]
    @test MOA.filter_nondominated(MOI.MIN_SENSE, solutions) ==
          solutions[[1, 4, 6]]
    @test MOA.filter_nondominated(MOI.MAX_SENSE, solutions) ==
          solutions[[3, 5, 6]]
    return
end

function test_filter_nondominated_knapsack()
    x = Dict{MOI.VariableIndex,Float64}()
    solutions = [
        MOA.SolutionPoint(x, [0, 1, 1]),
        MOA.SolutionPoint(x, [0, 1, 1]),
        MOA.SolutionPoint(x, [1, 0, 1]),
        MOA.SolutionPoint(x, [1, 1, 0]),
        MOA.SolutionPoint(x, [1, 1, 0]),
    ]
    result = solutions[[1, 3, 4]]
    @test MOA.filter_nondominated(MOI.MIN_SENSE, solutions) == result
    @test MOA.filter_nondominated(MOI.MAX_SENSE, solutions) == result
    return
end

function test_filter_nondominated_triple()
    x = Dict{MOI.VariableIndex,Float64}()
    for p in MOA.Combinatorics.permutations(1:3)
        solutions = [
            MOA.SolutionPoint(x, [0, 1, 1][p]),
            MOA.SolutionPoint(x, [0, 2, 0][p]),
            MOA.SolutionPoint(x, [1, 1, 1][p]),
        ]
        # The permutation can change the ordering of the solutions that are
        # returnned, so we can't use `@test min_sol == solutions[1:2]`
        min_sol = MOA.filter_nondominated(MOI.MIN_SENSE, solutions)
        @test solutions[1] in min_sol && solutions[2] in min_sol
        @test length(min_sol) == 2
        max_sol = MOA.filter_nondominated(MOI.MAX_SENSE, solutions)
        @test solutions[2] in max_sol && solutions[3] in max_sol
        @test length(max_sol) == 2
    end
    return
end

end

TestUtilities.run_tests()
