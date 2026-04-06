#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

using JuMP
import DataFrames
import Gurobi
import HiGHS
import MathOptInterface
import MultiObjectiveAlgorithms as MOA
import Statistics

function solve_tamby_mokp(;
    p::Int,
    n::Int,
    instance::Int,
    optimizer::Any,
    algorithm::MOA.AbstractAlgorithm,
    silent::Bool = false,
)
    filename = joinpath(@__DIR__, "MOKP", "MOKP_p-$(p)_n-$(n)_$(instance).dat")
    lines = readlines(filename)
    @assert p == parse(Int, lines[1])
    @assert n == parse(Int, lines[2])
    b = parse(Float64, lines[3])
    C = reduce(vcat, [Float64.(x.args) for x in Meta.parse(lines[4]).args]')
    w = Meta.parse(lines[5]).args
    model = Model(() -> MOA.Optimizer(optimizer))
    if silent
        set_silent(model)
    end
    set_attribute(model, MOA.Algorithm(), algorithm)
    @variable(model, x[1:n], Bin)
    @constraint(model, w' * x <= b)
    @objective(model, Max, C * x)
    optimize!(model)
    return (;
        p,
        n,
        instance,
        termination_status = termination_status(model),
        result_count = result_count(model),
        solve_time = solve_time(model),
        solve_time_inner = get_attribute(model, MOA.SolveTimeSecInner()),
        subproblem_count = get_attribute(model, MOA.SubproblemCount()),
    )
end

ALGORITHMS = Dict(
    "TambyVanderpooten" => MOA.TambyVanderpooten(),
    "Dichotomy" => MOA.Dichotomy(),
    "EpsilonConstraint" => MOA.EpsilonConstraint(),
)

OPTIMIZERS = Dict("HiGHS" => HiGHS.Optimizer, "Gurobi" => Gurobi.Optimizer)

TESTSETS = Dict(
    (2, 200) => (
        ["TambyVanderpooten", "Dichotomy", "EpsilonConstraint"],
        ["HiGHS", "Gurobi"],
    ),
    (3, 100) => (["TambyVanderpooten"], ["Gurobi"]),
    (4, 50) => (["TambyVanderpooten"], ["Gurobi"]),
)

results_filename = "results.log"
results = Any[]
for ((p, n), (algorithms, solvers)) in TESTSETS, instance in 1:10
    for alg in algorithms, solver in solvers
        algorithm, optimizer = ALGORITHMS[alg], OPTIMIZERS[solver]
        println("Running: p-$(p)_n-$(n)_$(instance) :: $alg :: $solver")
        ret = solve_tamby_mokp(; p, n, instance, optimizer, algorithm)
        push!(results, (; solver, alg, ret...))
        open(results_filename, "a") do io
            return println(io, results[end])
        end
    end
end

l = eval.(Meta.parse.(readlines(results_filename)))
df = DataFrames.DataFrame(l)
df.percent_time = 100 .* df.solve_time_inner ./ df.solve_time
ret = DataFrames.combine(
    DataFrames.groupby(df, [:p, :n, :alg, :solver]),
    :result_count => Statistics.mean => :result_count,
    :solve_time => Statistics.mean => :solve_time,
    :subproblem_count => Statistics.mean => :subproblem_count,
    :percent_time => Statistics.mean => :percent_time,
)
ret.solve_time .= max.(1, round.(Int, ret.solve_time))
ret.percent_time .= max.(1, round.(Int, ret.percent_time))
sort!(ret, [:p, :n, :alg, :solver])
open("final_results.out", "w") do io
    for row in eachrow(ret)
        print(io, row.p, " & ", row.n, " & \\texttt{", row.alg, "} & ")
        print(io, row.solver, " & ", row.result_count, " & ")
        print(io, row.subproblem_count, " & ", row.solve_time, " & ")
        println(io, row.percent_time, " \\\\")
    end
    return
end
