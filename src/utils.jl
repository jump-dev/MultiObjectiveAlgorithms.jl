#  Copyright 2019, Oscar Dowson. This Source Code Form is subject to the terms
#  of the Mozilla Public License, v.2.0. If a copy of the MPL was not
#  distributed with this file, You can obtain one at
#  http://mozilla.org/MPL/2.0/.

struct ParetoSolution
    x::Dict{MOI.VariableIndex, Float64}
    y::Vector{Float64}
end

function Base.isapprox(a::ParetoSolution, b::ParetoSolution; kwargs...)
    return isapprox(a.y, b.y; kwargs...)
end

function _scalarise(f::MOI.VectorOfVariables, w::Vector{Float64})
    @assert MOI.output_dimension(f) == length(w)
    return MOI.ScalarAffineFunction(
        [MOI.ScalarAffineTerm(w[i], f.variables[i]) for i = 1:length(w)], 0.0
    )
end

function _scalarise(f::MOI.VectorAffineFunction, w::Vector{Float64})
    @assert MOI.output_dimension(f) == length(w)
    constant = sum(w[i] * f.constants[i] for i = 1:length(w))
    terms = MOI.ScalarAffineTerm{Float64}[
        MOI.ScalarAffineTerm(
            w[term.output_index] * term.scalar_term.coefficient,
            term.scalar_term.variable_index
        ) for term in f.terms
    ]
    return MOI.ScalarAffineFunction(terms, constant)
end
