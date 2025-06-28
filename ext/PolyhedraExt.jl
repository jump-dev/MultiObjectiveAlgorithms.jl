module PolyhedraExt

using MultiObjectiveAlgorithms
using Polyhedra

function MultiObjectiveAlgorithms._halfspaces(IPS::Vector{Vector{Float64}})
    V = vrep(IPS)
    H = halfspaces(doubledescription(V))
    return [(-H_i.a, -H_i.β) for H_i in H]
end

end