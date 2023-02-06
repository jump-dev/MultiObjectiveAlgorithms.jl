#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

"""
    KirlikSayin()

`KirlikSayin` implements the Kirlik & Sayın (2014) algorithm.

## Supported optimizer attributes

"""

mutable struct KirlikSayin <: AbstractAlgorithm
  #yI::Union{Nothing, Vector{Float64}}
  #yN::Union{Nothing, Vector{Float64}}
  #L::Vector{Rectangle}

  #KirlikSayin() = new(nothing, nothing, Rectangle[])
  KirlikSayin() = new()
end

#=
MOI.supports(::KirlikSayin, ::IdealPoint) = true

function MOI.get(alg::KirlikSayin, attr::IdealPoint)
  return alg.yI
end

function MOI.set(alg::KirlikSayin, attr::IdealPoint, value)
  alg.yI = value
  return
end

function MOI.set(alg::KirlikSayin, attr::IdealPoint, index, value)
  alg.yI[index] = value
  return
end

MOI.supports(::KirlikSayin, ::NadirPoint) = true

function MOI.get(alg::KirlikSayin, attr::NadirPoint)
  return alg.yN
end

function MOI.set(alg::KirlikSayin, attr::NadirPoint, value)
  alg.yN = value
  return
end

function MOI.set(alg::KirlikSayin, attr::NadirPoint, index, value)
  alg.yN[index] = value
  return
end
=#
struct Rectangle
  l::Vector{Float64}
  u::Vector{Float64}
  dim::Int
  function Rectangle(l::Vector{Float64}, u::Vector{Float64})
    @assert length(l) == length(u) "Dimension mismatch between l and u"
    return new(l, u, length(l))
  end
end

function ⊆(Rᵢ::Rectangle, Rⱼ::Rectangle)
	@assert Rᵢ.dim == Rⱼ.dim "Dimension mismatch"
	all(Rᵢ.l .≥ Rⱼ.l) && all(Rᵢ.u .≤ Rⱼ.u)
end

function remove_rect!(L::Vector{Rectangle}, R::Rectangle)
  ix_to_pop = []
  for (t, Rₜ) in enumerate(L)
    if Rₜ ⊆ R
      push!(ix_to_pop, t)
    end
  end
  # Lᵤ = L[ix_to_pop]
  deleteat!(L, ix_to_pop)
  # return Lᵤ
end

function split_rectangle(r::Rectangle, axis::Int, f::Float64)
  l = [i != axis ? r.l[i] : f for i = 1:r.dim]
  u = [i != axis ? r.u[i] : f for i = 1:r.dim]
  return Rectangle(r.l, u), Rectangle(l, r.u)
end

function update_list(L::Vector{Rectangle}, f::Vector{Float64})
  @info L
  @info f
  L̄, L = L, Vector{Rectangle}()
  for Rᵢ in L̄
    lᵢ, uᵢ = Rᵢ.l, Rᵢ.u
    T = [Rᵢ]
    for j = 1:length(f)
      if lᵢ[j] < f[j] < uᵢ[j]
        @info "anything?"
        T̄ = Vector{Rectangle}()
        for Rₜ in T
          push!(T̄, split_rectangle(Rₜ, j, f[j])...)
        end
        T = T̄
      end
    end
    push!(L, T...)
  end
  return L
end

#=
=#
function optimize_multiobjective!(algorithm::KirlikSayin, model::Optimizer)
  
  # Problem with p objectives.
  # Set k = 1, meaning the nondominated points will get projected 
  # down to the objective {2, 3, ..., p}

  k = 1

  XE, YN = Vector{Dict{MOI.VariableIndex,Float64}}(), Vector{Vector{Float64}}()

  solutions = ParetoSolution[]
  
  variables = MOI.get(model.inner, MOI.ListOfVariableIndices())
  sense = MOI.get(model.inner, MOI.ObjectiveSense())
  nbObj = MOI.output_dimension(model.f)
  yI, yN = zeros(nbObj), zeros(nbObj)
  δ = ifelse(sense == MOI.MIN_SENSE, -1, 1)

  # Ideal and Nadir point estimation
  for (i, f_i) in enumerate(MOI.Utilities.scalarize(model.f))
    MOI.set(model.inner, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), f_i)
    MOI.set(model.inner, MOI.ObjectiveSense(), sense)
    @info sense
    MOI.optimize!(model.inner)
    X = Dict{MOI.VariableIndex,Float64}(
      x => MOI.get(model.inner, MOI.VariablePrimal(), x) for
      x in variables
    )
    yI[i] = MOI.Utilities.eval_variables(x -> X[x], f_i)
    MOI.set(model.inner, MOI.ObjectiveSense(), ifelse(sense == MOI.MIN_SENSE, MOI.MAX_SENSE, MOI.MIN_SENSE))
    MOI.optimize!(model.inner)
    X = Dict{MOI.VariableIndex,Float64}(
      x => MOI.get(model.inner, MOI.VariablePrimal(), x) for
      x in variables
    )
    yN[i] = MOI.Utilities.eval_variables(x -> X[x], f_i)
  end

  proj(x::Vector{Float64}, axis::Int) = x[begin:end .!= axis]

  volume(r::Rectangle, l::Vector{Float64}) = prod(r.u - l)

  L = [Rectangle(proj(yI, k), proj(yN, k))]

  SetType = ifelse(
    sense == MOI.MIN_SENSE,
    MOI.LessThan{Float64},
    MOI.GreaterThan{Float64},
  )

  while !isempty(L)

    Rᵢ = L[argmax(volume(Rᵢ, proj(yI, k)) for Rᵢ in L)]
    lᵢ, uᵢ = Rᵢ.l, Rᵢ.u

    # Solving the first stage model: P_k(ε)
    # Set ε := uᵢ
    ε = insert!(copy(uᵢ), k, 0.0)
    ε_constraints = Dict{Int, Any}()

    MOI.set(
      model.inner, 
      MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 
      getindex(MOI.Utilities.scalarize(model.f), k)
    )
    MOI.set(model.inner, MOI.ObjectiveSense(), sense)

    for (i, f_i) in enumerate(MOI.Utilities.scalarize(model.f))
      if i ≠ k
        ε_constraints[i] = MOI.add_constraint(model.inner, f_i, SetType(ε[i] + δ))
      end
    end

    MOI.optimize!(model.inner)
    if MOI.get(model.inner, MOI.TerminationStatus()) == MOI.OPTIMAL

      zₖ = MOI.get(model.inner, MOI.ObjectiveValue())

      # Solving the second stage model: Q_k(ε, zₖ)
      # Set objective sum(model.f)

      MOI.set(
        model.inner, 
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 
        sum(MOI.Utilities.scalarize(model.f))
      )
      MOI.set(model.inner, MOI.ObjectiveSense(), sense)
  
      # Constraint to eliminate weak dominance
      zₖ_constraint = MOI.add_constraint(model.inner, getindex(MOI.Utilities.scalarize(model.f), k), MOI.EqualTo(zₖ))
  
      MOI.optimize!(model.inner)

      for (i, ε_constraint) in ε_constraints
        MOI.delete(model, ε_constraint)
      end
      MOI.delete(model, zₖ_constraint)

      if MOI.get(model.inner, MOI.TerminationStatus()) == MOI.OPTIMAL
        X = Dict{MOI.VariableIndex,Float64}(
          x => MOI.get(model.inner, MOI.VariablePrimal(), x) for
          x in variables
        )
        f = MOI.Utilities.eval_variables(x -> X[x], model.f)
        f̄ = proj(f, k)
        if f ∈ YN # solution already exists!
        else # new solution!
          @info f
          push!(YN, f); push!(XE, X)
					L = update_list(L, f̄)
          @info "Length of L: $(length(L))"
        end
        remove_rect!(L, Rectangle(f̄, uᵢ))
      else # Q is infeasible
        remove_rect!(L, Rectangle(proj(yI, k), uᵢ))
      end
    else 
      remove_rect!(L, Rectangle(proj(yI, k), uᵢ))
    end
    
  end
  
  for (Xi, Yi) in zip(XE, YN)
    push!(solutions, ParetoSolution(Xi, Yi))
  end
  
  return MOI.OPTIMAL, solutions
end