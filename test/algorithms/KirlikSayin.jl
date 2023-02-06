module TestKirlikSayin

using Test
using JuMP

import HiGHS
import MOO

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

function test_knapsack_min_p3()
  p = 3
  n = 10
  W = 2137.
  C = Float64[
    566 611 506 180 817 184 585 423 26 317
    62 84 977 979 874 54 269 93 881 563
    664 982 962 140 224 215 12 869 332 537
  ]
  w = Float64[557, 898, 148, 63, 78, 964, 246, 662, 386, 272]
  model = Model(() -> MOO.Optimizer(HiGHS.Optimizer))
  set_optimizer_attribute(model, MOO.Algorithm(), MOO.Lexicographic())
  set_silent(model)
  @variable(model, x[1:n], Bin)
  @objective(model, Min, -C * x)
  @constraint(model, sum(x) ≤ W)
  optimize!(model)
  X_E = Float64[
    1 0 1 1 1 0 1 1 0 1
    0 1 1 1 1 0 1 0 1 1
    1 1 1 1 1 0 0 0 0 1
    0 1 1 1 1 0 0 1 0 1
    1 1 1 1 1 0 0 0 1 0
    1 0 1 1 1 0 0 1 1 0
    0 0 1 1 1 0 1 1 1 1
  ]
  Y_N = Float64[
    -3394 -3817 -3408
    -3042 -4627 -3189
    -2997 -3539 -3509
    -2854 -3570 -3714
    -2706 -3857 -3304
    -2518 -3866 -3191
    -2854 -4636 -3076
  ]
  @test isapprox(
    X_E, 
    vcat([transpose(value.(x)) for i = 1:result_count(model)]...); atol = 1e-6
  )
  @test isapprox(
    Y_N, 
    vcat([transpose(objective_value(model, result=i)) for i = 1:result_count(model)]...); atol = 1e-6
  )
  return  
end

function test_knapsack_max_p3()
  p = 3
  n = 10
  W = 2137.
  C = Float64[
    566 611 506 180 817 184 585 423 26 317
    62 84 977 979 874 54 269 93 881 563
    664 982 962 140 224 215 12 869 332 537
  ]
  w = Float64[557, 898, 148, 63, 78, 964, 246, 662, 386, 272]
  model = Model(() -> MOO.Optimizer(HiGHS.Optimizer))
  set_optimizer_attribute(model, MOO.Algorithm(), MOO.Lexicographic())
  set_silent(model)
  @variable(model, x[1:n], Bin)
  @objective(model, Max, C * x)
  @constraint(model, sum(x) ≤ W)
  optimize!(model)
  X_E = Float64[
    1 0 1 1 1 0 1 1 0 1
    0 1 1 1 1 0 1 0 1 1
    1 1 1 1 1 0 0 0 0 1
    0 1 1 1 1 0 0 1 0 1
    1 1 1 1 1 0 0 0 1 0
    1 0 1 1 1 0 0 1 1 0
    0 0 1 1 1 0 1 1 1 1
  ]
  Y_N = Float64[
    3394 3817 3408
    3042 4627 3189
    2997 3539 3509
    2854 3570 3714
    2706 3857 3304
    2518 3866 3191
    2854 4636 3076
  ]
  @test isapprox(
    X_E, 
    vcat([transpose(value.(x)) for i = 1:result_count(model)]...); atol = 1e-6
  )
  @test isapprox(
    Y_N, 
    vcat([transpose(objective_value(model, result=i)) for i = 1:result_count(model)]...); atol = 1e-6
  )
  return
end

TestKirlikSayin.run_tests()

end