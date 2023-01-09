# MOO: the Multi-Objective Optimizer

MOO is a collection of algorithms for multi-objective optimization.

## Installation

This package is currently under development. It needs development branches of at
least one upstream package. You can install it as follows:

```julia
] add MathOptInterface#od/vector-optimization
] add JuMP#od/vector-optimization
] add https://github.com/odow/MOO.jl
```

## Algorithms

### The Non-inferior Set Estimation algorithm of Cohon et al. (1979).

```julia
using JuMP
import HiGHS, MOO
model = JuMP.Model(() -> MOO.Optimizer(HiGHS.Optimizer))
set_optimizer_attribute(model, "algorithm", MOO.NISE())
```
