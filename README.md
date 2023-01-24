# MOO: the Multi-Objective Optimizer

MOO is a collection of algorithms for multi-objective optimization.

## Installation

This package is currently under development. You can install it as follows:

```julia
] add MathOptInterface#od/vector-optimization
] add JuMP#od/vector-optimization
] add https://github.com/odow/MOO.jl
```

## Usage with JuMP

Use `MOO` with JuMP as follows:

```julia
using JuMP
import HiGHS, MOO
model = JuMP.Model(() -> MOO.Optimizer(HiGHS.Optimizer))
set_optimizer_attribute(model, MOO.Algorithm(), MOO.NISE())
set_optimizer_attribute(model, MOO.SolutionLimit(), 4)
```

Replace `HiGHS.Optimizer` with an optimizer capable of solving a
single-objective instance of your optimization problem.

You must set the `MOO.Algorithm` attribute to choose the solution algorithm.

You may set additional optimizer attributes, the supported attributes depend on
the choice of solution algorithm.

## Algorithm

There are a number of algorithms supported by the algorithms in MOO.

 * `MOO.NISE()`
 * `MOO.Hierarchical()`

Consult their docstrings for details.

## Other optimizer attributes

There are a number of optimizer attributes supported by the algorithms in MOO.

Each algorithm supports only a subset of the attributes. Consult the algorithm's
docstring for details on which attributes it supports, and how it uses them in
the solution process.

 * `MOO.SolutionLimit()`
 * `MOO.ObjectivePriority(index::Int)`
 * `MOO.ObjectiveWeight(index::Int)`
 * `MOO.ObjectiveRelativeTolerance(index::Int)`
