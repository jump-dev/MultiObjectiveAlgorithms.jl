<img src="moa.png" alt="An image of the Moa bird. Licensed into the Public Domain by https://freesvg.org/moa" width="100px"/>

# MultiObjectiveAlgorithms.jl

[![Build Status](https://github.com/jump-dev/MultiObjectiveAlgorithms.jl/workflows/CI/badge.svg?branch=master)](https://github.com/jump-dev/MultiObjectiveAlgorithms.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/jump-dev/MultiObjectiveAlgorithms.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jump-dev/MultiObjectiveAlgorithms.jl)

MultiObjectiveAlgorithms.jl (MOA) is a collection of algorithms for
multi-objective optimization.

## Installation

Install MOA using `Pkg.add`:

```julia
import Pkg; Pkg.add("MultiObjectiveAlgorithms")
```

## Usage with JuMP

**Compat: this requires at least JuMP v1.8.0**

Use `MultiObjectiveAlgorithms` with JuMP as follows:

```julia
using JuMP
import HiGHS
import MultiObjectiveAlgorithms as MOA
model = JuMP.Model(() -> MOA.Optimizer(HiGHS.Optimizer))
set_optimizer_attribute(model, MOA.Algorithm(), MOA.NISE())
set_optimizer_attribute(model, MOA.SolutionLimit(), 4)
```

Replace `HiGHS.Optimizer` with an optimizer capable of solving a
single-objective instance of your optimization problem.

You may set additional optimizer attributes, the supported attributes depend on
the choice of solution algorithm.

## Algorithm

Set the algorithm using the `MOA.Algorithm()` attribute.

The value must be one of the algorithms supported by MOA:

 * `MOA.EpsilonConstraint()`
 * `MOA.Hierarchical()`
 * `MOA.KirlikSayin()`
 * `MOA.Lexicographic()` [default]
 * `MOA.NISE()`

Consult their docstrings for details.

## Other optimizer attributes

There are a number of optimizer attributes supported by the algorithms in MOA.

Each algorithm supports only a subset of the attributes. Consult the algorithm's
docstring for details on which attributes it supports, and how it uses them in
the solution process.

 * `MOA.EpsilonConstraintStep()`
 * `MOA.ObjectiveAbsoluteTolerance(index::Int)`
 * `MOA.ObjectivePriority(index::Int)`
 * `MOA.ObjectiveRelativeTolerance(index::Int)`
 * `MOA.ObjectiveWeight(index::Int)`
 * `MOA.SolutionLimit()`
