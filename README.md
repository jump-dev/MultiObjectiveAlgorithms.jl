<img src="https://raw.githubusercontent.com/jump-dev/MultiObjectiveAlgorithms.jl/master/moa.png" alt="An image of the Moa bird. Licensed into the Public Domain by https://freesvg.org/moa" width="100px"/>

# MultiObjectiveAlgorithms.jl

[![Build Status](https://github.com/jump-dev/MultiObjectiveAlgorithms.jl/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/jump-dev/MultiObjectiveAlgorithms.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/jump-dev/MultiObjectiveAlgorithms.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jump-dev/MultiObjectiveAlgorithms.jl)

[MultiObjectiveAlgorithms.jl](https://github.com/jump-dev/MultiObjectiveAlgorithms.jl)
(MOA) is a collection of algorithms for multi-objective optimization.

## License

`MultiObjectiveAlgorithms.jl` is licensed under the [MPL 2.0 License](https://github.com/jump-dev/MultiObjectiveAlgorithms.jl/blob/master/LICENSE.md).

## Getting help

If you need help, please ask a question on the [JuMP community forum](https://jump.dev/forum).

If you have a reproducible example of a bug, please [open a GitHub issue](https://github.com/jump-dev/MultiObjectiveAlgorithms.jl/issues/new).

## Installation

Install MOA using `Pkg.add`:

```julia
import Pkg
Pkg.add("MultiObjectiveAlgorithms")
```

## Use with JuMP

Use `MultiObjectiveAlgorithms` with JuMP as follows:

```julia
using JuMP
import HiGHS
import MultiObjectiveAlgorithms as MOA
model = JuMP.Model(() -> MOA.Optimizer(HiGHS.Optimizer))
set_attribute(model, MOA.Algorithm(), MOA.Dichotomy())
set_attribute(model, MOI.SolutionLimit(), 4)
```

For worked examples, see the [Simple multi-objective examples](https://jump.dev/JuMP.jl/stable/tutorials/linear/multi_objective_examples/)
tutorial in the JuMP documentation.

Replace `HiGHS.Optimizer` with an optimizer capable of solving a
single-objective instance of your optimization problem.

You may set additional optimizer attributes, the supported attributes depend on
the choice of solution algorithm.

## Algorithm

Set the algorithm using the `MOA.Algorithm()` attribute.

The value must be one of the algorithms supported by MOA:

 * `MOA.Chalmet()`
 * `MOA.Dichotomy()`
 * `MOA.DominguezRios()`
 * `MOA.EpsilonConstraint()`
 * `MOA.Hierarchical()`
 * `MOA.KirlikSayin()`
 * `MOA.Lexicographic()` [default]
 * `MOA.RandomWeighting()`
 * `MOA.Sandwiching()`
 * `MOA.TambyVanderpooten()`

Consult their docstrings for details.

## Other optimizer attributes

There are a number of optimizer attributes supported by the algorithms in MOA.

Each algorithm supports only a subset of the attributes. Consult the algorithm's
docstring for details on which attributes it supports, and how it uses them in
the solution process.

 * `MOA.EpsilonConstraintStep()`
 * `MOA.LexicographicAllPermutations()`
 * `MOA.ObjectiveAbsoluteTolerance(index::Int)`
 * `MOA.ObjectivePriority(index::Int)`
 * `MOA.ObjectiveRelativeTolerance(index::Int)`
 * `MOA.ObjectiveWeight(index::Int)`
 * `MOI.SolutionLimit()`
 * `MOI.TimeLimitSec()`

Query the number of scalar subproblems that were solved using

 * `MOA.SubproblemCount()`

## Ideal point

By default, MOA will compute the ideal point, which can be queried using the
`MOI.ObjectiveBound` attribute.

Computing the ideal point requires as many solves as the dimension of the
objective function. Thus, if you do not need the ideal point information, you
can improve the performance of MOA by setting the `MOA.ComputeIdealPoint()`
attribute to `false`.

## Citation

If you use this package for academic research, please cite the [following preprint](https://www.arxiv.org/abs/2507.05501):

```bibtex
@misc{dowson2025MOA.jl,
      title={MultiObjectiveAlgorithms.jl: a Julia package for solving multi-objective optimization problems},
      author={Oscar Dowson and Xavier Gandibleux and G{\"o}khan Kof},
      year={2025},
      eprint={2507.05501},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2507.05501}
}
```
