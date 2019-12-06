# MOO: the Multi-Objective Optimizer

MOO is a collection of algorithms for multi-objective optimization.

It currently implements:

- The Non-inferior Set Estimation algorithm of Cohon et al. (1979).
    ```julia
    using GLPK
    using MOO
    model = MOO.NISE(GLPK.Optimizer())
    ```

## Installation

This package is currently under development. It needs development branches of at
least one upstream package. You can install it as follows:

```julia
] add https://github.com/odow/MOO.jl
```

## Use with JuMP

You cannot use MOO with JuMP.
