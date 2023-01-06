**This package was an experimental attempt at multi-objective optimization in JuMP. It no longer works.**

# MOO: the Multi-Objective Optimizer

MOO is a collection of algorithms for multi-objective optimization.

It currently implements:

- The Non-inferior Set Estimation algorithm of Cohon et al. (1979).
    ```julia
    using MOO
    import HiGHS
    model = MOO.NISE(HiGHS.Optimizer())
    ```

## Installation

This package is currently under development. It needs development branches of at
least one upstream package. You can install it as follows:

```julia
] add https://github.com/odow/MOO.jl
```

## Use with JuMP

You cannot use MOO with JuMP.
