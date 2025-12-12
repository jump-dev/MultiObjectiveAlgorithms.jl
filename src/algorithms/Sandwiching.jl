#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

"""
    Sandwiching(precision::Float64)

An algorithm that implemennts the paper described in Koenen, M., Balvert, M., &
Fleuren, H. A. (2023). A Renewed Take on Weighted Sum in Sandwich Algorithms:
Modification of the Criterion Space. (Center Discussion Paper; Vol. 2023-012).
CentER, Center for Economic Research.

## Supported problem classes

This algorithm supports all problem classes.

## Compat

To use this algorithm you MUST first load the Polyhedra.jl Julia package:

```julia
import MultiObjectiveAlgorithms as MOA
import Polyhedra
algorithm = MOA.Sandwiching(0.0)
```
"""
mutable struct Sandwiching <: AbstractAlgorithm
    precision::Float64
end
