#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

module MultiObjectiveAlgorithmsPolyhedraExt

import MathOptInterface as MOI
import MultiObjectiveAlgorithms as MOA
import Polyhedra

include("Polyhedra/GeneralDichotomy.jl")
include("Polyhedra/Sandwiching.jl")

end  # module MultiObjectiveAlgorithmsPolyhedraExt
