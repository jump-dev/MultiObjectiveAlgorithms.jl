#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

import Pkg

if get(ENV, "CI", "false") == "true"
    Pkg.pkg"add MathOptInterface#master"
    Pkg.pkg"add JuMP#od/vector-optimization"
end

using Test

@testset "$file" for file in readdir(joinpath(@__DIR__, "algorithms"))
    include(joinpath(@__DIR__, "algorithms", file))
end
