#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

using Test

@testset "$file" for file in readdir(joinpath(@__DIR__, "algorithms"))
    include(joinpath(@__DIR__, "algorithms", file))
end

@testset "$file" for file in readdir(@__DIR__)
    if startswith(file, "test_") && endswith(file, ".jl")
        include(joinpath(@__DIR__, file))
    end
end
