#  Copyright 2019, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v.2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

is_test_file(f) = startswith(f, "test_") && endswith(f, ".jl")

testsuite = Dict{String,Expr}()
for (root, dirs, files) in walkdir(@__DIR__)
    for file in joinpath.(root, filter(is_test_file, files))
        testsuite[file] = :(include($file))
    end
end

import MultiObjectiveAlgorithms
import ParallelTestRunner

ParallelTestRunner.runtests(MultiObjectiveAlgorithms, ARGS; testsuite)
