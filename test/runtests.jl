using Test
using Distributed
using LinearAlgebra
using SymmetricTensors
using Cumulants
using CumulantsFeatures
using Combinatorics
using Distributed
using Random
using CumulantsUpdates
import CumulantsFeatures: reduceband, greedestep, unfoldsym, hosvdstep, greedesearchdata, mev, mormbased, hosvdapprox
import CumulantsFeatures: updatemoments

include("test_select.jl")
include("test_detectors.jl")
include("symten2mattest.jl")

addprocs(3)
@everywhere using CumulantsFeatures
@everywhere using SymmetricTensors
@everywhere using Cumulants
@everywhere import SymmetricTensors: getblock
include("paralleltests.jl")
