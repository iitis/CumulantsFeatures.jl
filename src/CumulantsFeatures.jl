module CumulantsFeatures
  using Distributions
  using SymmetricTensors
  using Cumulants
  using Distributed
  using StatsBase
  using LinearAlgebra
  using SharedArrays
  using CumulantsUpdates
  #using DistributedArrays
  import SymmetricTensors: issymetric, getblock, pyramidindices

  include("optimizationalgorithms.jl")
  include("detectors.jl")
  include("symten2mat.jl")

  export cumfsel, rxdetect, hosvdc4detect, cum2mat
end
