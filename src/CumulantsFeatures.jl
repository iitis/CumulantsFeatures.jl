module CumulantsFeatures
  using Distributions
  using SymmetricTensors
  using Cumulants
  using StatsBase
  import SymmetricTensors: issymetric

  include("optimizationalgorithms.jl")
  include("detectors.jl")

  export cumfsel, rxdetect, hosvdc4detect
end
