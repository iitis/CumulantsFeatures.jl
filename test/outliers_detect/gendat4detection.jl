#!/usr/bin/env julia

using Distributed
using Random
using LinearAlgebra
procs_id = addprocs(4)
using DatagenCopulaBased
@everywhere using Distributions
@everywhere using Cumulants
@everywhere using SymmetricTensors
using CumulantsFeatures
using JLD2
using FileIO
using ArgParse
@everywhere import CumulantsFeatures: reduceband
@everywhere using DatagenCopulaBased
@everywhere using CumulantsFeatures
@everywhere using HypothesisTests

Random.seed!(1234)


@everywhere function gmarg2t(X::Matrix{T}, nu::Int) where T <: AbstractFloat
  # transform univariate gasussian marginals into t-Student with nu degrees of freedom
  Y = copy(X)
  for i = 1:size(X, 2)
    x = X[:,i]
    s = var(x)
    mu = mean(x)
    d = Normal(mu, s)
    u = cdf.(d, x)
    pvalue(ExactOneSampleKSTest(u,Uniform(0,1)))>0.0001 || throw(AssertionError("$i marg. not unif."))
    Y[:,i] = quantile.(TDist(nu), u)
  end
  return Y
end

function main(args)
  s = ArgParseSettings("description")
  @add_arg_table s begin
    "--nu", "-n"
    default = 6
    help = "the number of degrees of freedom for the t-Student copula"
    arg_type = Int

    "--nuu", "-u"
    default = 6
    help = "the number of degrees of freedom for the t-Student marginal"
    arg_type = Int
  end
  parsed_args = parse_args(s)
  ν = parsed_args["nu"]
  νu = parsed_args["nuu"]

  println("copula's degreen of freedom = ", ν)
  println("matginal's degree of freedom = ", νu)
  # parameters, data sie
  @everywhere t = 1000
  # no of marginals
  @everywhere n = 50
  # no of marginals with t-Student copula
  @everywhere malf_size = 10
  # outliers places on the beginning of data
  @everywhere a = 100
  data_dir = "."
  # number of generated data sets
  test_number = 3
  filename = "tstudent_$(ν)_marg$(νu)-t_size-$(n)_malfsize-$malf_size-t_$(t)_$a.jld2"

  data = Dict{String, Any}("variables_no" => n,
                         "sample_number" => t,
                         "ν" => ν,
                         "test_number" => test_number,
                         "malf_size" => malf_size,
                         "a" => a,
                         "data" => Dict{String, Dict{String,Any}}())


  known_data_size = 0
  if isfile("$data_dir/$filename")
   data["data"] = load("$data_dir/$filename")["data"]
   known_data_size += length(data["data"])
   println("Already have $known_data_size samples \n Will generate $(test_number-known_data_size) more")
  end

  #true calculations
  println("Calculation started")
  for m=(known_data_size+1):test_number
    @time begin
      println(" > $m ($ν)")
      malf = randperm(n)[1:malf_size]
      Σ = cormatgen_rand(n)
      samples_orig = rand(MvNormal(Σ), t)'

      # gcop2tstudent(x[1:a, :], malf, ν) - outliers, given gaussian multivariate copula is changed for Gaussian

      # x[a+1:end, :] - ordinary data

      # gmarg2t( ..., νu) - univariate marginals are changed to t-Student

      # "malf" - dict key

      versions = [(x->gmarg2t(vcat(gcop2tstudent(x[1:a, :], malf, ν), x[a+1:end, :]), νu), "malf")]

      cur_dict = Dict{String, Any}("malf" => malf,
                                   "cor_source" => Σ)

      data_dict = @distributed (merge) for (sampler, label)=versions
        println(label)
        samples = sampler(samples_orig)
        Σ_malf = cov(samples)
        Dict("cor_$label" => Σ_malf,
             "x_$label" => samples)
      end

      data["data"]["$m"] = merge(cur_dict, data_dict)
      save("$data_dir/$filename", data)
    end
  end

end

main(ARGS)
