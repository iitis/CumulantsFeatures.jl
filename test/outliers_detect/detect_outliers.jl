#!/usr/bin/env julia

using Distributed
using JLD2
using FileIO
using ArgParse
using Distributions
using CumulantsFeatures
using LinearAlgebra
using Cumulants
addprocs(3)
@everywhere using CumulantsFeatures
@everywhere using Cumulants
#using StatsBase
#using ROCAnalysis
using PyCall
@pyimport matplotlib as mpl
@pyimport matplotlib.colors as mc
mpl.rc("text", usetex=true)
mpl.use("Agg")
using PyPlot

"""
rand_withour_return(s::Int, b::Int)

  out of numbers 1,2,3 , ... s returns no_pools random iterms without replacements

"""
function rand_withour_return(s::Int, no_pools::Int)
  data = collect(1:s)
  ret = []
  for i in 1:no_pools
    temp = rand(data)
    data = filter!(i -> i != temp, data)
    ret = vcat(ret, temp)
  end
  ret
end

"""
random_choice(data::Dict, β::Int)

Given data Dict and the parameter no_pools
returns a vector of random choice detections: [TruePositive, FalsePosotiveRate]

FalsePosotiveRate is the probability of type 1 error.

"""

function random_choice(data::Dict, no_pools::Int)
  no_cases = length(data["data"])
  ret = zeros(no_cases, 2)
  println("random_choice detector")
  for c=1:no_cases
    println("case n. = ", c)
    s = size(data["data"]["$c"]["x_malf"],1)
    detected = rand_withour_return(s, no_pools)
    no_positive = data["a"]
    # it is assumed by the random data generation
    #that outliers are at the beginning of data set
    theshold = no_positive
    no_true_detected = count(detected .<= theshold)
    no_false_detercted = count(detected .> theshold)

    sample_size = size(data["data"]["$c"]["x_malf"], 1)
    no_negative = sample_size-no_positive
    truePositiveRate = no_true_detected/no_positive
    #type 1 error
    flasePositiveRate = no_false_detercted/no_negative
    ret[c,:] = [truePositiveRate, flasePositiveRate]
  end
  ret
end


"""
detection_hosvd(data::Dict, β::Float64, r::Int = 3)

Given data Dict and the parameter β (threshold) and r (number of direction data
are projected) returns a vector of hosvd detection:

[TruePositive, FalsePosotiveRate]

"""

function detection_hosvd(data::Dict, β::Float64, r::Int = 3)
  no_cases = length(data["data"])
  ret = zeros(no_cases, 2)
  println("hosvd detector")
  for c=1:no_cases
    println("case no = ", c)
    detected = hosvdc4detect(data["data"]["$c"]["x_malf"], β, r)

    no_positive = data["a"]
    # it is assumed by the random data generation
    #that outliers are at the beginning of data set
    theshold = no_positive

    no_true_detected = count(findall(detected) .<= theshold)
    no_false_detected = count(findall(detected) .> theshold)

    sample_size = size(data["data"]["$c"]["x_malf"], 1)
    no_negative = sample_size-no_positive
    truePositiveRate = no_true_detected/no_positive
    #type 1 error
    flasePositiveRate = no_false_detected/no_negative
    ret[c,:] = [truePositiveRate, flasePositiveRate]
  end
  ret
end

"""
  detection_rx(data::Dict, α::Float64 = 0.99)

  Given data Dict and the parameter α ("probability" threshold)
   returns a vector of RX detection:

  [TruePositive, FalsePosotiveRate]

"""
function detection_rx(data::Dict, α::Float64 = 0.99)

  no_cases = length(data["data"])
  ret = zeros(no_cases, 2)
  print("rx detector")
  for c=1:no_cases
    println("case no = ", c)
    detected = rxdetect(data["data"]["$c"]["x_malf"], α)

    no_positive = data["a"]
    # it is assumed by the random data generation
    #that outliers are at the beginning of data set
    theshold = no_positive

    no_true_detected = count(findall(detected) .<= theshold)
    no_false_detected = count(findall(detected) .> theshold)

    sample_size = size(data["data"]["$c"]["x_malf"], 1)
    no_negative = sample_size-no_positive
    truePositiveRate = no_true_detected/no_positive
    #type 1 error
    flasePositiveRate = no_false_detected/no_negative
    ret[c,:] = [truePositiveRate, flasePositiveRate]
  end
  ret
end

"""
  plotdet(hosvd, rx, rand, nu::Int = 6)

plot results of detection
"""

function plotdet(hosvd, rx, rand, nu::Int = 6)
  mpl.rc("font", family="serif", size = 7)
  fig, ax = subplots(figsize = (2.5, 2.))
  # raking each data case
  for i in 1:size(hosvd[1],1)
    xh = [k[i,2] for k in hosvd]
    yh = [k[i,1] for k in hosvd]

    # excluding [0., 0.] case where the parameter is out of range
    j = (xh .> 0.) .* (yh .> 0.)
    xh = xh[j]
    yh = yh[j]

    xrx = [k[i,2] for k in rx]
    yrx = [k[i,1] for k in rx]

    xrand = [k[i,2] for k in rand]
    yrand = [k[i,1] for k in rand]

    plt[:plot](xh, yh, "o-", label = "hosvd", color = "blue")
    plt[:plot](xrx, yrx, "d-", label = "RX", color = "red")
    plt[:plot](xrand, yrand, "x-", label = "random", color = "gray")
    ax[:legend](fontsize = 6., loc = 2, ncol = 2)
    subplots_adjust(left = 0.15, bottom = 0.16)
    show()
    xlabel("False Positive Rate (type 1 error rate)", labelpad = -1.0)
    ylabel("True Positive Rate", labelpad = 0.)
    savefig("./pics/$(nu)_$(i)detect.pdf")
    PyPlot.clf()
  end
end


function main(args)
  s = ArgParseSettings("description")
  @add_arg_table s begin
    "file"
    help = "the file name"
    arg_type = String
  end
  parsed_args = parse_args(s)
  str = parsed_args["file"]
  data = load(str)
  # copulas parameter for outliers
  ν = data["ν"]

  if !isfile("./data/roc_rand"*str)
    no_pools = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50, 10]
    print("number random= ", size(no_pools))
    roc = [random_choice(data, k) for k in no_pools]
    save("./data/roc_rand"*str, "roc", roc, "ks", no_pools)
    print(roc)
  end

  if !isfile("./data/roc"*str)
    threshold = [8., 6., 5., 4., 3., 2.5, 2., 1.8, 1.6, 1.40, 1.30, 1.28]
    print("number hosvd= ", size(threshold))
    roc = [detection_hosvd(data, k) for k in  threshold]
    print(threshold)
    print(roc)
    save("./data/roc"*str, "roc", roc, "ks",  threshold)
  end

  if !isfile("./data/rocrx"*str)
    as = vcat([0.0005, 0.002, 0.01], collect(0.02:0.1:0.85), [0.9, 0.99, 0.999, 0.9999, 0.99999])
    print("number rx= ", size(as))
    rocrx = [detection_rx(data, k) for k in as]

    print(as)
    print(rocrx)
    save("./data/rocrx"*str, "roc", rocrx, "alpha", as)
  end

  r = load("./data/roc"*str)["roc"]
  rx = load("./data/rocrx"*str)["roc"]
  rr = load("./data/roc_rand"*str)["roc"]
  plotdet(r, rx, rr, ν)

end

main(ARGS)
