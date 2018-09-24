procs_id = addprocs(8)
using DatagenCopulaBased
@everywhere using Distributions
@everywhere using Cumulants
using SymmetricTensors
using CumFSel
using JLD2
using FileIO
@everywhere import CumFSel: reduceband
@everywhere using DatagenCopulaBased
@everywhere using CumFSel

ν = parse(ARGS[1])
println(ν)


@everywhere t = 100_000
@everywhere n = 50
@everywhere malf_size = 10
@everywhere a = 1_000
data_dir = "data_outliers"
test_number = 10
filename = "tstudent_$(ν)-t_size-$(n)_malfsize-$malf_size-t_$(t)_$a.jld2"

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


    versions = [(x->x, "original"),
                (x->vcat(gcop2tstudent(x[1:a, :], malf, ν), x[a+1:end, :]), "malf")]

    cur_dict = Dict{String, Any}("malf" => malf,
                                 "cor_source" => Σ)

    data_dict = @parallel (merge) for (sampler, label)=versions
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

rmprocs(procs_id)
exit()
