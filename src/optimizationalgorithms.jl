mev(Σ::Matrix{T}, c, ls::Vector{Bool}) where T <: AbstractFloat = det(reduceband(Σ, ls))

function mormbased(Σ::Matrix{T}, c::Array{T, N}, ls::Vector{Bool}) where {T <: AbstractFloat, N}
  vecnorm(reduceband(c, ls))/vecnorm(reduceband(Σ, ls))^(N/2)
end

"""
  detoverdetfitfunction(a::Array{N}, b::Array{N})

computes the maximizing function det(C_n)/det(C_2)^(n/2). It assumes, that product
of singular values from HOSVD of tensor is a good approximation of hyperdeterminant
of the tensor (whatever that means).
Returns the value of the maximizin function
"""
function hosvdapprox(Σ::Matrix{T}, c::Array{T,N}, fibres::Vector{Bool} = [fill(true, size(Σ, 1))...]) where {T <: AbstractFloat, N}
  c = reduceband(c, fibres)
  Σ = reduceband(Σ, fibres)
  cunf = unfoldsym(c)
  eigc = abs.(eigvals(cunf*cunf'))
  eigΣ = abs.(eigvals(Σ*Σ'))
  sum(log.(eigc)-N/2*log.(eigΣ))/2
end

"""
  reduceband(ar::Array{N}, k::Vector{Bool})

Returns n-array without values at indices in ind
```jldoctest
julia>  reshape(collect(1.:27.),(3,3,3))
3×3×3 Array{Float64,3}:
[:, :, 1] =
 1.0  4.0  7.0
 2.0  5.0  8.0
 3.0  6.0  9.0

[:, :, 2] =
 10.0  13.0  16.0
 11.0  14.0  17.0
 12.0  15.0  18.0

[:, :, 3] =
 19.0  22.0  25.0
 20.0  23.0  26.0
 21.0  24.0  27.0

julia> reduceband(reshape(collect(1.:27.),(3,3,3)), [true, false, false])
1×1×1 Array{Float64,3}:
[:, :, 1] =
 1.0
```
"""
reduceband(ar::Array{T,N}, fibres::Vector{Bool}) where {T <: AbstractFloat, N} =
  ar[fill(fibres, N)...]


"""
  function unfoldsym{T <: Real, N}(ar::Array{T,N})

Returns a matrix of size (i, k^(N-1)) that is an unfold of symmetric array ar
"""
function unfoldsym(ar::Array{T,N}) where {T <: AbstractFloat, N}
  i = size(ar, 1)
  return reshape(ar, i, i^(N-1))
end

unfoldsym(t::SymmetricTensor{T, N}) where {T <: AbstractFloat, N} = unfoldsym(Array(t))

#greedy algorithm

"""
  greedestep(c::Vector{Array{Float}}, maxfunction::Function, ls::Vector{Bool})

Returns vector of bools that determines bands that maximise a function. True means include
a band, false exclude a band. It changes one true to false in input ls

```jldoctest
julia> a = reshape(collect(1.:9.), 3,3);

julia> b = reshape(collect(1.: 27.), 3,3,3);

julia> testf(ar,bool)= det(ar[1][bool,bool])

julia> greedestep(ar, testf, [true, true, true])
3-element Array{Bool,1}:
true
true
false
```
"""
function greedestep(Σ::Matrix{T}, c::Array{T, N}, maxfunction::Function,
                    ls::Vector{Bool}) where {T <: AbstractFloat, N}
  inds = find(ls)
  bestval = SharedArray{T}(length(ls))
  bestval .= -Inf
  bestls = copy(ls)
  @sync @parallel for i in inds
    templs = copy(ls)
    templs[i] = false
    bestval[i] = maxfunction(Σ, c, templs)
  end
  v, i = findmax(bestval)
  bestls[i] = false
  return bestls, v, i
end

function greedesearchdata(Σ::Matrix{T}, c::Array{T, N}, maxfunction::Function, k::Int) where {T <: AbstractFloat, N}
  ls =  [true for i=1:size(Σ,1)]
  result = []
  k <= size(Σ,1) || throw(AssertionError(" for k = $(k) > size(Σ, 1)"))
  for i = 1:k
    ls, value, j = greedestep(Σ, c, maxfunction, ls)
    push!(result, (ls,value,j))
  end
  result
end

function greedesearchdata(Σ::Matrix{T}, maxfunction::Function, k::Int) where T <: AbstractFloat
  greedesearchdata(Σ, zeros(2,2,2), maxfunction, k)
end

"""
  function cumfsel(Σ::Matrix{T}, c::Array{T, N}, f::String, k::Int = size(Σ, 1)) where {T <: AbstractFloat, N}

Given the covariance matrix of data `Σ` and `c` - the `N`-th cumulant tensor used to
measure the `N`'th order dependencies selects `k`-features (marginals) orderred from lowest to highest
`N`'th order dependencies. `f` is the optimization function, `["hosvd", "norm", "mev"]` are possible.

Returns an Array of tuples `(ind::Array{Bool}, fval::Float64, i::Int)`. First tuple corresponds to the marginal with lowest `N`'th order dependencies with other marginals, while last tuple to the marginal with highest
`N`'th order dependencies.

```jldoctest

julia> srand(42);

julia> using Cumulants

julia> using SymmetricTensors

julia> x = rand(12,10);

julia> c = cumulants(x, 4);

julia> cumfsel(Array(c[2]), Array(c[4]), "hosvd")
10-element Array{Any,1}:
 (Bool[true, true, true, false, true, true, true, true, true, true], 27.2519, 4)
 (Bool[true, true, false, false, true, true, true, true, true, true], 22.6659, 3)
 (Bool[true, true, false, false, false, true, true, true, true, true], 18.1387, 5)
 (Bool[false, true, false, false, false, true, true, true, true, true], 14.4492, 1)
 (Bool[false, true, false, false, false, true, true, false, true, true], 11.2086, 8)
 (Bool[false, true, false, false, false, true, true, false, true, false], 7.84083, 10)
 (Bool[false, false, false, false, false, true, true, false, true, false], 5.15192, 2)
 (Bool[false, false, false, false, false, false, true, false, true, false], 2.56748, 6)
 (Bool[false, false, false, false, false, false, true, false, false, false], 0.30936, 9)
 (Bool[false, false, false, false, false, false, false, false, false, false], 0.0, 7)

```
"""
function cumfsel(Σ::Matrix{T}, c::Array{T, N}, f::String, k::Int = size(Σ, 1)) where {T <: AbstractFloat, N}
  issymetric(c, 1e-4)
  issymetric(Σ, 1e-4)
  if f == "hosvd"
    return greedesearchdata(Σ, c, hosvdapprox, k)
  elseif f == "norm"
    return greedesearchdata(Σ, c, mormbased, k)
  elseif f == "mev"
    return greedesearchdata(Σ, mev, k)
  end
  throw(AssertionError("$(f) not supported use hosvd, norm or mev"))
end

cumfsel(Σ::Matrix{T}, k::Int = size(Σ, 1)) where T <: AbstractFloat = cumfsel(Σ, ones(2,2,2), "mev", k)
