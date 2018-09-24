#CumFSel.jl
[![Build Status](https://travis-ci.org/ZKSI/CumFSel.jl.svg?branch=master)](https://travis-ci.org/ZKSI/CumFSel.jl)




CumFSel.jl provides Cumulants based algorithms used to select features subset or detect an outlier subset that posses higher order cross-correlations. 
An outlier subset is assumed to be modelled by non-Gausian multivariate distribution in contrary to an ordinary data subset that is assumed to be modelled by a Gaussian multivariate distribution.

As of 24/09/2018 [@kdomino](https://github.com/kdomino) is the lead maintainer of this package.

Julia 0.6 is required.

## Features selction

Given the covariance matrix of data `Σ` and `c` - the `N`-th cumulant tensor used to measure the `N`'th order dependencies to select `k`-features (marginals) orderred from lowest to highest
`N`'th order dependencies run:

```julia

julia> cumfsel(Σ::Matrix{T}, c::Array{T, N}, f::String, k::Int = size(Σ, 1)) where {T <: AbstractFloat, N}

```

Here `f` is optimization function, `["hosvd", "norm", "mev"]` are possible. The "hosvd" uses the Higher Order Singular Value decomposition approximation of higher order cumulant's tensor to extract information
about higher order correlations. The "norm" uses simpy the norm of the cumulant's tensor. The "mev" takes only the corrlation matric and second order correlations.

```julia

julia>  using CumFSel

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

Returns an Array of tuples `(ind::Array{Bool}, fval::Float64, i::Int)`. First tuple corresponds to the marginal with lowest `N`'th order dependencies with other marginals, while last tuple to the marginal with highest
`N`'th order dependencies. The `fval` is the value of the target function and `i` indexes the marginal. 


If one wants to get only a few (e.g. `2`) marginals with lowest `N`'th order dependencies run: 

```julia

julia> cumfsel(Array(c[2]), Array(c[4]), "hosvd", 2)
2-element Array{Any,1}:
 (Bool[true, true, true, false, true, true, true, true, true, true], 27.2519, 4) 
 (Bool[true, true, false, false, true, true, true, true, true, true], 22.6659, 3)

```

If running

```julia

julia> cumfsel(Σ::Matrix{T}, k::Int = size(Σ, 1))

```
The mev optimization function will be used.

## Detection

### RX detector

```julia

  rxdetect(X::Matrix{T}, alpha::Float64 = 0.99)
  
```

Takes data `X` in the form of matrix where first index correspond to realisations and
second to feratures (marginals). Using the RX (Reed-Xiaoli) Anomaly Detection returns the array of Bool that
correspond to outlier realisations. `alpha` is the sensitivity parameter of the RX detector.


```julia
julia> srand(42);

julia> x = vcat(rand(8,2), 20*rand(2,2))
10×2 Array{Float64,2}:
  0.533183    0.956916
  0.454029    0.584284
  0.0176868   0.937466
  0.172933    0.160006
  0.958926    0.422956
  0.973566    0.602298
  0.30387     0.363458
  0.176909    0.383491
 11.8582      5.25618 
 14.9036     10.059   

julia> rxdetect(x, 0.95)
10-element Array{Bool,1}:
 false
 false
 false
 false
 false
 false
 false
 false
  true
  true
```

### The HOSVD of the `4` th cumulant

```julia

  function hosvdc4detect(X::Matrix{T}, β::Float64 = 4.1, r::Int = 3)
  
```


Takes data in the form of matrix where first index correspond to realisations and
second to feratures (marginals).Using the HOSVD of the `4`'th cumulant's tensor of data returns the array of Bool that
correspond to outlier realisations. `β` is the sensitivity parameter while `r` a
number of specific directions, data are projected onto.

```julia

julia> srand(42);

julia> x = vcat(rand(8,2), 20*rand(2,2))
10×2 Array{Float64,2}:
  0.533183    0.956916
  0.454029    0.584284
  0.0176868   0.937466
  0.172933    0.160006
  0.958926    0.422956
  0.973566    0.602298
  0.30387     0.363458
  0.176909    0.383491
 11.8582      5.25618
 14.9036     10.059

julia> rxdetect(x, 0.95)
10-element Array{Bool,1}:
 false
 false
 false
 false
 false
 false
 false
 false
  true
  true
```


# Citing this work

This project was partially financed by the National Science Centre, Poland – project number 2014/15/B/ST6/05204.

Please cite K. Domino: ' The use of the Higher Order Singular Value Decomposition of the 4-cumulant's tensors in features selection and outlier detection', [arXiv:1804.00541] (https://arxiv.org/abs/1804.00541).
