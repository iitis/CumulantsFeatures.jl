# CumulantsFeatures.jl


[![Build Status](https://travis-ci.org/iitis/CumulantsFeatures.jl.svg?branch=master)](https://travis-ci.org/iitis/CumulantsFeatures.jl)
[![Coverage Status](https://coveralls.io/repos/github/iitis/CumulantsFeatures.jl/badge.svg?branch=master)](https://coveralls.io/github/iitis/CumulantsFeatures.jl?branch=master)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3454453.svg)](https://doi.org/10.5281/zenodo.3454453)

CumulantsFeatures.jl uses multivariate cumulants to provide the algorithms for the outliers detection and the features selection given the multivariate data represented in the form of `t x n` matrix of Floats, `t` numerates realisations, while `n` numerates number of marginals.

Requires SymmetricTensors.jl Cumulants.jl and CumulantsUpdates.jl to compute and update multivariate cumulants of data.

As of 24/09/2018 [@kdomino](https://github.com/kdomino) is the lead maintainer of this package.

## Installation

Within Julia, run

```julia
pkg> add CumulantsFeatures
```

## Features selection

Given `n`-variate data,  iteratively determines its `k`-marginals that are little informative.
Uses `C2`- the covariance matrix, and `CN` - the `N`th cumulant's tensor, both in the `SymmetricTensor` type, see SymmetricTensors.jl. Uses one of the following optimisation functions
`f`: `["hosvd", "norm", "mev"].

```julia

julia> function cumfsel(C2::SymmetricTensor{T,2}, CN::SymmetricTensor{T, N}, f::String, k::Int = n) where {T <: AbstractFloat, N}

```
The "norm" uses the norm of the higher-order cumulant's tensor, this is a benchmark method for comparison. 

The "mev" uses only the corrlelation matrix, see: C. Sheffield, 'Selecting band combinations from multispectral data', Photogrammetric Engineering and Remote Sensing, vol. 51 (1985)

The "hosvd" uses the Higher Order Singular Value decomposition of cumulant's tensor to extract information. For the `N=3` case, the Joint Skewness Band Selection (JSBS), see X. Geng, K. Sun, L. Ji, H. Tang & Y. Zhao 'Joint Skewness and Its Application in Unsupervised Band Selection for Small Target Detection Sci Rep. vol.5 (2015) (https://www.nature.com/articles/srep09915). For the JSBS application in biomedical data analysis see: M. Domino, K. Domino, Z. Gajewski, 'An application of higher order multivariate cumulants in modelling of myoelectrical activity of porcine uterus during early pregnancy', Biosystems (2018), (https://doi.org/10.1016/j.biosystems.2018.10.019). For `N = 4` and `N = 5` see also P. Głomb, K. Domino, M. Romaszewski, M. Cholewa 'Band selection with Higher Order Multivariate Cumulants for small target detection in hyperspectral images' (2018) (https://arxiv.org/abs/1808.03513). 

```julia

julia> Random.seed!(42);

julia> using Cumulants

julia> using SymmetricTensors

julia> x = rand(12,10);

julia> c = cumulants(x, 4);

julia> cumfsel(c[2], c[4], "hosvd")
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
`N`'th order dependencies. The `k`'th array gives an outcome after `k` steps. Here `ind` shows `k` marginals that yields lowest `N`'th order dependencies, `fval` the value of the target function at `k`'th step and `i` numerates the marginal removed at step `k`.

To limit number of steps (e.g. to `2`) run:

```julia

julia> cumfsel(Array(c[2]), Array(c[4]), "hosvd", 2)
2-element Array{Any,1}:
 (Bool[true, true, true, false, true, true, true, true, true, true], 27.2519, 4)
 (Bool[true, true, false, false, true, true, true, true, true, true], 22.6659, 3)

```

If running

```julia

julia> cumfsel(Σ::SymmetricTensor{T,2}, k::Int = Σ.dats)

```
The mev optimization function will be used.

## Matrix of higher order correlations

```julia

  cum2mat(c::SymmetricTensor{T, N}) where {T <: AbstractFloat, N}

```
Returns a matrix SymmetricTensor{T, 2} being a contraction of tensor c
with itself in all modes but one.

```julia

julia> Random.seed!(42);

julia> t = rand(SymmetricTensor{Float64, 3}, 4);

julia> cum2mat(t)
SymmetricTensor{Float64,2}(Union{Nothing, Array{Float64,2}}[[7.69432 4.9757; 4.9757 5.72935] [6.09424 4.92375; 5.05157 3.17723]; nothing [7.33094 4.93128; 4.93128 4.7921]], 2, 2, 4, true)

Parallel computation is supported
```

## Outliers detection

### RX detector

```julia

  rxdetect(X::Matrix{T}, α::Float64 = 0.99)

```

Takes data `X` in the form of matrix where first index correspond to realisations and
second to features (marginals). Using the RX (Reed-Xiaoli) Anomaly Detection returns the array of Bool that
correspond to outlier realisations. `α` is the sensitivity (threshold) parameter of the RX detector.


```julia
julia> Random.seed!(42);

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

### The HOSVD of the `4`'th cumulant

```julia

  function hosvdc4detect(X::Matrix{T}, β::Float64 = 4.1, r::Int = 3)

```


Takes data in the form of matrix where first index correspond to realisations and
second to features (marginals). Using the HOSVD of the `4`'th cumulant's tensor of data returns the array of `Bool` that correspond to outlier realisations. For the detector introduction see see K. Domino: 'The use of the Higher Order Singular Value Decomposition of the 4-cumulant's tensors in features selection and outlier detection', [arXiv:1804.00541] (https://arxiv.org/abs/1804.00541) (2018). The parameter `β` is the sensitivity parameter while `r` a
number of specific directions, data are projected onto.

```julia

julia> Random.seed!(42);

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
## Data generation and tests

In folder `test\outliers_detect` and `test\features_select` there are Julia executable files testing selection and detection algorithms on artificial data.

### Features selection

 The executable file `gendat4selection.jl` generates multivariate data with non-Gaussian subset of marginals modelled by the t-Student copula. This file is parametrised by an integer being a number of degrees of freedom of the t-Student copula. Returns a `.jld2` file with data. Run `jkfs_selection.jl` to achieve results of features selection given different methods.

### Outlier detection

 The executable file `gendat4detection.jl` generates multivariate data with non-Gaussian outliers subset of realisations modeled by the t-Student copula.
 This file is parametrised by an integer being a number of degrees of freedom of the t-Student copula. Returns a `.jld2` file with data. Run `detect_outliers.jl` to detect outliers and compare the "HOSVD" based method with the "RX" detector.

# Citing this work

This project was partially financed by the National Science Centre, Poland – project number 2014/15/B/ST6/05204.

While using `hosvdc4detect()` or `cumfsel()` please cite K. Domino: 'Multivariate cumulants in features selection and outlier detection for financial data analysis', [arXiv:1804.00541] (https://arxiv.org/abs/1804.00541).
