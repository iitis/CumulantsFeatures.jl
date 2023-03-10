# CumulantsFeatures.jl

[![Coverage Status](https://coveralls.io/repos/github/iitis/CumulantsFeatures.jl/badge.svg?branch=master)](https://coveralls.io/github/iitis/CumulantsFeatures.jl?branch=master)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7716695.svg)](https://doi.org/10.5281/zenodo.7716695)

CumulantsFeatures.jl uses multivariate cumulants to provide the algorithms for the outliers detection and the features selection given the multivariate data represented in the form of `t x n` matrix of Floats, `t` numerates the realisations, while `n` numerates the marginals.

Requires SymmetricTensors.jl Cumulants.jl and CumulantsUpdates.jl to compute and update multivariate cumulants of data.

As of 24/09/2018 [@kdomino](https://github.com/kdomino) is the lead maintainer of this package.

## Installation

Within Julia, run

```julia
pkg> add CumulantsFeatures
```

Parallel computation is supported

## Features selection

Given `n`-variate data,  iteratively determines its `k`-marginals that are little informative.
Uses `C2`- the covariance matrix, and `CN` - the `N`th cumulant's tensor, both in the `SymmetricTensor` type, see SymmetricTensors.jl. Uses one of the following optimisation functions
`f`: `["hosvd", "norm", "mev"].

```julia

julia> function cumfsel(C2::SymmetricTensor{T,2}, CN::SymmetricTensor{T, N}, f::String, k::Int = n) where {T <: AbstractFloat, N}

```
The "norm" uses the norm of the higher-order cumulant tensor, this is a benchmark method for comparison. 

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

The output is the Array of tuples `(ind::Array{Bool}, fval::Float64, i::Int)`, each tuple corresponds to the one step
of the features selection. Marginals are removed in the information hierarchy, starting from the least informatve and ending on the most infomrative.

The vector `ind` consist of `false` that determines the removed marginal, and `true` that determines the left marginal. 

The `fval` is the value of the target function.

The `i` numerates the marginal removed at the given step.

To limit number of steps use the default parameter `k`:

```julia

julia> cumfsel(Array(c[2]), Array(c[4]), "hosvd", 2)
2-element Array{Any,1}:
 (Bool[true, true, true, false, true, true, true, true, true, true], 27.2519, 4)
 (Bool[true, true, false, false, true, true, true, true, true, true], 22.6659, 3)

```

For the mev optimization run:

```julia

julia> cumfsel(Σ::SymmetricTensor{T,2}, k::Int = Σ.dats)

```


## The higher-order cross-correlation matrix

```julia

  cum2mat(c::SymmetricTensor{T, N}) where {T <: AbstractFloat, N}

```
Returns the higher-order cross-correlation matrix in the form of `SymmetricTensor{T, 2}`. Such matrix is the contraction of the corresponding higher-order cumulant tensor `c::SymmetricTensor{T, N}`
with itself in all modes but one.

```julia

julia> Random.seed!(42);

julia> t = rand(SymmetricTensor{Float64, 3}, 4);

julia> cum2mat(t)
SymmetricTensor{Float64,2}(Union{Nothing, Array{Float64,2}}[[7.69432 4.9757; 4.9757 5.72935] [6.09424 4.92375; 5.05157 3.17723]; nothing [7.33094 4.93128; 4.93128 4.7921]], 2, 2, 4, true)

```

## Outliers detection

Let `X` be the multivariate data represented in the form of `t x n` matrix of Floats, `t` numerates the realisations, while `n` numerates the marginals.

### RX detector

```julia

  rxdetect(X::Matrix{T}, α::Float64 = 0.99)

```

The RX (Reed-Xiaoli) Anomaly Detection returns the array of Bool, where `true`
corresponds to the outlier realisations while `false` corresponds to the ordinary data. The parameter `α` is the sensitivity (threshold) parameter of the RX detector.


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

### The 4th order multivariate cumulant outlier detector

```julia

  function hosvdc4detect(X::Matrix{T}, β::Float64 = 4.1, r::Int = 3)

```
The 4th order multivariate cumulant outlier detector returns the array of Bool, where `true`
corresponds to the outlier realisations while `false` corresponds to the ordinary data. The parameter `β` is the sensitivity parameter, the parameter `r` is the number of specific directions (with high `4`th order cumulant) on which data are projected. See K. Domino: 'Multivariate cumulants in outlier detection for financial data analysis', [arXiv:1804.00541] (https://arxiv.org/abs/1804.00541). 

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
## Tests on artificial data.

In folder `benchmarks/outliers_detect` and `benchmarks/features_select` there are the Julia executable files for testing features selection and outliers detection on artificial data.

### Features selection

In `./benchmarks/features_select` the executable file `gendat4selection.jl` generates multivariate data where the subset of `infomrative` margianls is modelled by the t-Student copula with `--nu` degrees of freedom (by defalt `4`). All univariate marginal distributions are t-Student with `-nuu` degrees of freedom (by defalt `25`).


The `gendat4selection.jl` returns a `.jld2` file with data. Run `jkfs_selection.jl` on this file to display the characteristics of features selection plotted in `./benchmarks/features_select/pics/`

### Outlier detection

In `./benchmarks/outliers_detect/` the executable file `gendat4detection.jl` generates multivariate data with outliers modelled by the t-Student copula with `--nu` degrees of freedom (by defalt `6`). All univariate marginal distributions are t-Student with `--nuu` degrees of freedom (by defalt `6`). The number of test realisations is `--reals` (by default `5`).

The `gendat4detection.jl` returns a `.jld2` file with data. Run `detect_outliers.jl` on this file to display the characteristics of outlier detection plotted in `./benchmarks/outliers_detect/pics/'
`

# Citing this work

This project was partially financed by the National Science Centre, Poland – project number 2014/15/B/ST6/05204.

While using `hosvdc4detect()` - please cite: K. Domino: 'Multivariate cumulants in outlier detection for financial data analysis', Physica A: Statistical Mechanics and its Applications Volume 558, 15 November 2020, 124995 (https://doi.org/10.1016/j.physa.2020.124995).


While using `cumfsel()` - please cite: P. Głomb, K. Domino, M. Romaszewski, M. Cholewa, 'Band selection with Higher Order Multivariate Cumulants for small target detection in hyperspectral images', Wroclaw University of Science and Technology, Conference Proceedings: PP-RAI'2019 (2019), ISBN: 978-83-943803-2-8; [arxiv: 1808.03513] (https://arxiv.org/abs/1808.03513).
