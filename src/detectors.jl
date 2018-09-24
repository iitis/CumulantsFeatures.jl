"""
  function rxdetect(X::Matrix{T}, alpha::Float64 = 0.99)

Takes data in the form of matrix where first index correspond to realisations and
second to feratures (marginals).
Using the RX (Reed-Xiaoli) Anomaly Detection returns the array of Bool that
correspond to outlier realisations. alpha is the sensitivity parameter of the RX detector

```jldoctest
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
"""
function rxdetect(X::Matrix{T}, alpha::Float64 = 0.99) where T <: AbstractFloat
  t = size(X,1)
  outliers = fill(false, t)
  mu = mean(X,1)[1,:]
  Kinv = inv(cov(X))
  d = Chisq(size(X,2))
  for i in 1:t
    if (X[i,:] - mu)'*Kinv*(X[i,:] - mu) > quantile(d, alpha)
      outliers[i] = true
    end
  end
  outliers
end


function hosvdstep(X::Matrix{T}, ls::Vector{Bool}, β::Float64, r::Int) where T <: AbstractFloat
  bestls = copy(ls)
  M = unfoldsym(cumulants(X[ls,:], 4)[4])
  W = svd(M)[1][:,1:r]
  Z = X*W
  mm = [mad(Z[ls,i]; center=median(Z[ls,i]), normalize=true) for i in 1:r]
  me = [median(Z[ls,i]) for i in 1:r]
  for i in find(ls)
    if maximum(abs.(Z[i,:].-me)./mm) .> β
     bestls[i] = false
   end
 end
 bestls, vecnorm([kurtosis(Z[bestls,i]) for i in 1:r])
end

"""
  function hosvdc4detect(X::Matrix{T}, β::Float64 = 4.1, r::Int = 3)

Takes data in the form of matrix where first index correspond to realisations and
second to feratures (marginals).
Using the HOSVD of the 4'th cumulant's tensor of data returns the array of Bool that
correspond to outlier realisations. β is the sensitivity parameter while r a
number of specific directions, data are projected onto.

```jldoctest
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
"""
function hosvdc4detect(X::Matrix{T}, β::Float64 = 4.1, r::Int = 3) where T <: AbstractFloat
  X = X.-mean(X,1)
  s = cov(X)
  X = X*Real.(sqrtm(inv(s)))
  ls = fill(true, size(X,1))
  lsold = []
  aold = 1000000000.
  while count(ls) > div(size(X,1)+r+1, 2)
    lsold = copy(ls)
    ls, a = hosvdstep(X, lsold, β, r)
    if (aold-a)/a < 0.0001
      return .!lsold
    end
    aold = a
  end
  .!lsold
end
