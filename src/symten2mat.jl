"""
  makeblocksize(bm::SymmetricTensor{T, N}, i::Tuple{Int, Int}) where {T <: AbstractFloat, N}

Returns Tuple{Int, Int} of the size ob the block in the output matrix

"""

function makeblocksize(bm::SymmetricTensor{T, N}, i::Tuple{Int, Int}) where {T <: AbstractFloat, N}
  a = bm.bls
  b = bm.bls
  if !bm.sqr
    if i[1] == bm.bln
      a = bm.dats - (bm.bln-1)*bm.bls
    end
    if i[2] == bm.bln
      b = bm.dats - (bm.bln-1)*bm.bls
    end
  end
  a,b
end

"""
  computeblock(bm::SymmetricTensor{T, N}, i::Tuple{Int, Int}, dims::Tuple) where {T <: AbstractFloat, N}

Returns Matrix{T}, the single block of the output higher correlation matrix

"""

function computeblock(bm::SymmetricTensor{T, N}, i::Tuple{Int, Int}, dims::Tuple) where {T <: AbstractFloat, N}
  x = bm.bln^(N-2)
  R = zeros(T, makeblocksize(bm, i))
  for j in 1:x
    @inbounds k = Tuple(CartesianIndices(dims)[j])
    for k1 in 1:bm.bln
      @inbounds M1 = unfold(getblock(bm, (i[1],k1, k...)),1)
      @inbounds M2 = unfold(getblock(bm, (k1,i[2],k...)),2)
      @inbounds R += M1*transpose(M2)
    end
  end
  return R
end

"""
  computeblock_p(bm::SymmetricTensor{T, N}, i::Tuple{Int, Int}, dims::Tuple) where {T <: AbstractFloat, N}

Returns Matrix{T}, the single block of the output higher correlation matrix
Parallel implementation
"""

function computeblock_p(bm::SymmetricTensor{T, N}, i::Tuple{Int, Int}, dims::Tuple) where {T <: AbstractFloat, N}
  x = bm.bln^(N-2)
  R = SharedArray(zeros(T, (x, makeblocksize(bm, i)...)))
  @sync @distributed for j in 1:x
    @inbounds k = Tuple(CartesianIndices(dims)[j])
    for k1 in 1:bm.bln
      @inbounds M1 = unfold(getblock(bm, (i[1],k1, k...)),1)
      @inbounds M2 = unfold(getblock(bm, (k1,i[2],k...)),2)
      @inbounds R[j,:,:] += M1*transpose(M2)
    end
  end
  R = Array(R)
  R = mapreduce(i -> R[i,:,:], +, 1:size(R,1))
  return R
end
"""
  cum2mat(bm::SymmetricTensor{T, N}) where {T <: AbstractFloat, N}

Returns higher order correlation matrix in the form of bm::SymmetricTensor{T, 2}


```jldoctest

julia> Random.seed!(42);

julia> t = rand(SymmetricTensor{Float64, 3}, 4);

julia> cum2mat(t)
SymmetricTensor{Float64,2}(Union{Nothing, Array{Float64,2}}[[7.69432 4.9757; 4.9757 5.72935] [6.09424 4.92375; 5.05157 3.17723]; nothing [7.33094 4.93128; 4.93128 4.7921]], 2, 2, 4, true)

```
"""

function cum2mat(bm::SymmetricTensor{T, N}) where {T <: AbstractFloat, N}
    ret = arraynarrays(T, bm.bln, bm.bln)
    ds = (fill(bm.bln, N-2)...,)
    if nworkers() == 1
      for i in pyramidindices(2, bm.bln)
        @inbounds ret[i...] = computeblock(bm, i, ds)
      end
    else
      for i in pyramidindices(2, bm.bln)
        @inbounds ret[i...] = computeblock_p(bm, i, ds)
      end
    end
    SymmetricTensor(ret; testdatstruct = false)
end
