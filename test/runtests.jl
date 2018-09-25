using Base.Test
using SymmetricTensors
using Cumulants
using CumulantsFeatures
using Combinatorics
import CumulantsFeatures: reduceband, greedestep, unfoldsym, hosvdstep, greedesearchdata, mev, mormbased, hosvdapprox

te = [-0.112639 0.124715 0.124715 0.268717 0.124715 0.268717 0.268717 0.046154]
st = (reshape(te, (2,2,2)))
mat = reshape(te[1:4], (2,2))
ar = reshape(collect(1.:27.),(3,3,3))

@testset "unfoldsym reduce" begin
  @test unfoldsym(st) == reshape(te, (2,4))
  stt = convert(SymmetricTensor, st)
  @test unfoldsym(stt) == reshape(te, (2,4))
  @test reduceband(ar, [true, false, false])  ≈ ones(Float64, (1,1,1))
  @test reduceband(ar, [true, true, true])  ≈ ar
end

a = reshape(collect(1.:9.), 3,3)
b = reshape(collect(1.: 27.), 3,3,3)
testf(a,b,bool)= det(a[bool,bool])
@testset "optimisation" begin
  @testset "greedestep" begin
    g = greedestep(a,b, testf, [true, true, true])
    @test g[1] == [true, true, false]
    @test g[3] == 3
    @test g[2] == -3.0
  end
  @testset "greedesearch" begin
    g = greedesearchdata(a,b, testf, 3)
    @test g[1][1] == [true, true, false]
    @test g[2][1] == [false, true, false]
    @test g[3][1] == [false, false, false]
    @test g[1][2] == -3.0
    @test g[2][2] == 5.0
    @test g[3][2] == 1.0
    @test g[1][3] == 3
    @test g[2][3] == 1
    @test g[3][3] == 2
  end
end

@testset "target functions" begin
  Σ = [1. 0.5 0.5; 0.5 1. 0.5; 0.5 0.5 1.]
  @test mev(Σ, ones(2,2,2), [true, true, true]) == 0.5
  c3 = ones(3,3,3)
  @test hosvdapprox(Σ,c3, [true, true, true]) ≈ -33.905320329609154
  c4 = ones(3,3,3,3)
  @test hosvdapprox(Σ,c4, [true, true, true]) ≈ -30.23685187275532
  @test mormbased(Σ,c4, [true, true, true]) ≈ 2.
  c5 = ones(3,3,3,3,3)
  @test hosvdapprox(Σ,c5, [true, true, true]) ≈ -29.34097213814129
end

@testset "hosvdapprox additional tests" begin
  srand(42)
  c3 = rand(SymmetricTensor{Float64, 3}, 5)
  Σ = rand(SymmetricTensor{Float64, 2}, 5)
  c3 = Array(c3)
  c3m = unfoldsym(c3)
  m3 = c3m*c3m'
  @test size(m3) == (5,5)
  Σ = Array(Σ)
  @test hosvdapprox(Σ, c3) ≈ log(det(m3)^(1/2)/det(Σ)^(3/2))
  c4 = rand(SymmetricTensor{Float64, 4}, 5)
  c4 = Array(c4)
  c4m = unfoldsym(c4)
  m4 = c4m*c4m'
  @test size(m4) == (5,5)
  @test hosvdapprox(Σ, c4) ≈ log(det(m4)^(1/2)/det(Σ)^(4/2))
  c5 = rand(SymmetricTensor{Float64, 5}, 5)
  c5 = Array(c5)
  c5m = unfoldsym(c5)
  m5 = c5m*c5m'
  @test size(m5) == (5,5)
  @test hosvdapprox(Σ, c5) ≈ log(det(m5)^(1/2)/det(Σ)^(5/2))
end

@testset "cumfsel tests" begin
 srand(43)
  Σ = rand(SymmetricTensor{Float64, 2}, 5)
  Σ = Array(Σ)
  c = 0.1*ones(5,5,5)
  c[1,1,1] = 20.
  c[2,2,2] = 10.
  c[3,3,3] = 10.
  for j in permutations([1,2,3])
      c[j...] = 20.
  end
  for j in permutations([1,2,2])
      c[j...] = 20.
  end
  for j in permutations([2,2,3])
      c[j...] = 10.
  end
  for j in permutations([1,3,3])
      c[j...] = 20.
  end
  ret = cumfsel(Σ, c, "hosvd", 5)
  @test ret[3][1] == [true, true, false, false, false]
  @test ret[3][2] ≈ 7.943479150509705
  @test (x->x[3]).(ret) == [4, 5, 3, 2, 1] #from lest important to most important"
  retn = cumfsel(Σ, c, "norm", 4)
  @test retn[3][1] == [true, true, false, false, false]
  @test retn[3][2] ≈ 24.285620999564703
  @test (x->x[3]).(retn) == [4, 5, 3, 2]
  @test cumfsel(Σ, c, "mev", 5)[1][3] == 5
  @test cumfsel(Σ, 5)[1][3] == 5
  @test_throws AssertionError cumfsel(Σ, c, "mov", 5)
  @test_throws AssertionError cumfsel(Σ, c, "hosvd", 7)
  @test_throws AssertionError cumfsel(Σ, rand(5,5,5), "hosvd", 5)
  @test_throws RemoteException cumfsel(Σ, c[1:4, 1:4, 1:4], "hosvd", 5)
end

@testset "detectors" begin
  srand(42)
  x = vcat(rand(8,2), 20*rand(2,2))
  @test rxdetect(x, 0.95) == [false, false, false, false, false, false, false, false, true, true]
  @test hosvdc4detect(x, 3., 1) == [false, false, false, false, false, false, false, false, true, true]
  ls = fill(true, 10)
  @test hosvdstep(x, ls, 3., 1)[1] == [true, true, true, true, true, true, true, true, false, false]
  @test hosvdstep(x, ls, 3., 1)[2] ≈ 1.43120851350894
end

addprocs(3)
@everywhere testf(a,b,bool)= det(a[bool,bool])
@everywhere using CumulantsFeatures
@testset "greedesearch parallel implementation" begin
  g = greedesearchdata(a,b, testf, 3)
  @test g[1][1] == [true, true, false]
  @test g[2][1] == [false, true, false]
  @test g[3][1] == [false, false, false]
  @test g[1][2] == -3.0
  @test g[2][2] == 5.0
  @test g[3][2] == 1.0
  @test g[1][3] == 3
  @test g[2][3] == 1
  @test g[3][3] == 2
end
