
te = [-0.112639 0.124715 0.124715 0.268717 0.124715 0.268717 0.268717 0.046154]
st = (reshape(te, (2,2,2)))
mat = reshape(te[1:4], (2,2))
ar = reshape(collect(1.:27.),(3,3,3))

@testset "unfoldsym reduce" begin
  @test unfoldsym(st) == reshape(te, (2,4))
  stt = SymmetricTensor(st)
  @test unfoldsym(stt) == reshape(te, (2,4))*reshape(te, (2,4))'
  @test reduceband(ar, [true, false, false])  ≈ ones(Float64, (1,1,1))
  @test reduceband(ar, [true, true, true])  ≈ ar
end

Random.seed!(42)
a = rand(SymmetricTensor{Float64, 2}, 3)
b = rand(SymmetricTensor{Float64, 3}, 3)
testf(a,b,bool)= det(a[bool,bool])
@testset "optimisation" begin
  @testset "greedestep" begin
    g = greedestep(Array(a), Array(b), testf, [true, true, true])
    if VERSION <= v"1.7"
      @test g[1] == [true, false, true]
      @test g[3] == 2
      @test g[2] ≈ 0.48918301293211774
    else
      @test g[1] == [true, true, false]
      @test g[3] == 3
      @test g[2] ≈ 0.09764869605558585
    end
  end
  @testset "greedesearch" begin
    g = greedesearchdata(a,b, testf, 3)
    if VERSION <= v"1.7"
      @test g[1][1] == [true, false, true]
      @test g[2][1] == [false, false, true]
      @test g[3][1] == [false, false, false]
      @test g[1][2] ≈ 0.48918301293211774
      @test g[2][2] ≈ 0.9735659798036858
      @test g[3][2] == 1.0
      @test g[1][3] == 2
      @test g[2][3] == 1
      @test g[3][3] == 3
    else
      @test g[1][1] == [true, true, false]
      @test g[2][1] == [true, false, false]
      @test g[3][1] == [false, false, false]
      @test g[1][2] ≈ 0.09764869605558585
      @test g[2][2] ≈ 0.6293451231426089
      @test g[3][2] == 1.0
      @test g[1][3] == 3
      @test g[2][3] == 2
      @test g[3][3] == 1
    end
  end
end

@testset "target functions" begin
  Σ = [1. 0.5 0.5; 0.5 1. 0.5; 0.5 0.5 1.]
  @test mev(Σ, ones(2,2,2), [true, true, true]) == 0.5
  c3 = ones(3,3,3)
  #@test hosvdapprox(Σ,c3, [true, true, true]) ≈ -33.905320329609154
  c4 = ones(3,3,3,3)
  #@test hosvdapprox(Σ,c4, [true, true, true]) ≈ -30.23685187275532
  @test mormbased(Σ,c4, [true, true, true]) ≈ 2.
  c5 = ones(3,3,3,3,3)
  #@test hosvdapprox(Σ,c5, [true, true, true]) ≈ -29.34097213814129
end

@testset "hosvdapprox additional tests" begin
  Random.seed!(44)
  c3 = rand(SymmetricTensor{Float64, 3}, 5)
  Σ = rand(SymmetricTensor{Float64, 2}, 5)
  m3 = unfoldsym(c3)
  @test size(m3) == (5,5)
  Σ = Array(Σ)
  @test hosvdapprox(Σ, Array(c3)) ≈ log(det(m3)^(1/2)/det(Σ)^(3/2))
  c4 = rand(SymmetricTensor{Float64, 4}, 5)
  m4 = unfoldsym(c4)
  @test size(m4) == (5,5)
  @test hosvdapprox(Σ, Array(c4)) ≈ log(det(m4)^(1/2)/det(Σ)^(4/2))
  c5 = rand(SymmetricTensor{Float64, 5}, 5)
  m5 = unfoldsym(c5)
  @test size(m5) == (5,5)
  @test hosvdapprox(Σ, Array(c5)) ≈ log(det(m5)^(1/2)/det(Σ)^(5/2))
end

@testset "cumfsel tests" begin
 Random.seed!(43)
  Σ = rand(SymmetricTensor{Float64, 2}, 5)
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
  c = SymmetricTensor(c)
  ret = cumfsel(Σ, c, "hosvd", 5)
  retn = cumfsel(Σ, c, "norm", 4)

  if VERSION <= v"1.7"
    @test ret[3][1] == [true, true, false, false, false]
    @test ret[3][2] ≈ 7.943479150509705
    @test (x->x[3]).(ret) == [4, 5, 3, 2, 1] #from lest important to most important"

    @test retn[3][1] == [true, true, false, false, false]
    @test retn[3][2] ≈ 24.285620999564703
    @test (x->x[3]).(retn) == [4, 5, 3, 2]
    @test cumfsel(Σ, c, "mev", 5)[1][3] == 5
    @test cumfsel(Σ, 5)[1][3] == 5
    @test_throws AssertionError cumfsel(Σ, c, "mov", 5)
    @test_throws AssertionError cumfsel(Σ, c, "hosvd", 7)
  else
    @test ret[3][1] == [false, true, true, false, false]
    @test ret[3][2] ≈ 10.943399558215603
    @test (x->x[3]).(ret) == [4, 5, 1, 3, 2] #from lest important to most important"

    @test retn[3][1] == [true, false, true, false, false]
    @test retn[3][2] ≈ 46.617412130431866
    @test (x->x[3]).(retn) == [4, 5, 2, 3]
    @test cumfsel(Σ, c, "mev", 5)[1][3] == 5
    @test cumfsel(Σ, 5)[1][3] == 5
    @test_throws AssertionError cumfsel(Σ, c, "mov", 5)
    @test_throws AssertionError cumfsel(Σ, c, "hosvd", 7)
  end
  Random.seed!(42)
  x = rand(12,10);
  c = cumulants(x,4);
  f = cumfsel(c[2], c[4], "hosvd")
  if VERSION <= v"1.7"
    @test f[9][1] == [false, false, false, false, false, false, true, false, false, false]
    @test f[9][3] == 9
    @test f[10][3] == 7
  else
    @test f[9][1] == [false, false, false, false, false, false, false, false, true, false]
    @test f[9][3] == 6
    @test f[10][3] == 9
  end
end

@testset "cumfsel tests on Float32" begin
  Random.seed!(42)
  x = rand(Float32, 12,10);
  c = cumulants(x,4);
  f = cumfsel(c[2], c[4], "hosvd")
  @test f[9][3] == 6
  @test f[10][3] == 5
end
