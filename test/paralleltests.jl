Random.seed!(42)
a = rand(SymmetricTensor{Float64, 2}, 3)
b = rand(SymmetricTensor{Float64, 3}, 3)
@everywhere using LinearAlgebra
@everywhere testf(a,b,bool)= det(a[bool,bool])
println("nworkers()")
println(nworkers())


@testset "greedesearch parallel implementation" begin
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

Random.seed!(42)

@testset "cum2mat parallel tests on cumulants" begin
    @testset "small data sie" begin
        x = rand(200,12)
        c = cumulants(x,5,2)
        s = cum2mat(c[3])
        X = unfold(Array(c[3]), 1)
        M = X*transpose(X)
        @test maximum(abs.(M - Array(s))) ≈ 0 atol = 10^(-10)

        s = cum2mat(c[4])
        X = unfold(Array(c[4]), 1)
        M = X*transpose(X)
        @test maximum(abs.(M - Array(s))) ≈ 0 atol = 10^(-10)

        s = cum2mat(c[5])
        X = unfold(Array(c[5]), 1)
        M = X*transpose(X)
        @test maximum(abs.(M - Array(s))) ≈ 0 atol = 10^(-10)
    end

    @testset "larger data size" begin
        x = rand(200,27)
        c = cumulants(x,5,5)
        s = cum2mat(c[3])
        X = unfold(Array(c[3]), 1)
        M = X*transpose(X)
        @test maximum(abs.(M - Array(s))) ≈ 0 atol = 10^(-10)

        s = cum2mat(c[4])
        X = unfold(Array(c[4]), 1)
        M = X*transpose(X)
        @test maximum(abs.(M - Array(s))) ≈ 0 atol = 10^(-10)

        s = cum2mat(c[5])
        X = unfold(Array(c[5]), 1)
        M = X*transpose(X)
        @test maximum(abs.(M - Array(s))) ≈ 0 atol = 10^(-10)
    end
end

@testset "cum2mat parallel tests on random tensors" begin
    @testset "m = 5, n = 24, d = 4" begin
        Random.seed!(42)
        t = rand(SymmetricTensor{Float64, 5}, 24, 4)
        println("cum2mat time = ")
        @time s = cum2mat(t)
        print("naive time = ")
        @time begin X = unfold(Array(t), 1)
        M = X*transpose(X)
        end
        @test maximum(abs.(M - Array(s))) ≈ 0 atol = 10^(-8)
    end

    @testset "m = 5, n = 40, d = 5" begin
        t = rand(SymmetricTensor{Float64, 5}, 40, 5)
        s = cum2mat(t)
        X = unfold(Array(t), 1)
        M = X*transpose(X)
        @test maximum(abs.(M - Array(s))) ≈ 0 atol = 10^(-8)
    end

    @testset "m = 5, n = 37, d = 6" begin
        t = rand(SymmetricTensor{Float64, 5}, 37, 6)
        s = cum2mat(t)
        X = unfold(Array(t), 1)
        M = X*transpose(X)
        @test maximum(abs.(M - Array(s))) ≈ 0 atol = 10^(-8)
    end

    @testset "m = 5, n = 26, d = 5" begin
        t = rand(SymmetricTensor{Float64, 5}, 26, 5)
        s = cum2mat(t)
        X = unfold(Array(t), 1)
        M = X*transpose(X)
        @test maximum(abs.(M - Array(s))) ≈ 0 atol = 10^(-8)
    end

    @testset "m = 5, n = 25, d = 5" begin
        t = rand(SymmetricTensor{Float64, 5}, 25, 5)
        s = cum2mat(t)
        X = unfold(Array(t), 1)
        M = X*transpose(X)
        @test maximum(abs.(M - Array(s))) ≈ 0 atol = 10^(-8)
    end

    @testset "m = 5, n = 29, d = 3" begin
        t = rand(SymmetricTensor{Float64, 5}, 29, 3)
        s = cum2mat(t)
        X = unfold(Array(t), 1)
        M = X*transpose(X)
        @test maximum(abs.(M - Array(s))) ≈ 0 atol = 10^(-8)
    end

    @testset "m = 4, n = 12, d = 3" begin
        t = rand(SymmetricTensor{Float64, 4}, 12, 3)
        s = cum2mat(t)
        X = unfold(Array(t), 1)
        M = X*transpose(X)
        @test maximum(abs.(M - Array(s))) ≈ 0 atol = 10^(-8)
    end
end
