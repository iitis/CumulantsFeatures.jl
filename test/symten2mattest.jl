import CumulantsFeatures: makeblocksize, computeblock

@testset "axiliary for cum2mat" begin
    Random.seed!(42)
    t = rand(SymmetricTensor{Float64, 3}, 9, 4)
    @test makeblocksize(t, (1,1)) == (4,4)
    @test makeblocksize(t, (1,3)) == (4,1)
    @test makeblocksize(t, (3,1)) == (1,4)
    t1 = rand(SymmetricTensor{Float64, 3}, 9, 3)
    @test makeblocksize(t1, (1,1)) == (3,3)
    dims = (fill(t.bln, 1)...,)
    if VERSION <= v"1.7"
        @test computeblock(t, (1,1), dims)[1,1] ≈ 21.8503 atol = 0.01
    else
        @test computeblock(t, (1,1), dims)[1,1] ≈ 19.1174 atol = 0.01
    end
end

@testset "cum2mat tests on cumulants" begin
    x = rand(20,13)
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

@testset "cum2mat tests on random tensor one core" begin
    @testset "order 3" begin
        Random.seed!(42)
        t = rand(SymmetricTensor{Float64, 3}, 12, 4)
        s = cum2mat(t)
        X = unfold(Array(t), 1)
        M = X*transpose(X)
        @test maximum(abs.(M - Array(s))) ≈ 0 atol = 10^(-10)
    end
    @testset "order 4" begin
        t = rand(SymmetricTensor{Float64, 4}, 12, 3)
        s = cum2mat(t)
        X = unfold(Array(t), 1)
        M = X*transpose(X)
        @test maximum(abs.(M - Array(s))) ≈ 0 atol = 10^(-10)
    end
    @testset "order 5" begin
        t = rand(SymmetricTensor{Float64, 5}, 10, 2)
        s = cum2mat(t)
        X = unfold(Array(t), 1)
        M = X*transpose(X)
        @test maximum(abs.(M - Array(s))) ≈ 0 atol = 10^(-10)
    end
    @testset "order 5 not squared" begin
        t = rand(SymmetricTensor{Float64, 5}, 11, 3)
        s = cum2mat(t)
        X = unfold(Array(t), 1)
        M = X*transpose(X)
        @test maximum(abs.(M - Array(s))) ≈ 0 atol = 10^(-10)
    end
    @testset "order 6" begin
        t = rand(SymmetricTensor{Float64, 6}, 9, 3)
        s = cum2mat(t)
        X = unfold(Array(t), 1)
        M = X*transpose(X)
        @test maximum(abs.(M - Array(s))) ≈ 0 atol = 10^(-10)
    end
    @testset "order 6 not squared" begin
        t = rand(SymmetricTensor{Float64, 6}, 7, 3)
        s = cum2mat(t)
        X = unfold(Array(t), 1)
        M = X*transpose(X)
        @test maximum(abs.(M - Array(s))) ≈ 0 atol = 10^(-10)
    end
    @testset "larger data set" begin
        t = rand(SymmetricTensor{Float64, 5}, 26, 4)
        s = cum2mat(t)
        X = unfold(Array(t), 1)
        M = X*transpose(X)
        @test maximum(abs.(M - Array(s))) ≈ 0 atol = 10^(-8)
    end
end


@testset "test on Gaussian data" begin
    Random.seed!(1234)
    s = 0.5*(ones(4,4)+1*Matrix(I, 4, 4))
    x = Array(rand(MvNormal(s), 1_000_000)');
    c = cumulants(x, 5)
    M3 = cum2mat(c[3])
    M4 = cum2mat(c[4])
    M5 = cum2mat(c[5])

    @test norm(c[2]) ≈ 2.64 atol = 10^(-2)
    @test norm(M3) ≈ 0. atol = 10^(-4)
    @test norm(M4)≈ 0. atol = 3*10^(-3)
    @test norm(M5)≈ 0. atol = 4*10^(-2)
end
