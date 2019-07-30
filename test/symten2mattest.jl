import CumulantsFeatures: makeblocksize, computeblock

@testset "axiliary" begin
    Random.seed!(42)
    t = rand(SymmetricTensor{Float64, 3}, 9, 4)
    @test makeblocksize(t, (1,1)) == (4,4)
    @test makeblocksize(t, (1,3)) == (4,1)
    @test makeblocksize(t, (3,1)) == (1,4)
    t1 = rand(SymmetricTensor{Float64, 3}, 9, 3)
    @test makeblocksize(t1, (1,1)) == (3,3)
    dims = (fill(t.bln, 1)...,)
    @test computeblock(t, (1,1), dims) ≈ [21.8503 16.2408 20.5519 18.4874; 16.2408 25.1741 18.1145 19.4899; 20.5519 18.1145 31.5073 22.8366; 18.4874 19.4899 22.8366 28.156] atol = 0.01
end

@testset "random sym tensors" begin
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
end


@testset "test on cumulants" begin
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

@testset "test on larger set" begin
    t = rand(SymmetricTensor{Float64, 5}, 60, 7)
    @time s = cum2mat(t)
    @time X = unfold(Array(t), 1)
    @time M = X*transpose(X)
    @test maximum(abs.(M - Array(s))) ≈ 0 atol = 10^(-6)
end
