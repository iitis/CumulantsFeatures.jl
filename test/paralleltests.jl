@testset "test on parallel cumulants" begin
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

@testset "test on parallel cumulants larger" begin
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

# something wrong for differet block sizes
@testset "test on parallel" begin
    Random.seed!(42)
    t = rand(SymmetricTensor{Float64, 5}, 60, 7)
    @time s = cum2mat(t)
    @time X = unfold(Array(t), 1)
    @time M = X*transpose(X)
    @test maximum(abs.(M - Array(s))) ≈ 0 atol = 10^(-6)
end
