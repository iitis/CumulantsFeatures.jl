@testset "detectors" begin
  Random.seed!(42)
  x = vcat(rand(8,2), 5*rand(1,2), 30*rand(1,2))
  x1 = vcat(rand(20,3), 5*rand(1,3), 30*rand(1,3))
  @test rxdetect(x, 0.8) == [false, false, false, false, false, false, false, false, true, true]
  @test hosvdc4detect(x, 4., 2; b=2) == [false, false, false, false, false, false, false, false, true, true]
  @test hosvdc4detect(x1, 3.9, 1; b=1) == vcat(fill(false, 20), [true, true])
  x2 = [[0. 0.]; [1. 2.]; [1. 1.]]
  @test hosvdc4detect(x2, .1, 2; b=2) == [false, false, false]
  @test rxdetect(x1, 0.9) == vcat(fill(false, 20), [true, true])
  ls = fill(true, 10)
  ls1 = [true, true, true, true, true, true, true, true, false, false]
  c = cumulants(x, 4)
  @test hosvdstep(x, ls, 4., 2, c[4])[1] == ls1
  @test hosvdstep(x, ls, 3., 1, c[4])[2] ≈ 1.15 atol= 0.1
end
