import LinearAlgebra
using Test
using Fitness

@testset "test_fast_helpers" begin
  tolerance = 1e-6
  B = [1. 1. 1. 1.; 1. 1. 1. 0.; 1. 1. 0. 0.; 1. 0. 0. 0.]
  BT = FFSDenseMatrix(B')
  L = FFSDenseMatrix([1. 2. 3. 4.; 4. 3. 2. 1.; 1. 0. 0. 1.; 1. 2. 1. 2.; 1. 1. 1. 1.])
  cov = [2. 1. 1. 1.; 1. 2. 1. 1.; 1. 1. 2. 1.; 1. 1. 1. 2.]
  (result, lower) = Fitness.check_pos_def(cov)
  S = Fitness.Sigma(cov, lower)
  c = [1.0, 1.5, 2.0, 2.5, 3.0]

  @test diag_prod(L, S, c) ≈ LinearAlgebra.diag(L.mat * S.cov * L.mat')./c atol=tolerance
  @test diag_inv_prod(BT, S) ≈ LinearAlgebra.diag(BT.mat * inv(S.cov) * BT.mat') atol=tolerance
end

@testset "test_helper_functions" begin
    tolerance = 1e-7
    R = rand(4,4)
    S = R' * R
    OtherS = [1. 2.; 2. 1.]

    @test Fitness.symmetrize(S) ≈ S atol=tolerance
    @test Fitness.symmetrize(R) ≈ (R + R')/2 atol=tolerance

    (result1, lower1) = Fitness.check_pos_def(S)
    @test result1 # should be true
    @test lower1 * lower1' ≈ S atol=tolerance

    (result2, lower2) = Fitness.check_pos_def(-S)
    @test !result2

    (result3, lower3) = Fitness.check_pos_def(OtherS)
    @test !result3
end


@test "gradient_and_hessian" begin

end

@test "optimization" begin

end

@test "measures" begin
    
end
