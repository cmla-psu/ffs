import LinearAlgebra
using Test
using Fitness

const ROW=1
const COL=1

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


@testset "gradient_and_hessian" begin
    return
    using Zygote
end

@testset "optimization" begin
   # TODO test objective function
end

@testset "problem conversion" begin
    tolerance = 1e-7
    Brand = rand(4,5) # B
    BT = FFSDenseMatrix(Brand')
    cov1 = zeros(4,4)
    for i in 1:size(cov1, 1)
        cov1[i,i]=1.0
    end
    tmp = rand(4,4)
    cov2 = tmp' * tmp
    Lrand = rand(8,4)
    L = FFSDenseMatrix(Lrand)
    c1 = [1., 2., 3., 4., 5., 6., 7., 8.]
    c2 = ones(8)

    new_cov1 = enforce_pcost(BT, cov1, 0.1)
    @test privacy_cost(BT, new_cov1) ≈ 0.1
    new_cov1_again = enforce_pcost(BT, 2*cov1, 0.1)
    @test privacy_cost(BT, new_cov1_again) ≈ 0.1
    new_cov2 = enforce_pcost(BT, cov2, 0.1)
    @test privacy_cost(BT, new_cov2) ≈ 0.1
    new_cov2_again = enforce_pcost(BT, 2*cov2, 0.1)
    @test privacy_cost(BT, new_cov2_again) ≈ 0.1

    ffs_cov1_1 = enforce_ffs(L, cov1, c1)
    @test ffs_overrun(L, ffs_cov1_1, c1) ≈ 1.0
    ffs_cov1_1_again = enforce_ffs(L, 2*cov1, c1)
    @test ffs_overrun(L, ffs_cov1_1_again, c1) ≈ 1.0
    ffs_cov1_2 = enforce_ffs(L, cov1, c2)
    @test ffs_overrun(L, ffs_cov1_2, c2) ≈ 1.0
    ffs_cov1_2_again = enforce_ffs(L, 2*cov1, c2)
    @test ffs_overrun(L, ffs_cov1_2_again, c2) ≈ 1.0

    ffs_cov2_1 = enforce_ffs(L, cov2, c1)
    @test ffs_overrun(L, ffs_cov2_1, c1) ≈ 1.0
    ffs_cov2_1_again = enforce_ffs(L, 2*cov2, c1)
    @test ffs_overrun(L, ffs_cov2_1_again, c1) ≈ 1.0
    ffs_cov2_2 = enforce_ffs(L, cov2, c2)
    @test ffs_overrun(L, ffs_cov2_2, c2) ≈ 1.0
    ffs_cov2_2_again = enforce_ffs(L, 2*cov2, c2)
    @test ffs_overrun(L, ffs_cov2_2_again, c2) ≈ 1.0


end

@testset "measures" begin
    tolerance = 1e-7
    Brand = rand(4,5) # B
    BT1 = FFSDenseMatrix(Brand')
    cov1 = zeros(4,4)
    for i in 1:size(cov1, 1)
        cov1[i,i]=1.0
    end
    tmp = rand(4,4)
    cov2 = tmp' * tmp
    Lrand = rand(8,4)
    L = FFSDenseMatrix(Lrand)
    c1 = [1., 2., 3., 4., 5., 6., 7., 8.]
    c2 = ones(8)

    # if covariance is identity, privacy cost is square root of largest column norm of B
    @test privacy_cost(BT1, cov1) ≈ sqrt(maximum(sum(Brand .* Brand, dims=ROW))) atol=tolerance
    @test privacy_cost(BT1, cov2) ≈ sqrt(maximum(LinearAlgebra.diag(Brand' * inv(cov2) * Brand))) atol=tolerance

    @test l2error_vector(L, cov1) ≈ LinearAlgebra.diag(Lrand * cov1 * Lrand') atol = tolerance
    @test l2error_vector(L, cov2) ≈ LinearAlgebra.diag(Lrand * cov2 * Lrand') atol = tolerance

    @test ffs_overrun(L, cov1, c1) ≈ maximum(LinearAlgebra.diag(Lrand * cov1 * Lrand')./c1) atol = tolerance
    @test ffs_overrun(L, cov1, c2) ≈ maximum(LinearAlgebra.diag(Lrand * cov1 * Lrand')./c2) atol = tolerance
    @test ffs_overrun(L, cov2, c1) ≈ maximum(LinearAlgebra.diag(Lrand * cov2 * Lrand')./c1) atol = tolerance
    @test ffs_overrun(L, cov2, c2) ≈ maximum(LinearAlgebra.diag(Lrand * cov2 * Lrand')./c2) atol = tolerance


end
