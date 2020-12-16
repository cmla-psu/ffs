import LinearAlgebra
using Test
using Fitness

const ROW=1
const COL=1

@testset "test_fast_helpers_functions" begin
    tolerance = 1e-6
    B = [1. 1. 1. 1.; 1. 1. 1. 0.; 1. 1. 0. 0.; 1. 0. 0. 0.]
    BT = FFSDenseMatrix(B)
    L = FFSDenseMatrix([1. 2. 3. 4.; 4. 3. 2. 1.; 1. 0. 0. 1.; 1. 2. 1. 2.; 1. 1. 1. 1.])
    cov = [2. 1. 1. 1.; 1. 2. 1. 1.; 1. 1. 2. 1.; 1. 1. 1. 2.]
    (result, lower) = Fitness.check_pos_def(cov)
    S = Fitness.Sigma(cov, lower)
    c = [1.0, 1.5, 2.0, 2.5, 3.0]

    @test Fitness.diag_prod(L, S, c) ≈ LinearAlgebra.diag(L.mat * S.cov * L.mat')./c atol=tolerance
    @test Fitness.diag_prod(L, cov, c) ≈ Fitness.diag_prod(L, S, c)
    @test Fitness.diag_inv_prod(BT, S) ≈ LinearAlgebra.diag(B' * inv(S.cov) * B) atol=tolerance

    anotherL = FFSDenseMatrix(rand(5,4))
    tmp = rand(4,4)
    anotherCov = tmp' * tmp + 0.01 * LinearAlgebra.I
    anotherC = rand(5)
    (_, anotherLower) = Fitness.check_pos_def(anotherCov)
    anotherS = Fitness.Sigma(anotherCov, anotherLower)
    anotherB = rand(4,4)
    anotherBT = FFSDenseMatrix(anotherB)
    @test Fitness.diag_prod(anotherL, anotherS, anotherC) ≈ LinearAlgebra.diag(anotherL.mat * anotherS.cov * anotherL.mat')./anotherC atol=tolerance
    @test Fitness.diag_prod(anotherL, anotherCov, anotherC) ≈ Fitness.diag_prod(anotherL, anotherS, anotherC)
    @test Fitness.diag_inv_prod(anotherBT, anotherS) ≈ LinearAlgebra.diag(anotherB' * inv(anotherS.cov) * anotherB) atol=tolerance


    B2 = rand(4,5)
    tmp2 = rand(4,4)
    cov2 = tmp2' * tmp2
    invcov2 = inv(cov2)
    (_, lower2) = Fitness.check_pos_def(cov2)
    S2 = Fitness.Sigma(cov2, lower2)
    L2 = rand(8,4)
    weightB = rand(5)
    weightL = rand(8)
    #sum_i weights[i] S^{-1} b_i bt_i' S^{-1} where the b_i are columns of B
    helperB = zeros(4,4)
    for i in 1:5
        helperB += invcov2 * B2[:,i] * B2[:,i]' * invcov2 * weightB[i]
    end
    @test Fitness.weighted_grad_helper_B(FFSDenseMatrix(B2), S2, weightB) ≈ helperB

    helperL = zeros(4,4)
    for i in 1:8
        helperL += L2[i,:] * L2[i,:]' * weightL[i]
    end

    @test Fitness.weightedLTL(FFSDenseMatrix(L2), weightL) ≈ helperL
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

    x1 = [5.0]
    @test Fitness.softmax(x1, 2.5) ≈ x1[1] atol=tolerance
    @test Fitness.softmax(x1 .+ 1.5, 3.0) ≈ x1[1] + 1.5 atol=tolerance
    x2 = rand(10)
    t = 4.3
    @test Fitness.softmax(x2, t) ≈ log(sum(exp.(x2 * t)))/t

    M = [1. 2. 3. 4.; 5. 6. 7. 8.]
    @test Fitness.col_sq_norm(M) ≈ [26., 40., 58., 80.] atol=tolerance
    @test Fitness.row_sq_norm(M) ≈ [30., 174.] atol=tolerance

end



@testset "optimization" begin
   # TODO test objective function
end

@testset "problem conversion" begin
    tolerance = 1e-7
    Brand = rand(4,5) # B
    BT = FFSDenseMatrix(Brand)
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
    BT1 = FFSDenseMatrix(Brand)
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
