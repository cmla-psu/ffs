using LinearAlgebra
using Test
using Fitness
using Zygote

@testset "gradient and hessian" begin
    tolerance = 1e-7
    function obj(cov)
        log(sum(exp.(diag(Brand' * inv(cov) * Brand) .* t1)))/t1 +
        log(sum(exp.(diag(Lrand * cov * Lrand') ./ c .* t2)))/t2
    end

    c = [1., 1.]
    cov0 = [1. 0; 0 1]
    t1 = 1.
    t2 = 1.
    Brand = [1 0; 0 1]
    B = Fitness.FFSDenseMatrix(Brand)
    Lrand = [1 0; 0 1]
    L = Fitness.FFSDenseMatrix(Lrand)
    (_, lower0) = Fitness.check_pos_def(cov0)
    S0 = Fitness.Sigma(cov0, lower0)
    g = gradient(obj, cov0)[1]
    g_ffs = Fitness.ffs_gradient(Fitness.FFSParams(B, L, c), S0, tb=t1, tl=t2)
    @test g ≈ g_ffs atol=tolerance

    d1 = 8
    d2 = 4
    d3 = 5
    Brand = rand(d2,d3) # B
    B = FFSDenseMatrix(Brand)
    cov1 = zeros(d2,d2)
    for i in 1:size(cov1, 1)
        cov1[i,i]=1.0
    end
    Lrand = rand(d1,d2)
    L = FFSDenseMatrix(Lrand)
    c = collect(1.0:d1)
    t1 = 2.1
    t2 = 3.3
    (_, lower1) = Fitness.check_pos_def(cov1)
    S1 = Fitness.Sigma(cov1, lower1)


    g = gradient(obj, cov1)[1]
    g_ffs = Fitness.ffs_gradient(Fitness.FFSParams(B, L, c), S1, tb=t1, tl=t2)
    @test g ≈ g_ffs atol=tolerance

    tmp = rand(d2,d2)
    cov2 = tmp' * tmp
    c = ones(d1)
    (_, lower2) = Fitness.check_pos_def(cov2)
    S2 = Fitness.Sigma(cov2, lower2)
    g2 = gradient(obj, cov2)[1]
    g2_ffs = Fitness.ffs_gradient(Fitness.FFSParams(B, L, c), S2, tb=t1, tl=t2)
    @test g2 ≈ g2_ffs atol=tolerance
end
