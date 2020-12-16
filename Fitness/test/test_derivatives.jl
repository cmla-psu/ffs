using LinearAlgebra
using Test
using Fitness
import Zygote
#import ForwardDiff  # seems to be buggy, returns non-symmetric gradients

@testset "gradient and hessian" begin
    tolerance = 1e-7
    function obj(cov)
        log(sum(exp.(diag(Brand' * inv(cov) * Brand) * t1)))/t1 +
        log(sum(exp.(diag(Lrand * cov * Lrand') ./ c * t2)))/t2
    end
    function directed(s)
        newpoint = cov + s * direction
        log(sum(exp.(diag(Lrand * newpoint * Lrand') ./ c * t2)))/t2
    end
    c = [1., 1.]
    cov = [1. 0; 0 1]
    t1 = 1.
    t2 = 1.
    Brand = [1 0; 0 1]
    B = Fitness.FFSDenseMatrix(Brand)
    Lrand = [1 0; 0 1]
    L = Fitness.FFSDenseMatrix(Lrand)
    (_, lower0) = Fitness.check_pos_def(cov)
    S0 = Fitness.Sigma(cov, lower0)
    #g = ForwardDiff.gradient(obj, cov0)
    g = Zygote.gradient(obj, cov)[1]
    params = Fitness.FFSParams(B, L, c)
    g_ffs = Fitness.ffs_gradient(params, S0, tb=t1, tl=t2)
    @test g ≈ g_ffs.gradient atol=tolerance

    direction = [0. 0.; 0. 0.] + I
    # compute cov * Hessian * direction (i.e. dot product between cov and (Hessian product with direction))
    cov_hess_dir = Zygote.hessian(((s,),) -> directed(s), [0.])[1]
    hprod = Fitness.hess_times_direction(g_ffs, direction, params, S0, tb=t1, tl=t2)
    println()
    display(hprod)
    println()
    @test cov_hess_dir ≈ dot(cov, hprod)

    ###########
    # Second Gradient/Hessian combo
    ##########
    d1 = 8
    d2 = 4
    d3 = 5
    Brand = rand(d2,d3) # B
    B = FFSDenseMatrix(Brand)
    cov = zeros(d2,d2) + I
    Lrand = rand(d1,d2)
    L = FFSDenseMatrix(Lrand)
    c = collect(1.0:d1)
    t1 = 2.1
    t2 = 3.3
    (_, lower1) = Fitness.check_pos_def(cov)
    S1 = Fitness.Sigma(cov, lower1)


    #g = ForwardDiff.gradient(obj, cov1)
    g = Zygote.gradient(obj, cov)[1]
    g_ffs = Fitness.ffs_gradient(Fitness.FFSParams(B, L, c), S1, tb=t1, tl=t2)
    #println()
    #display(g)
    #println()
    #println()
    #display(g_ffs.gradient)
    #println()
    @test g ≈ g_ffs.gradient atol=tolerance

    ##########################
    # Third gradient Hessian test
    #########################

    tmp = rand(d2,d2)
    cov = tmp' * tmp + 0.01 * I
    c = ones(d1)
    (_, lower2) = Fitness.check_pos_def(cov)
    S2 = Fitness.Sigma(cov, lower2)
    #g2 = ForwardDiff.gradient(obj, cov2)
    g2 = Zygote.gradient(obj, cov)[1]
    g2_ffs = Fitness.ffs_gradient(Fitness.FFSParams(B, L, c), S2, tb=t1, tl=t2)
    @test g2 ≈ g2_ffs.gradient atol=tolerance
end
