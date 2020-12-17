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
        obj(cov + s * direction)
    end
    function partsB(x,y,z) # used to test directional derivative of gradient
                         # of the privacy part,  separating out the 3 components of the chain rule
        deriv = zeros(size(cov))
        denom = 0.
        for i in 1:size(Brand,2) #for each colum
            expterm = exp(t1 * Brand[:,i]' * inv(cov + x * direction) * Brand[:, i])
            denom += exp(t1 * Brand[:,i]' * inv(cov + y * direction) * Brand[:, i])
            deriv -= expterm * inv(cov + z * direction) * Brand[:,i] * Brand[:,i]' * inv(cov + z * direction)'
        end
        deriv/denom
    end
    function partsL(x,y)
        deriv = zeros(size(cov))
        denom = 0.
        for i in 1:size(Lrand, 1) #rows
            expterm = exp(t2/c[i] * Lrand[i,:]' * (cov + x * direction) * Lrand[i,:])
            denom += exp(t2/c[i] * Lrand[i,:]' * (cov + y * direction) * Lrand[i,:])
            deriv -= expterm *  Lrand[i,:] * Lrand[i,:]'/c[i]
        end
        deriv/denom
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
    @test cov_hess_dir ≈ dot(cov, hprod) atol=tolerance
    step = 0.00001
    println("\n gradient of L")
    display(partsL(0., 0.))
    println("\n estimated L part 1")
    display((partsL(step, 0.) - partsL(0., 0.)) / step)
    println("\n estimated L part 2")
    display((partsL(0., step) - partsL(0., 0.)) / step)

    println("\n gradient of B")
    display(partsB(0., 0., 0.))
    println("\n estimated B part 1")
    display((partsB(step, 0., 0.) - partsB(0., 0., 0.)) / step)
    println("\n estimated B part 2")
    display((partsB(0., step, 0.) - partsB(0., 0., 0.)) / step)
    println("\n estimated B part 3")
    display((partsB(0., 0., step) - partsB(0., 0., 0.)) / step)
    println()

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
    params = Fitness.FFSParams(B, L, c)
    g_ffs = Fitness.ffs_gradient(params, S1, tb=t1, tl=t2)
    @test g ≈ g_ffs.gradient atol=tolerance

    direction = rand(d2, d2)
    direction = (direction + direction')/2 #symmetric direction
    # compute cov * Hessian * direction (i.e. dot product between cov and (Hessian product with direction))
    cov_hess_dir = Zygote.hessian(((s,),) -> directed(s), [0.])[1]
    hprod = Fitness.hess_times_direction(g_ffs, direction, params, S1, tb=t1, tl=t2)
    @test cov_hess_dir ≈ dot(cov, hprod) atol=tolerance

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
    params = Fitness.FFSParams(B, L, c)
    g2_ffs = Fitness.ffs_gradient(params, S2, tb=t1, tl=t2)
    @test g2 ≈ g2_ffs.gradient atol=tolerance

    direction = rand(d2, d2)
    direction = (direction + direction')/2 #symmetric direction
    # compute cov * Hessian * direction (i.e. dot product between cov and (Hessian product with direction))
    cov_hess_dir = Zygote.hessian(((s,),) -> directed(s), [0.])[1]
    hprod = Fitness.hess_times_direction(g_ffs, direction, params, S2, tb=t1, tl=t2)
    @test cov_hess_dir ≈ dot(cov, hprod) atol=tolerance

end
