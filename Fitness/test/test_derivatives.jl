using LinearAlgebra
using Test
using Fitness
import Zygote
#import ForwardDiff  # seems to be buggy, returns non-symmetric gradients

@testset "gradient and hessian" begin
    tolerance = 1e-7
    fd_tolerance = 1e-3 #tolerance for finite differences
    fd_step = 0.00001 # step size for finite differences
    objB(cov) = log(sum(exp.(diag(Brand' * inv(cov) * Brand) * t1)))/t1
    objL(cov) = log(sum(exp.(diag(Lrand * cov * Lrand') ./ c * t2)))/t2
    obj(cov) = objB(cov) + objL(cov) #objective function
    function objBstable(cov)
        elements = diag(Brand' * inv(cov) * Brand)
        maxel = maximum(elements)
        maxel + log(sum(exp.((elements .- maxel) * t1)))/t1
    end
    function objLstable(cov)
        elements = diag(Lrand * cov * Lrand') ./ c
        maxel = maximum(elements)
        maxel + log(sum(exp.((elements .- maxel) * t2)))/t2
    end
    objstable(cov) = objBstable(cov) + objLstable(cov)

    function directed(s) # 2nd derivatives with respect to s evaluated at s=0 should return direction' hessian direction
        obj(cov + s * direction)
    end
    directed_stable(s) = objstable(cov + s*direction)
    function partsB(x,y,z) # used to test directional derivative of gradient
                         # of the privacy part,  separating out the 3 components of the chain rule
        deriv = zeros(size(cov))
        denom = 0.
        pieces_x = zeros(size(Brand, 2))
        pieces_y = zeros(size(Brand, 2))
        for i in 1:size(Brand, 2)
            pieces_x[i]=t1 * Brand[:,i]' * inv(cov + x * direction) * Brand[:, i]
            pieces_y[i]=t1 * Brand[:,i]' * inv(cov + y * direction) * Brand[:, i]
        end
        max_x = maximum(pieces_x)
        max_y = maximum(pieces_y)
        themax = max(max_x, max_y)
        for i in 1:size(Brand,2) #for each colum
            expterm = exp(pieces_x[i]-themax)
            denom += exp(pieces_y[i]-themax)
            deriv -= expterm * inv(cov + z * direction) * Brand[:,i] * Brand[:,i]' * inv(cov + z * direction)'
        end
        deriv/denom
    end
    function partsL(x,y)
        deriv = zeros(size(cov))
        denom = 0.
        pieces_x = zeros(size(Lrand, 1))
        pieces_y = zeros(size(Lrand, 1))
        for i in 1:size(Lrand, 1)
            pieces_x[i] = t2/c[i] * Lrand[i,:]' * (cov + x * direction) * Lrand[i,:]
            pieces_y[i] = t2/c[i] * Lrand[i,:]' * (cov + y * direction) * Lrand[i,:]
        end
        max_x = maximum(pieces_x)
        max_y = maximum(pieces_y)
        themax = max(max_x, max_y)
        for i in 1:size(Lrand, 1) #rows
            expterm = exp(pieces_x[i] - themax)
            denom += exp(pieces_y[i] - themax)
            deriv += expterm *  Lrand[i,:] * Lrand[i,:]'/c[i]
        end
        deriv/denom
    end
    function print_debug(message)
        println("\n ##################",message,"##############")
        step = fd_step
        println("\n gradient of L")
        display(partsL(0., 0.))
        println("\n from Zygote")
        display(Zygote.gradient(objL, cov)[1])
        println("\n estimated L part 1")
        display((partsL(step, 0.) - partsL(0., 0.)) / step)
        println("\n estimated L part 2")
        display((partsL(0., step) - partsL(0., 0.)) / step)
        println("\n gradient of B")
        display(partsB(0., 0., 0.))
        println("\n from Zygote")
        display(Zygote.gradient(objB, cov)[1])
        println("\n estimated B part 1")
        display((partsB(step, 0., 0.) - partsB(0., 0., 0.)) / step)
        println("\n estimated B part 2")
        display((partsB(0., step, 0.) - partsB(0., 0., 0.)) / step)
        println("\n estimated B part 3")
        display((partsB(0., 0., step) - partsB(0., 0., 0.)) / step)
        println()
        cov_hess_dir = Zygote.hessian(((s,),) -> directed(s), [0.])[1] # zygote hessian not always reliable for this problem
        println("dir' Hess dir Estimate from zygote: ",cov_hess_dir)
        println("dir' Hess dir Finite differences: ", (directed(2*step) - 2*directed(step) + directed(0.))/step^2)
    end
    function fd_hess_and_prod(step)
        base_l = partsL(0., 0.)
        base_b = partsB(0., 0., 0.)
        fd_prod = (directed_stable(2*step) - 2*directed_stable(step) + directed_stable(0.))/step^2
        fd_hess1 = (partsL(step, 0.) -base_l) / step +
                   (partsL(0., step) - base_l) / step +
                   (partsB(step, 0., 0.) - base_b) / step +
                   (partsB(0., step, 0.) - base_b) / step +
                   (partsB(0., 0., step) - base_b) / step
        fd_hess2 = (partsL(step, step) - base_l) / step +
                   (partsB(step, step, step) - base_b) / step
        (fd_prod, fd_hess1, fd_hess2)
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

    direction = [1. 1.; 1. 1.] + I

    # compute direction' * Hessian * direction (i.e. dot product between cov and (Hessian product with direction))
    #cov_hess_dir = Zygote.hessian(((s,),) -> directed(s), [0.])[1] # zygote diverges from finite difference test
    #@test cov_hess_dir ≈ dot(direction, hprod) atol=tolerance

    hprod = Fitness.hess_times_direction(g_ffs, direction, params, S0, tb=t1, tl=t2)
    (fd_prod, fd_hess1, fd_hess2) = fd_hess_and_prod(fd_step)
    #print_debug("test1")
    @test fd_prod ≈ dot(direction, hprod) rtol=fd_tolerance
    @test fd_hess1 ≈ hprod rtol=fd_tolerance
    @test fd_hess2 ≈ hprod rtol=fd_tolerance

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
    # compute direction' * Hessian * direction (i.e. dot product between cov and (Hessian product with direction))
    #cov_hess_dir = Zygote.hessian(((s,),) -> directed(s), [0.])[1] #zygote unreliable for this computation
    #@test cov_hess_dir ≈ dot(direction, hprod) atol=tolerance
    hprod = Fitness.hess_times_direction(g_ffs, direction, params, S1, tb=t1, tl=t2)
    (fd_prod, fd_hess1, fd_hess2) = fd_hess_and_prod(fd_step)
    #print_debug("test2")
    @test fd_prod ≈ dot(direction, hprod) rtol=fd_tolerance
    @test fd_hess1 ≈ hprod rtol=fd_tolerance
    @test fd_hess2 ≈ hprod rtol=fd_tolerance

    ##########################
    # Third gradient Hessian test
    #########################

    tmp = rand(d2,d2)
    cov = tmp' * tmp + 0.1 * I
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
    # compute direction * Hessian * direction (i.e. dot product between cov and (Hessian product with direction))
    #cov_hess_dir = Zygote.hessian(((s,),) -> directed(s), [0.])[1] #zygote result unreliable
    #@test cov_hess_dir ≈ dot(direction, hprod) atol=tolerance
    hprod = Fitness.hess_times_direction(g2_ffs, direction, params, S2, tb=t1, tl=t2)
    (fd_prod, fd_hess1, fd_hess2) = fd_hess_and_prod(fd_step)
    #print_debug("test3")
    @test fd_prod ≈ dot(direction, hprod) rtol=fd_tolerance
    @test fd_hess1 ≈ hprod rtol=fd_tolerance
    @test fd_hess2 ≈ hprod rtol=fd_tolerance

end
