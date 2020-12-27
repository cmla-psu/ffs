using LinearAlgebra
using Test
using Fitness
import Random

@testset "line search" begin
    rng = Random.MersenneTwister(4)
    tolerance = 1e-7
    d1 = 8
    d2 = 4 #must be even
    d3 = 5
    Brand = rand(rng, d2,d3) # B
    B = FFSDenseMatrix(Brand)
    tmp = rand(rng, d2, d2)
    cov = tmp' * tmp + 0.01 * I
    Lrand = rand(rng, d1,d2)
    L = FFSDenseMatrix(Lrand)
    c = collect(1.0:d1)
    tb = 2.1
    tl = 3.3
    (_, lower) = Fitness.check_pos_def(cov)
    S = Fitness.Sigma(cov, lower)
    params = Fitness.FFSParams(B, L, c)
    g_ffs = Fitness.ffs_gradient(params, S, tb=tb, tl=tl)
    direction = diagm(vcat(ones(convert(Int, d2/2)), -ones(convert(Int, d2/2)))) .+ 0.01
    if dot(direction, g_ffs.gradient) > 0
        direction = -direction
    end
    direction = direction * 10 # for checking if result is in the positive definite range
    new_sigma = Fitness.line_search(params, S, direction, g_ffs.gradient, tb=tb, tl=tl)
    new_cov = new_sigma.cov
    (is_pos_def, _) = Fitness.check_pos_def(new_cov)
    @test is_pos_def
    the_scale = (new_cov - cov) ./ direction
    @test minimum(the_scale) ≈ maximum(the_scale) atol=tolerance
    @test minimum(the_scale) >= 0
    old_obj = Fitness.objective(params, S, tb=tb, tl=tl)
    new_obj = Fitness.objective(params, new_sigma, tb=tb, tl=tl)
    @test new_obj < old_obj
end

@testset "conjugate gradient" begin
    rng = Random.MersenneTwister(3)
    tolerance = 1e-7
    rtol = 0.01
    # L is d1 x d2
    # B is d2 x d3
    # Sigma is d2 x d2
    d1 = 8
    d2 = 4 #must be even
    d3 = 5
    Brand = rand(rng, d2,d3) # B
    B = FFSDenseMatrix(Brand)
    tmp = rand(rng, d2, d2)
    cov = tmp' * tmp + 0.01 * I
    Lrand = rand(rng, d1,d2)
    L = FFSDenseMatrix(Lrand)
    c = collect(1.0:d1)
    tb = 2.1
    tl = 3.3
    (_, lower) = Fitness.check_pos_def(cov)
    S = Fitness.Sigma(cov, lower)
    params = Fitness.FFSParams(B, L, c)
    g_ffs = Fitness.ffs_gradient(params, S, tb=tb, tl=tl)
    direction = Fitness.conjugate_gradient(g_ffs, params, S, d2^4, tb=tb, tl=tl)
    @test Fitness.hess_times_direction(g_ffs, direction, params, S, tb=tb, tl=tl) ≈ -g_ffs.gradient atol=tolerance rtol=rtol
end

@testset "initialization" begin
    tolerance = 1e-7
    # L is d1 x d2
    # B is d2 x d3
    # Sigma is d2 x d2
    d1 = 8
    d2 = 4 #must be even
    d3 = 5
    Brand = rand(d2,d3) # B
    B = FFSDenseMatrix(Brand)
    Lrand = rand(d1,d2)
    L = FFSDenseMatrix(Lrand)
    c = collect(1.0:d1)
    tb = 2.1
    tl = 3.3
    params = Fitness.FFSParams(B, L, c)

    @test_throws FFSException Fitness.initialize(params, Q=rand(d2, d3))
    @test_throws FFSException Fitness.initialize(params, Q=I(d2 + 1))
    @test_throws FFSException Fitness.initialize(params, Q=rand(d2, d2))
    @test_throws FFSException Fitness.initialize(params, Q=-1.0 * I(d2))
    tmp = randn(d2, d2)
    fudge = rand()
    S = Fitness.initialize(params, Q=tmp' * tmp + 0.01 * I, fudge=fudge)
    @test maximum(diag(Lrand * S.cov * Lrand') ./ c) ≈ fudge atol=tolerance
end
