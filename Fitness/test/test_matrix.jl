using LinearAlgebra
using Fitness
using Test

""" Tests that f behaves like the intended matrix `realized` """
function check_product(f::Fitness.FFSMatrix, realized::Matrix{Float64}, tolerance=1e-9)
    x = rand(size(realized)...)
    @test f * x ≈ realized * x atol=tolerance
    @test x * f ≈ realized * x atol=tolerance
end

function check_size(f::Fitness.FFSMatrix, realized::Matrix{Float64})
    @test size(f) == size(realized)
    @test size(f,1) == size(realized, 1)
    @test size(f,2) == size(realized, 2)
end

function check_weightedLTL(f::Fitness.FFSMatrix, realized::Matrix{Float64}, tolerance=1e-9)
   weights = rand(size(realized,1))
   @test Fitness.weightedLTL(f, weights) ≈ realized' * diagm(weights) * realized
end

""" Checks the diagprod method when S is a Matrix, not FFSMatrix"""
function check_diagprod_matrixS(f::Fitness.FFSMatrix, realized::Matrix{Float64}, tolerance=1e-9)
    k = size(realized,2)
    tmp = rand(k,k)
    S = tmp' * tmp + 0.1 * I

    c1 = rand(size(realized,1)) .+ 1
    @test Fitness.diag_prod(f, S, c1) ≈ diag(realized * S * realized') ./c1 atol=tolerance

    c2 = rand() .+ 1
    @test Fitness.diag_prod(f, S, c2) ≈ diag(realized * S * realized') ./c2 atol=tolerance
end

function check_optimize_as_L(f::Fitness.FFSMatrix, realized::Matrix{Float64}, tolerance=1e-9)
    d1, d2 = size(realized)
    d3 = d2 + 4
    Brand = rand(d2,d3)
    B = FFSDenseMatrix(Brand)
    L = f
    L2 = FFSDenseMatrix(realized)
    c = rand(d1) .+ 1
    time1 = @elapsed cov1 = ffs_optimize(B=B, L=L, c=c)
    time2 = @elapsed cov2 = ffs_optimize(B=B, L=L2, c=c)
    println("L Optimization for ", typeof(f), " for problem size ", (d1,d2,d3), " is ", time1)
    println("L Optimization for dense matrix for problem size ", (d1,d2,d3), " is ", time2)
    @test cov1 ≈ cov2 atol=tolerance
end

function check_optimize_as_B(f::Fitness.FFSMatrix, realized::Matrix{Float64}, tolerance=1e-9)
    d2, d3 = size(realized)
    d1 = d2 + 4
    B = f
    B2 = FFSDenseMatrix(realized)
    Lrand = rand(d1, d2)
    L = FFSDenseMatrix(Lrand)
    c = rand(d1) .+ 1
    time1 = @elapsed cov1 = ffs_optimize(B=B, L=L, c=c)
    time2 = @elapsed cov2 = ffs_optimize(B=B2, L=L, c=c)
    println("B Optimization for ", typeof(f), " for problem size ", (d1,d2,d3), " is ", time1)
    println("B Optimization for dense matrix for problem size ", (d1,d2,d3), " is ", time2)
    @test cov1 ≈ cov2 atol=tolerance
end


@testset "FFSId" begin
    for k in 10:10
        f = Fitness.FFSId(k)
        realized = collect(I(k)) * 1.0
        check_product(f, realized)
        check_size(f, realized)
        check_weightedLTL(f, realized)
        check_diagprod_matrixS(f, realized)
        check_optimize_as_L(f, realized)
        check_optimize_as_B(f, realized)
    end
end
