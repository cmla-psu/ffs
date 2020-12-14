using LinearAlgebra
using Test
using Fitness
using Zygote

@testset "gradient_and_hessian" begin
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
    c = [1., 2., 3., 4., 5., 6., 7., 8.]
    function(covariance)

    end
    c = ones(8)
end
