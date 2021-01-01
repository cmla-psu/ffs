using Fitness
using LinearAlgebra
using Profile
using Test


function make_problem(d1,d2,d3)
    Brand = rand(d2,d3)
    B = FFSDenseMatrix(Brand)
    Lrand = rand(d1,d2)
    L = FFSDenseMatrix(Lrand)
    c = rand(d1) .+ 1
    ffs_optimize(B=B, L=L, c=c)
    Profile.clear()
    Profile.init(n = 10^9, delay = 0.01)
    @profile ffs_optimize(B=B, L=L, c=c)
    Profile.print(combine=true,sortedby=:count)
end

@testset "profile" begin
    make_problem(12, 10, 13)
end
