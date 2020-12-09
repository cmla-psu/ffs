import LinearAlgebra
using Test
using Fitness

@testset "test_fast_helpers" begin
  B = [1. 1. 1. 1.; 1. 1. 1. 0.; 1. 1. 0. 0.; 1. 0. 0. 0.]
  BT = FFSDenseMatrix(B')
  L = FFSDenseMatrix([1. 2. 3. 4.; 4. 3. 2. 1.; 1. 0. 0. 1.; 1. 2. 1. 2.; 1. 1. 1. 1.])
  cov = [2. 1. 1. 1.; 1. 2. 1. 1.; 1. 1. 2. 1.; 1. 1. 1. 2.]
  (result, lower) = Fitness.check_pos_def(cov)
  S = Fitness.Sigma(cov, lower)
  c = [1.0, 1.5, 2.0, 2.5, 3.0]

  @test diag_prod(L, S, c) ≈ LinearAlgebra.diag(L.mat * S.cov * L.mat')./c atol=0.0001

end
