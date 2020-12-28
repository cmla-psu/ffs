########################################
# Subtypes of FFSMatrix and
# overloaded multiplication operations
# and overloaded weightedLTL
########################################

# What to overload:
#
# 1. Subtype FFSMatrix
# 2. Overload Base.size(ffsmat) and Bae.size(ffsmat, dim)
# 3. Overload Base.:*(ffsmat, Matrix) and Base.*:(Matrix, ffsmat), left and right multiplication by matrix
# 4. Overload weightedLTL(L::FFSMatrix, weights::Vector{Float64})::Matrix{Float64}
#      That computes sum_i weights[i] v_i' v_i where v_i is a row of L. This is
#      the same as L' diag(weights) L
# 5. Overload diag_prod(L::FFSMatrix, S::Matrix{Float64}, c::Union{Float64, Vector{Float64}})::Vector{Float64}
#      which returns the (diagonal of L S L') divided pointwise by c. S is symmetric positive definite
# 6. Optional: overload other functions such as
#      a) diag_prod where S has type Sigma (so cholesky decomposition is available)
#      b) diag_inv_prod
#      c) weighted_grad_helper_B
#      d) weighted_hess_helper_B

######################################
# Dense matrices
######################################
"""
The default structure for a dense matrix
"""
struct FFSDenseMatrix <: FFSMatrix
   mat::Matrix{Float64}
end
Base.size(x::FFSDenseMatrix) = size(x.mat)
Base.size(x::FFSDenseMatrix, y::Int64) = size(x.mat, y)
Base.:*(M::FFSDenseMatrix, A::Matrix{Float64}) = M.mat * A
Base.:*(A::Matrix{Float64}, M::FFSDenseMatrix) = A * M.mat
"""
For each row v_i of L, computes ``sum_i weights[i] v_i' v_i``
which is the same as ``L' diag(weights) L``.
"""
weightedLTL(L::FFSDenseMatrix, weights::Vector{Float64})::Matrix{Float64} = L.mat' * diagm(weights) * L.mat #(weights' .* L.mat') * L.mat

function diag_prod(L::FFSDenseMatrix,
                   S::Matrix{Float64},
                   c::Union{Float64, Vector{Float64}})::Vector{Float64}
    result = sum((L.mat * S) .* L.mat, dims=COL) ./ c
    reshape(result, :) # from m x 1 matrix to col vecctor
end

#####################################
# Identity matrix for FFS.
####################################
struct FFSId <: FFSMatrix
    size::Int64
end
Base.size(x::FFSId) = x.size
Base.size(x::FFSId, y::Int64) = size(I(x.size),y)
Base.:*(M::FFSId, A::Matrix{Float64}) = M.mat * A
Base.:*(A::Matrix{Float64}, M::FFSId) = A * M.mat
weightedLTL(L::FFSId, weights::Vector{Float64})::Matrix{Float64} = diagm(weights)
function diag_prod(L::FFSId,
                   S::Matrix{Float64},
                   c::Union{Float64, Vector{Float64}})::Vector{Float64}
    diag(S) ./ c
end


#TODO test
#TODO prefix
#TODO marginals
#TODO range queries
#TODO unions
#TODO sparse matrix
