module Fitness
using LinearAlgebra
include("FFSMatrix.jl")

const START = 1 #starting index
const ROW = 1 # first dimension is ROW
const COL = 2 # second dimension is COL
############################
##  Types for FFS
###########################

"""
Base type for L and B matrices. Extend this for subtypes that have fast
    maxtrix operations for ``L S L' `` and/or ``B' S^{-1} B``
"""
abstract type FFSMatrix end




"""
Types allowed for the covariance matrix
"""
CovType = Matrix{Float64}


"""
Keeps track of the problem settings, including the basis matrix,
representation matrix, and target cost vector.
"""
struct FFSParams
   B::FFSMatrix # B
   L::FFSMatrix # L
   c::Vector{Float64} # target variances
end



"""
Keeps track of the covariance matrix and its cholesky decomposition.
"""
struct Sigma
   cov::CovType # the covariance matrix
   lower::Matrix{Float64} # the lower triangular matrix from its cholesky decomposition
   lowerinv::Matrix{Float64}
   Sigma(c, l) = new(c, l, l \ I)
end




######################################
# Helper methods with
# potential for speeding up FFS
#
# To speed up your own W=LB
# decomposition, create a subtype
# of FFSMatrix and implement
# a fast version of these functions
# but usually it is faster to write your
# matrix multipliation methods for LX and XB
# where L and B are the representation and basis matrices
######################################

"""
Returns the diagonal of ``L S L'`` divided pointwise by `c`.
For dense matrices, we compute the squared norm of each row of ``L * lower``,
where `lower` is the lower triangular matrix of the cholesky decomposition
of the covariance `S`.
"""
function diag_prod(L::FFSDenseMatrix,
                   S::Sigma,
                   c::Union{Float64, Vector{Float64}})::Vector{Float64}

    result = row_sq_norm(L * S.lower)
    result ./= c
    result
end


"""
Returns the diagonal of ``B' S^{-1} B``.
For dense matrices, we compute the squared norm of each column of ``x``, where ``x`` is obtained
by solving ``lower x = B``,
where ``lower`` is the lower triangular matrix of the cholesky decomposition
of the covariance `S`.
"""
function diag_inv_prod(B::FFSDenseMatrix, S::Sigma)::Vector{Float64}
   col_sq_norm(S.lowerinv * B)
end


"""
For each column b_i of B, computes ``sum_i weights[i] S^{-1} b_i bt_i' S^{-1}``,
which is the same as S^{-1} B diag(weights) B' S^{-1}.
"""
function weighted_grad_helper_B(B::FFSMatrix, S::Sigma, weights::Vector{Float64})
    left = S.lowerinv' * (S.lowerinv * B)
    (weights' .* left ) * left'
end



#############################
# Helper functions
############################

"""
Computes the squared L2 norms of each row of `mat`
"""
function row_sq_norm(mat::Matrix{Float64})::Vector{Float64}
    result = zeros(Float64, size(mat, ROW)) # same dim as num rows
    for col in START:size(mat, COL) # matrix in column major layout
        for row in START:size(mat, ROW)
            result[row] += mat[row, col]^2
        end
    end
    result
end


"""
Computes the squared L2 norms of each column of `mat`
"""
function col_sq_norm(mat::Matrix{Float64})::Vector{Float64}
    result = zeros(Float64, size(mat, COL)) # same dim as num rows
    for col in START:size(mat, COL) # matrix in column major layout
        for row in START:size(mat, ROW)
            result[col] += mat[row, col]^2
        end
    end
    result
end


"""
Symmetrizes `S`. Useful for when rounding error causes the matrix to not be
symmetric.
"""
function symmetrize(S::Matrix{Float64})::Matrix{Float64}
   # makes sure the matrix S is symmetric
   (S' .+ S) ./ 2
end


"""
Checks if the matrix `cov` is positive definite.

If yes, returns (true, lower) where lower is the lower triangular matrix of the
  cholesky decomposition.

If no, returns (false, [NaN]).
"""
function check_pos_def(cov::CovType)::Tuple{Bool,Matrix{Float64}}
   info = cholesky(cov; check=false) # don't throw exception
   if issuccess(info)
      (true, info.L)
   else
      (false, fill(NaN, 1, 1))
   end
end


"""
Computes the softmax of `x` with temperature parameter t: ``(1/t) log(sum_{i=1}^n e^(t * x[i]))``
"""
function softmax(x::Vector{Float64}, t::Float64)::Float64
    themax = maximum(x)
    sm = themax + log(sum(elt ->  exp(t*(elt - themax)), x))/t
    sm
end
#####################################
# Gradient and Hessian Calculations
#####################################
"""
Computes the gradient of the objective function
"""
function ffs_gradient(params::FFSParams,
                  S::Sigma;
                  tb::Float64,
                  tl::Float64)::Matrix{Float64}
    l_diag = diag_prod(params.L, S, params.c)
    l_max = maximum(l_diag)
    b_diag = diag_inv_prod(params.B, S)
    b_max = maximum(b_diag)
    weightsL = exp.((l_diag .- l_max) .* tl)
    weightsL ./= sum(weightsL)
    weightsL ./= params.c
    weightsB = exp.((b_diag .- b_max) .* tb)
    weightsB ./= sum(weightsB)
    -weighted_grad_helper_B(params.B, S, weightsB) + weightedLTL(params.L, weightsL)
end

function hess_times_vec(thevec::Vector{Float64},
                        params::FFSParams,
                        S::Sigma;
                        tb::Float64,
                        tl::Float64)::Float64
   #TODO
end


#############################
# Functions for Optimization
#############################

"""
Computes the unconstrained objective function for the fitness-for-use problem.
    The basis, representation, and target variances are provided in the `param`
    structure. `S` is the current covariance matrix. `tb` is the soft-max multiplier
    for the privacy part of the objective and `tl` is the soft-max multiplier for
    the fitness-for-use part.
"""
function objective(params::FFSParams, S::Sigma; tb::Float64, tl::Float64)::Float64
   # computes the objective function
   # params is the FFS setup (BT, L, and c)
   # S contains the covariance matrix information
   # tb is the multiplier for the basis matrix part of the objective function
   # tl is the multiplier for the fitness-for-use constraint part of the objective function

   b_diag = diag_inv_prod(params.B, S) # diagonal of B' * S * B
   l_diag = diag_prod(params.L, S, params.c) # diagonal of L * S * LT
   softmax(b_diag, tb) + softmax(l_diag, tl) #TODO
end


function line_search(params::FFSParams,
		     S::Sigma,
		     direction::Matrix{Float64},
		     gradient::Matrix{Float64};
		     tb::Float64,
		     tl::Float64,
		     dec::Float64 = 0.01,
		     beta = 0.5
		    )::Sigma
   # Performs a backtracking line search starting
   # from S in the specified direction
   # it uses the gradient as part of the sufficient decrease condition
   # dec is part of the sufficient decrease condition
   # beta is the candidate step size multiplier

   alpha = 1.0 #TDOD test
   fcurr = objective(params, S, tb=tb, tl=tl)
   done = false
   result = S
   while !done
      fprev = fcurr
      candidate .= direction .* alpha .+ S.cov
      (status, lower) = check_pos_def(candidate)
      if status
          fcurr = objective(params, Sigma(candidate, lower), tb=tb, tl=tl)
	  if fcurr <= fprev + alpha * dec * dot(direction, gradient)
	      result = Sigma(candidate, lower)
	      done = true
	  end
      end
      alpha = alpha * beta
   end
   result
end

function conjugate_gradient()
      #TODO
end

function ffs_optimize()
    #TODO
end

function initialize()
  #TODO
end

############################################
# Problem conversion functions
############################################

"""
Returns a rescaled covariance matrix so that the mechanism
``M(x) = L(Bx +N(0, cov))`` has the desired target privacy cost.
`B` is the basis matrix, `cov` is the unscaled covariance
matrix and `privcost` is the target privacy cost.
"""
function enforce_pcost(B::FFSMatrix, cov::CovType, privcost::Float64)::CovType
   actualcost = privacy_cost(B, cov)
   cov * (actualcost/privcost)^2
end


"""
Returns a rescaled covariance matrix so that the mechanism
``M(x) = L(Bx +N(0, cov))`` has the desired target ffs.
`L` is the representation matrix, `cov` is the unscaled covariance
matrix and `c` is the target ffs variance.
"""
function enforce_ffs(L::FFSMatrix, cov::CovType, c::Vector{Float64})::CovType
    overrun = ffs_overrun(L, cov, c)
    cov / overrun
end

###########################################
# Functions for Measuring Solution Quality
###########################################

"""
Returns the privacy cost (total precision, generalization of
L2 sensitivity) of a mechanism that adds Gaussian noise with covariance
`cov` to query answers answered with ``B``, where ``B`` is the basis
matrix.
"""
function privacy_cost(B::FFSMatrix, cov::CovType)::Float64
    (result, lower) = check_pos_def(cov)
    @assert result "Covariance must be positive definite"
    profile = diag_inv_prod(B, Sigma(cov, lower))
    cost = sqrt(maximum(profile))
    cost
end


"""
Returns the variance of each workload query for the Gaussian mechanism
    ``M(x) = L(Bx +N(0, cov))`` where `L` is the representation matrix
    and `cov` is the covariance matrix
"""
function l2error_vector(L::FFSMatrix, cov::CovType)::Vector{Float64}
    (result, lower) = check_pos_def(cov)
    @assert result "Covariance must be positive definite"
    l_vec = diag_prod(L, Sigma(cov, lower), 1.0)
    l_vec
end


"""
Returns the maximum ratio by which the fitness-for-use constraints are violated
for the Gaussian mechanism
    ``M(x) = L(Bx +N(0, cov))`` where `L` is the representation matrix,
     `cov` is the covariance matrix, and `c` is the vector of variance
     targets or a single float.
Ratio > 1.0 means the fitness for use constraints don't hold. Ratio < 1.0 means
that every query is more accurate than required.
"""
function ffs_overrun(L::FFSMatrix,
                     cov::CovType,
                     c::Union{Float64, Vector{Float64}})::Float64
    # Computes max ratio of variance to target variaance
    # at a given privacy cost
    l_vec = l2error_vector(L, cov) ./ c
    maximum(l_vec)
end

###########################################
# Exports
###########################################

export ffs_optimize,
       FFSMatrix,
       FFSDenseMatrix,
       FFSId,
       privacy_cost,
       l2error_vector,
       ffs_overrun,
       enforce_pcost,
       enforce_ffs
end
