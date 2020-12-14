module Fitness
using LinearAlgebra


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
The default structure for a dense matrix
"""
struct FFSDenseMatrix <: FFSMatrix
   mat::Array{Float64,2}
end


"""
Types allowed for the covariance matrix
"""
CovType = Array{Float64, 2}


"""
Keeps track of the problem settings, including the basis matrix,
representation matrix, and target cost vector.
"""
struct FFSParams
   BT::FFSMatrix # B transpose
   L::FFSMatrix # L
   c::Array{Float64,1} # target variances
end


"""
Keeps track of the covariance matrix and its cholesky decomposition.
"""
struct Sigma
   cov::CovType # the covariance matrix
   lower::Array{Float64,2} # the lower triangular matrix from its cholesky decomposition
end



######################################
# Helper methods with
# potential for speeding up FFS
#
# To speed up your own W=LB
# decomposition, create a subtype
# of FFSMatrix and implement
# a fast version of these functions
#
######################################

"""
Returns the diagonal of ``A S A'`` divided pointwise by `c`.
For dense matrices, we compute the squared norm of each row of ``A * lower``,
where `lower` is the lower triangular matrix of the cholesky decomposition
of the covariance `S`. This should be specialized to other matrix representations
for speed.
"""
function diag_prod(A::FFSDenseMatrix,
                   S::Sigma,
                   c::Union{Float64, Array{Float64,1}})::Array{Float64,1}

   halfway = A.mat * S.lower # next compute squared norm of each row
   result = zeros(Float64, size(halfway, ROW)) # same dim as num rows
   for col in START:size(halfway, COL) # matrix in column major layout
       for row in START:size(halfway, ROW)
           result[row] += halfway[row, col]^2
       end
   end
   result ./= c
   result
end


"""
Returns the diagonal of ``A S^{-1} A'``.
For dense matrices, we compute the squared norm of each column of ``x``, where ``x`` is obtained
by solving ``lower x = A`` through forward substitution,
where ``lower`` is the lower triangular matrix of the cholesky decomposition
of the covariance `S`. This should be specialized to other matrix representations
for speed.
"""
function diag_inv_prod(A::FFSDenseMatrix, S::Sigma)::Array{Float64,1}
   halfway_t = S.lower \ A.mat'  # next compute squared norm of each column
   result = zeros(Float64, size(halfway_t, COL)) # same dim as num cols
   for col in START:size(halfway_t, COL) # matrix in column major layout
       for row in START:size(halfway_t, ROW)
           result[col] += halfway_t[row, col]^2
       end
   end
   result
end


#############################
# Helper functions
############################

"""
Symmetrizes `S`. Useful for when rounding error causes the matrix to not be
symmetric.
"""
function symmetrize(S::Array{Float64,2})::Array{Float64,2}
   # makes sure the matrix S is symmetric
   (S' .+ S) ./ 2
end


"""
Checks if the matrix `cov` is positive definite.

If yes, returns (true, lower) where lower is the lower triangular matrix of the
  cholesky decomposition.

If no, returns (false, [NaN]).
"""
function check_pos_def(cov::CovType)::Tuple{Bool,Array{Float64,2}}
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
function softmax(x::Array{Float64,1}, t::Float64)::Float64
    themax = maximum(x)
    sm = themax + log(sum(elt ->  exp(t*(elt - themax)), x))/t
    sm
end
#####################################
# Gradient and Hessian Calculations
#####################################

function gradient(params::FFSParams,
                  S::Sigma;
                  tb::Float64,
                  tl::Float64)::Array{Float64,2}
  #TODO
end

function hess_times_vec(thevec::Array{64,2},
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

   b_diag = diag_inv_prod(params.BT, S) # diagonal of BT * S * B
   l_diag = diag_prod(params.L, S, params.c) # diagonal of L * S * LT
   softmax(b_diag, tb) + softmax(l_diag, tl) #TODO
end


function line_search(params::FFSParams,
		     S::Sigma,
		     direction::Array{Float64,2},
		     gradient::Array{Float64,2};
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

   alpha = 1.0
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
`BT` is the transpose of the basis matrix, `cov` is the unscaled covariance
matrix and `privcost` is the target privacy cost.
"""
function enforce_pcost(BT::FFSMatrix, cov::CovType, privcost::Float64)::CovType
   actualcost = privacy_cost(BT, cov)
   cov * (actualcost/privcost)^2
end


"""
Returns a rescaled covariance matrix so that the mechanism
``M(x) = L(Bx +N(0, cov))`` has the desired target ffs.
`L` is the representation matrix, `cov` is the unscaled covariance
matrix and `c` is the target ffs variance.
"""
function enforce_ffs(L::FFSMatrix, cov::CovType, c::Array{Float64,1})::CovType
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
matrix whose transpose is `BT`.
"""
function privacy_cost(BT::FFSMatrix, cov::CovType)::Float64
    (result, lower) = check_pos_def(cov)
    @assert result "Covariance must be positive definite"
    profile = diag_inv_prod(BT, Sigma(cov, lower))
    cost = sqrt(maximum(profile))
    cost
end


"""
Returns the variance of each workload query for the Gaussian mechanism
    ``M(x) = L(Bx +N(0, cov))`` where `L` is the representation matrix
    and `cov` is the covariance matrix
"""
function l2error_vector(L::FFSMatrix, cov::CovType)::Array{Float64,1}
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
                     c::Union{Float64, Array{Float64,1}})::Float64
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
       diag_prod,
       diag_inv_prod,
       privacy_cost,
       l2error_vector,
       ffs_overrun,
       enforce_pcost,
       enforce_ffs
end
