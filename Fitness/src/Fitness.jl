module Fitness
using LinearAlgebra

############################
##  Types for FFS
###########################


# Extend the type FFSMatrix for
# L and B matrices where you
# can speed up computation

abstract type FFSMatrix end

struct FFSDenseMatrix <: FFSMatrix
   mat::Array{Float64,2}
end


# Keeps track of the problem settings

struct FFSParams
   BT::FFSMatrix # B transpose
   L::FFSMatrix # L
   c::Array{Float64,1} # target variances
end

# Keeps track of useful information about the covariance matrix

struct Sigma
   cov::Array{Float64,2} # the covariance matrix
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

function diag_prod(A::FFSDenseMatrix, S::Sigma, c::Array{Float64,1})::Array{Float64,1}
   # returns the diagonal of ASA' divided pointwise by c
   # lower is the lower diaongal matrix of the cholesky decomposition of S
   # For dense matrices, we compute the squared norm of each column of A * lower

   halfway = A.mat * S.lower # next compute squared norm of each row
   result = zeros(Float64, size(halfway, 1)) # same dim as num rows
   for col in 1:size(halfway, 2) # matrix in column major layout
       for row in 1:size(halfway, 1)
           result[row] += halfway[row, col]^2
       end
   end
   result ./= c
   result
end


function diag_inv_prod(A::FFSDenseMatrix, S::Sigma)::Array{Float64,1}
   # returns the diagonal of A * inv(S) *A'
   # lower is the lower diaongal matrix of the cholesky decomposition of S
   # For dense matrices, we compute the squared norm of each column of A * lower

   halfway_t = S.lower \ A.mat'  # next compute squared norm of each column
   result = zeros(Float64, size(halfway_t, 2)) # same dim as num cols
   for col in 1:size(halfway_t, 2) # matrix in column major layout
       for row in 1:size(halfway_t, 1)
           result[col] += halfway_t[row, col]^2
       end
   end
   result
end


#############################
# Helper functions
############################

function symmetrize(S::Array{Float64,2})::Array{Float64,2}
   # makes sure the matrix S is symmetric
   (S' .+ S) ./ 2
end


function check_pos_def(cov::Array{Float64,2})::Tuple{Bool,Array{Float64,2}}
   # checks if the matrix S is positive definite
   # If yes, returns (true, lower) where lower is the lower triangular matrix
   #   if no, returns (false, [NaN])
   info = cholesky(cov; check=false) # don't throw exception
   if issuccess(info)
      (true, info.L)
   else
      (false, fill(NaN, 1, 1))
   end
end


#####################################
# Gradient and Hessian Calculations
#####################################

function gradient()
end

function hess_times_vec()
end


#############################
# Functions for Optimization
#############################


function objective(params::FFSParams, S::Sigma; tb::Float64, tl::Float64)::Float64
   # computes the objective function
   # params is the FFS setup (BT, L, and c)
   # S contains the covariance matrix information
   # tb is the multiplier for the basis matrix part of the objective function
   # tl is the multiplier for the fitness-for-use constraint part of the objective function

   b_diag = diag_inv_prod(params.BT, S) # diagonal of BT * S * B
   l_diag = diag_prod(params.L, S, params.c) # diagonal of L * S * LT

   log(sum(x -> tb * exp(x), b_diag))/tb + log(sum(x -> tl * exp(x), l_diag))/tl
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
end

function ffs_optimize()
end

function initialize()

end
###########################################
# Functions for Measuring Solution Quality
###########################################

function privacy_cost()
end

function expected_error_vector()
end


###########################################
# Exports
###########################################

export ffs_optimize, FFSMatrix, FFSDenseMatrix, diag_prod, diag_inv_prod

end
