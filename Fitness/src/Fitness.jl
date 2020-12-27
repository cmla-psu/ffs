module Fitness
using LinearAlgebra


const START = 1 #starting index
const ROW = 1 # first dimension is ROW
const COL = 2 # second dimension is COL
############################
##  Types for FFS
###########################

 struct FFSException <: Exception
    msg::String
 end

"""
Tuning parameters for FFS optimization. Must create used named parameters.

nttol: lower bound for dot(search direction, gradient)
gaptol: limits size of softmax relatxion (number queries + domainsize)/t
softmu: multiplier for softmax parameter t
ls_dec: paramter for line search sufficient decrease condition
ls_beta: step size multiplier in line search
fudge: initialization parameter
"""
struct FFSTuningParams #TODO test
    nttol::Float64
    gaptol::Float64
    softmu::Float64
    ls_dec::Float64
    ls_beta::Float64
    cg_tol::Float64
    cg_iter::Int64
    fudge::Float64
    FFSTuningParams(;nttol=0.01,
                     gaptol=0.1,
                     softmu=2.0,
                     ls_dec=0.01,
                     ls_beta=0.5,
                     cg_tol=1e-10,
                     cg_iter=5,
                     fudge=0.99) = new(nttol,gaptol,softmu,ls_dec,ls_beta,
                                      cg_tol, cg_iter,fudge)
end


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
struct FFSParams{TB<:FFSMatrix,TL<:FFSMatrix}
   B::TB # B
   L::TL # L
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


########################################
# Subtypes of FFSMatrix and
# overloaded multiplication operations
# and overloaded weightedLTL
########################################
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
For each row v_i of L, computes ``sum_i weights[i] v_i' v``
which is the same as ``L' diag(weights) L``.
"""
weightedLTL(L::FFSDenseMatrix, weights::Vector{Float64})::Matrix{Float64} = (weights' .* L.mat') * L.mat

function diag_prod(L::FFSDenseMatrix,
                   S::Matrix{Float64},
                   c::Union{Float64, Vector{Float64}})::Vector{Float64}
    result = sum((L.mat * S) .* L.mat, dims=COL) ./ c
    reshape(result, :) # from m x 1 matrix to col vecctor
end

#####################################

# Identity matrix for FFS. The identity matrix is a singleton called FFSId
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
Specialized versions of diag_prod when cholesky decomposition is not available
"""
function diag_prod(L::FFSMatrix,
                   S::Matrix{Float64},
                   c::Union{Float64, Vector{Float64}})::Vector{Float64}
    diag(L*(L * S)') ./ c
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
Computes ``sum_i weights[i] S^{-1} b_i b_i' S^{-1}``,
which is the same as S^{-1} B diag(weights) B' S^{-1}.
"""
function weighted_grad_helper_B(B::FFSMatrix, S::Sigma, weights::Vector{Float64})::Matrix{Float64}
    left = S.lowerinv' * (S.lowerinv * B)
    (weights' .* left ) * left'
end

"""
For each column b_i of B, computes the inner product <S^{-1} b_i b_i' S^{-1},  C>
for a symmetric matrix C. This is equal to the diagonals of (S^{-1} B)' C (S^{-1} B)
"""
function weighted_hess_helper_B(B::FFSMatrix, S::Sigma, C::Matrix{Float64})::Vector{Float64}
    left = S.lowerinv' * (S.lowerinv * B) # S^{-1}B
    reshape(sum(left .* (C * left), dims=1), :)
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
   (S' + S) / 2
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

struct GradInfo
   gradient::Matrix{Float64}
   gradB::Matrix{Float64} #gradient of the privacy part
   gradL::Matrix{Float64} #gradient of the utility part
   weightsB::Vector{Float64} #exponential weights in the utility gradient
   weightsL::Vector{Float64} #exponential weights in the privacy gradient
   GradInfo(;gradB, gradL, weightsB, weightsL) = new(gradB+gradL, gradB, gradL, weightsB, weightsL)
end

"""
Computes the gradient of the objective function
"""
function ffs_gradient(params::FFSParams,
                  S::Sigma;
                  tb::Float64,
                  tl::Float64)::GradInfo
    l_diag = diag_prod(params.L, S, params.c)
    l_max = maximum(l_diag)
    b_diag = diag_inv_prod(params.B, S)
    b_max = maximum(b_diag)
    weightsL = exp.((l_diag .- l_max) * tl)
    weightsL /= sum(weightsL)
    weightsL ./= params.c
    weightsB = exp.((b_diag .- b_max) * tb)
    weightsB /= sum(weightsB)
    gradB = -weighted_grad_helper_B(params.B, S, weightsB)
    gradL = weightedLTL(params.L, weightsL)
    GradInfo(gradB=gradB, gradL=gradL, weightsB=weightsB, weightsL=weightsL)
end


""" Computes directly the product of the Hessian times the direction
Since our variable of interest is a matrix, this is equivalent to
flattening the variable, taking the Hessian, multipliying by the
flattened direction, and reshaping the result.

The variable ginfo contains the gradient along with some intermediate
gradient calculations that can be re-used. The algorithm in this function
is based on the idea that if f is an objective function g is its gradient,
h is its hessian, the h(x) * direction = d g(x+t*direction)/dt evaluated at
t=0.
"""
function hess_times_direction(ginfo::GradInfo,
                        direction::Matrix{Float64},
                        params::FFSParams,
                        S::Sigma;
                        tb::Float64,
                        tl::Float64)::Matrix{Float64}

    new_weights_L = diag_prod(params.L, direction, 1.0) .* ginfo.weightsL * tl #vector of derivatives of each exponential term
    hess_L_part1 = weightedLTL(params.L, new_weights_L ./ params.c)
    hess_L_part2 = -sum(new_weights_L) * ginfo.gradL
    hess_prod_L = hess_L_part1 + hess_L_part2
    new_weights_B = weighted_hess_helper_B(params.B, S, direction) .* ginfo.weightsB * (-tb)
    hess_B_part1 = -weighted_grad_helper_B(params.B, S, new_weights_B)
    hess_B_part2 = -sum(new_weights_B) * ginfo.gradB
    hess_B_part3half = -(ginfo.gradB * direction * S.lowerinv' * S.lowerinv)
    hess_prod_B = hess_B_part1 + hess_B_part2 + hess_B_part3half + hess_B_part3half'
    hess_prod_L + hess_prod_B
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
   softmax(b_diag, tb) + softmax(l_diag, tl)
end


function line_search(params::FFSParams,
		     S::Sigma,
		     direction::Matrix{Float64},
		     gradient::Matrix{Float64};
		     tb::Float64,
		     tl::Float64,
		     dec::Float64 = 0.01,
		     beta::Float64 = 0.5
		    )::Sigma
   # Performs a backtracking line search starting
   # from S in the specified direction
   # it uses the gradient as part of the sufficient decrease condition
   # dec is part of the sufficient decrease condition
   # beta is the candidate step size multiplier

   alpha = 1.0
   foriginal = objective(params, S, tb=tb, tl=tl)
   fcurr = foriginal
   done = false
   result = S
   while !done
      candidate = direction * alpha + S.cov
      (status, lower) = check_pos_def(candidate)
      if status
          fcurr = objective(params, Sigma(candidate, lower), tb=tb, tl=tl)
	      if fcurr <= foriginal + alpha * dec * dot(direction, gradient)
	          result = Sigma(candidate, lower)
	          done = true
	      end
      end
      alpha = alpha * beta
   end
   result
end

function conjugate_gradient(ginfo::GradInfo,
                            params::FFSParams,
                            S::Sigma,
                            maxcg::Int;
                            tb::Float64,
                            tl::Float64,
                            tol2::Float64=1e-10)::Matrix{Float64}
    direction = zeros(size(ginfo.gradient))
    r = -ginfo.gradient
    p = copy(r)
    rs_old = dot(r,r)
    for i in 1:maxcg
        Hp = hess_times_direction(ginfo, p, params, S, tb=tb, tl=tl)
        a = rs_old / dot(p, Hp)
        @. direction += a * p
        @. r -= a * Hp
        rs_new = dot(r,r)
        if rs_new <= tol2
            break
        end
        b = rs_new/rs_old
        @. p = r + b * p
        rs_old = rs_new
    end
    direction
end

""" Perform Fitness for Use optimization """
function ffs_optimize(;B::FFSMatrix,
                       L::FFSMatrix,
                       c=Vector{Float64},
                       tune::FFSTuningParams=FFSTuningParams())::Matrix{Float64}
    br, bc = size(B)
    lr, lc = size(L)
    numc = length(c)
    if lc != br
        throw(FFSException("Number of columns in L must equal number of rows in B"))
    elseif lr != numc
        throw(FFSException("Number of rows in L must equal number of fitness for use constraints (length of c)"))
    end
    params = FFSParams(B, L, c)
    S = initialize(params, fudge=tune.fudge)
    #TODO test
    S.cov
end





"""
Creates an initial covarinace matrix. Q is either the identity or a guess
for the covariance. Hence the Q should be a symmetric positive definite
square matrix having the same
number of rows as B
"""
function initialize(params::FFSParams; Q=I(size(params.B,ROW)*1.0), fudge = 0.99)::Sigma
    if size(Q) != (size(params.B,ROW), size(params.B,ROW))
        throw(FFSException("Q must be square matrix with same number of rows as B"))
    elseif Q != Q'
        throw(FFSException("Q must be symmetric"))
    end
    cov = fudge * enforce_ffs(params.L, symmetrize(collect(Q)), params.c)
    (ispd, lower) = check_pos_def(cov)
    if ispd
        Sigma(cov,lower)
    else
        throw(FFSException("Initial Matrix Q is not symmetric positive definite"))
    end
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
    if !result
        throw(FFSException("Covariance must be positive definite"))
    end
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
    if !result
        throw(FFSException("Covariance must be positive definite"))
    end
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
       enforce_ffs,
       FFSException,
       FFSTuningParams
end
