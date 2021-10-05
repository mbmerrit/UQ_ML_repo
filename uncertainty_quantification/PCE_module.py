import numpy as np
import numpy.polynomial.hermite_e as H
import numpy.polynomial.legendre as L
import numpy.random as rnd
from scipy.special import factorial
from matplotlib import pyplot as plt

# This code now runs without need for the PyApprox library
# import pyapprox as pya

__all__ = ['evaluate_orthogonal_polynomials','get_polynomial_norms','evaluate_pce','get_sobol_using_pce','transf_uniform','transf_normal','EvalMulti_totdeg']


def evaluate_orthogonal_polynomials(multiindices, randomvector, basis):
    "This code evaluates the full set of multi-indices for a single random vector"
    "The dimension of the multi-inidces must be the same as the random vector"
    
    # Some initialization
    polynomial_order = np.max(multiindices) # this may not work if not using a total order scheme
    if multiindices.ndim > 1:
        dimension, P = np.shape(multiindices)
        multivariate_polynomial_value = np.ones(P) # the value of the polynomial, initialize to 1
        for i in range(dimension):
            coef = np.zeros((polynomial_order+1, P)) # this stores the coefficients of the polynomials

            coef[multiindices[i,:], np.arange(P)] = 1  # convert multi-indices into coefficients for numpy
            if basis == 'hermite':
                multivariate_polynomial_value *= H.hermeval(randomvector[i], coef)
            elif basis == 'legendre':
                multivariate_polynomial_value *= L.legval(randomvector[i], coef)
            else:
                return 'Error: use either Hermite or Legendre'
    else: 
        dimension = multiindices.size
        P = 1
        multivariate_polynomial_value = np.ones(P) # the value of the polynomial, initialize to 1
    
        for i in range(dimension):
            coef = np.zeros((polynomial_order+1,)) # this stores the coefficients of the polynomials

            coef[multiindices[i]] = 1  # convert multi-indices into coefficients for numpy
            if basis == 'hermite':
                multivariate_polynomial_value *= H.hermeval(randomvector[i], coef)
            elif basis == 'legendre':
                multivariate_polynomial_value *= L.legval(randomvector[i], coef)
            else:
                return 'Error: use either Hermite or Legendre'  
        
    return multivariate_polynomial_value

# For our 2 basis polynomials, this will return a vector of analytic norms, given corresponding multi-indices
def get_polynomial_norms(multiindices, basis, p1=-1.0, p2=1.0):
    "p1 and p2 are the parameters for each distribution, for U[p1, p2] and N(p1, p2)"
    if basis == 'hermite':
        norms = np.prod(p2*factorial(multiindices), axis=0)
    elif basis == 'legendre':
        # this formula might need to be checked, I believe this is what is used in Dakota
        norms = np.prod(1/(2*multiindices + 1), axis=0)
    else: return 'use either hermite or legendre basis'
    
    return norms

# This is for evaluating the PCE at a single random vector, so I would only use if you have to
def evaluate_pce(pce_coef, multiindices, basis, sampled_rv):
        k = np.dot(pce_coef, evaluate_orthogonal_polynomials(multiindices, sampled_rv, basis))
        return k
    
def get_sobol_using_pce(multiindices, pce_coef, basis):
    "This code requires the full spectrum of PCE coefficients to accurately estimate the variance."
    "If there is a sharp drop in the expansion coefficients, it should be sufficient to capture the variance"
    polynomial_norms = get_polynomial_norms(multiindices, basis)
    variance = np.sum(pce_coef[1:]**2*polynomial_norms[1:])
    params = multiindices.shape[0]
     
    total_indices = [0]*params
    first_order_indices = [0]*params
    for i in range(params):
        # first we compute total indices
        basis_elements = np.nonzero(multiindices[i,:])
        total_indices[i] = np.sum(pce_coef[basis_elements]**2*polynomial_norms[basis_elements])/variance
        # Now first order or main effects
        idx = multiindices[i,1:] == np.sum(multiindices[:,1:], axis=0)
        first_order_indices[i] = np.sum(pce_coef[1:][idx]**2*polynomial_norms[1:][idx])/variance
    
    print('var: ', variance)
    print('first order: ', first_order_indices)
    print('total indices: ', total_indices)
    
    return variance, first_order_indices, total_indices

def transf_uniform(x,a,b):
    "transforms uniform RVs on [a,b] to their values in the canonical interval [-1,1].  Works with multivariate RVs."
    variables, N = x.shape
    return ((2 * x) - np.tile((b + a), (N,1)).T) / np.tile((b - a), (N,1)).T

def transf_normal(x,mu,sigma):
    "transforms normal RVs on with mean mu and std dev sigma to N(0,1)"
    return (x - mu) / sigma

def inv_transf_uniform(u,a,b):
    '''Performs the inverse transform, mapping uniform RVs on [-1, 1] to [a,b]. Works with multivariate RVs.'''
    variables, N = x.shape
    return (u * np.tile((b - a), (N,1)).T + np.tile((b + a), (N,1)).T) /2

def inv_transf_normal(u,mu,sigma):
    return u*sigma + mu

def EvalMulti_totdeg(dim,deg):
    """
    Compute the multi-indices for a total order polynomial expansion
    Algorithm extracted from OLM & Knio, p. 517
    """

    # total number of terms
    P_tot = int(np.math.factorial(dim+deg) / ( np.math.factorial(dim) * np.math.factorial(deg) ))
    #print('Total number of PC coefficients: ', P_tot)

    # multi-index initialization    
    multiindices = np.zeros([dim,P_tot],dtype=int)
    p_vec = np.zeros([dim,deg],dtype=int)
    
    if ( deg >=1 ):
        
        # zeroth order
        multiindices[0:,1:dim+1] = 0

        # first order        
        for i in range(1,dim+1):
            
            multiindices[i-1,i] = 1
        
            loc_id = dim+1
            p_vec[:,0] = 1
            
        # higher order    
        for k in range(2,deg+1):
            
            L = loc_id
            for i in range(0,dim):
                p_vec[i,k-1] = np.sum( p_vec[i:,k-2] )
                
            for j in range(0,dim):
                for m in range(L-p_vec[j,k-1]+1,L+1):
                    loc_id = loc_id + 1
                    multiindices[:,loc_id-1] =  multiindices[:,m-1]
                    multiindices[j,loc_id-1] =  multiindices[j,loc_id-1] + 1
    
    return multiindices 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
