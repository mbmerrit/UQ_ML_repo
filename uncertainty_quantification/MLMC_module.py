import numpy as np
import PCE_module as p
import numpy.random as rnd
from sympy.tensor.array import MutableDenseNDimArray as MArray
from itertools import combinations


__all__ = ['Multi_Level_Model','generate_samples_orig','generate_samples','sample_polynomial','evaluate_ML_functions','sample_allocation_giles','estimate_kth_pce_coef','estimate_pce_ensemble','evaluate_ml_variance','evaluate_ml_covariance','evaluate_moments','extract_coefs_4gsa','estimate_sobol_var','evaluate_var_sobol','extract_multiple_gsa_coefs','constraint_var_sobol','get_sobol_indices','decompose_variance','sobol_drop_tolerance']


# create a class that stores all of the input data for the MLMC PCE task
# the ML class of functions will include the number of levels, the function at each level, the basis, the costs, etc.
class Multi_Level_Model:
    
    def __init__(self, name):
        self.name = name
        self.levels = 0
        self.functions = []
        self.costs = np.array([])
        self.var_analytical = []
    
    def basis(self, basis):
        self.basis = basis
    
    def basis_params(self, basis_params):
        self.basis_params = basis_params
        
    def variables(self, variables):
        self.variables = variables
        
    def add_level(self, level_func, cost, var='n/a'):
        self.levels += 1
        self.functions.append(level_func)
        self.costs = np.append(self.costs, cost)
        self.var_analytical.append(var) # optional: add analytical variance of this level

       # this is an older version of generate_samples, compatible with the individual and worst case methods
def generate_samples_orig(N, MLM, samples, samples_on_ab):
    '''Given a vector of number of samples to add by level, return input random variables.
    Think of N as a DeltaN_l vector of number of new samples by level
    Implemented by Michael Merritt, July 2020.'''
    A, B = MLM.basis_params # these can be uniform bounds or normal mu and sigma
    vars = MLM.variables # this needs to be variable, determined by function
    L = MLM.levels
    
    # for initializing the samples
    if not(samples) and not(samples_on_ab):
        samples = [[]]*L; samples_on_ab = [[]]*L
        for i in range(L):
            if MLM.basis == 'legendre':
                samples_on_ab[i] = rnd.uniform(A, B, (vars,N[i]))
                samples[i] = p.transf_uniform(samples_on_ab[i], A, B) 
            elif MLM.basis == 'hermite':
                samples_on_ab[i] = rnd.normal(A, B, (vars,N[i]))
                samples[i] = p.transf_normal(samples_on_ab[i], A, B)
    # for enlarging the samples
    else:
        for i in range(L):
            if MLM.basis == 'legendre':
                newsamples_on_ab = rnd.uniform(A, B, (vars,N[i]))
                newsamples = p.transf_uniform(newsamples_on_ab, A, B) 
                samples_on_ab[i] = np.append(samples_on_ab[i], newsamples_on_ab, axis=1)
                samples[i] = np.append(samples[i], newsamples, axis=1)
            elif MLM.basis == 'hermite':
                newsamples_on_ab = rnd.normal(A, B, (vars,N[i]))
                newsamples = p.transf_normal(newsamples_on_ab, A, B)
                samples_on_ab[i] = np.append(samples_on_ab[i], newsamples_on_ab, axis=1)
                samples[i] = np.append(samples[i], newsamples, axis=1)

    return samples, samples_on_ab

    # This sample generating code is compatible with the new PCE estimation code for GSA
def generate_samples(N, MLM, samples, samples_on_ab):
    '''Given a vector of number of samples to add by level, return input random variables.
    Think of N as a DeltaN_l vector of number of new samples by level.
    Implemented by Michael Merritt, September 2020.'''
    A, B = MLM.basis_params # these can be uniform bounds or normal mu and sigma
    vars = MLM.variables # this needs to be variable, determined by function
    L = MLM.levels
    
    # for initializing the samples
    for i in range(L):
        if samples[i] == [] and samples_on_ab[i] == []:
            if MLM.basis == 'legendre':
                samples_on_ab[i] = rnd.uniform(A, B, (vars,N[i]))
                samples[i] = p.transf_uniform(samples_on_ab[i], A, B) 
            elif MLM.basis == 'hermite':
                samples_on_ab[i] = rnd.normal(A, B, (vars,N[i]))
                samples[i] = p.transf_normal(samples_on_ab[i], A, B)
        # for enlarging the samples
        else:
            if MLM.basis == 'legendre':
                newsamples_on_ab = rnd.uniform(A, B, (vars,N[i]))
                newsamples = p.transf_uniform(newsamples_on_ab, A, B) 
                samples_on_ab[i] = np.append(samples_on_ab[i], newsamples_on_ab, axis=1)
                samples[i] = np.append(samples[i], newsamples, axis=1)
            elif MLM.basis == 'hermite':
                newsamples_on_ab = rnd.normal(A, B, (vars,N[i]))
                newsamples = p.transf_normal(newsamples_on_ab, A, B)
                samples_on_ab[i] = np.append(samples_on_ab[i], newsamples_on_ab, axis=1)
                samples[i] = np.append(samples[i], newsamples, axis=1)

    return samples, samples_on_ab

def sample_polynomial(N, MLM, samples, sampled_polynomials, multiindex):
    ''' Takes the newly generated samples and evaluates the orthogonal polynomial.
    Think of N as a DeltaN_l vector of number of new samples by level.
    Implemented by Michael Merritt, July 2020.'''
    if not(sampled_polynomials):
        sampled_polynomials = [[]]*MLM.levels
    else:
        pass
    for i in range(MLM.levels):
        L_update = np.zeros((N[i],))
        start = len(sampled_polynomials[i])
        for j in range(N[i]):
            L_update[j] = p.evaluate_orthogonal_polynomials(multiindex, samples[i][:,start+j], MLM.basis)
        sampled_polynomials[i] = np.append(sampled_polynomials[i], L_update)
    
    return sampled_polynomials

def evaluate_ML_functions(N, MLM, samples_on_ab, function_evals):
    ''' Takes the newly generated samples and evaluates the function at them.
    Think of N as a DeltaN_l vector of number of new samples by level.
    Implemented by Michael Merritt, July 2020.'''
    if not(function_evals):
        function_evals = [[]]*MLM.levels
    else:
        pass
    for i in range(MLM.levels):
        fun_update = np.zeros((N[i],)) 
        start = len(function_evals[i])
        if i == 0:
            fun_update = MLM.functions[i](samples_on_ab[i][:,start:]) # only for first level
        else: 
            fun_update = MLM.functions[i](samples_on_ab[i][:,start:]) - MLM.functions[i-1](samples_on_ab[i][:,start:])
        function_evals[i] = np.append(function_evals[i], fun_update)
    
    return function_evals

def sample_allocation_giles(function_evals, sampled_polynomials, MLM, eps2, N, polynomial_norm):
    '''Evaluates the sample variance with the given function evaluations and returns the optimal 
    sample allocation strategy, given a target variance reduction, subject to total cost. Reference
    for sample allocation: Giles, Michael B. "Multilevel monte carlo methods." Acta Numerica (2015)
    Implemented by Michael Merritt, July 2020.'''
    var_sampled = np.zeros((MLM.levels,))
    for i in range(MLM.levels):
        # compute sampled variance and store it for each level, this is the same as Var[P_ell, k}] for a given k
        var_sampled[i] = np.var(function_evals[i][:N[i]] * sampled_polynomials[i][:N[i]]) #/ polynomial_norm
        
    # compute the Lagrange multiplier
    mu = np.sum(np.sqrt( var_sampled * MLM.costs )) / eps2
    N_opt = np.round(mu * np.sqrt(var_sampled / MLM.costs)).astype(int)
    
    return var_sampled, N_opt      


# This is one of the main functions that calls the above sub-functions
def estimate_kth_pce_coef(k, eps2, MLM, max_iters):
    '''This code estimates the kth PCE coefficient using a L level MLMC scheme. 
    The essential inputs are the index of the coefficient k, the target accuracy eps2,
    a and multi level function class with the other relevant pieces.
    Implemented by Michael Merritt, July 2020.'''
    
    vars = MLM.variables # this needs to be variable, determined by function
    L = MLM.levels
    C = MLM.costs
    
    # Initialization: set up the PCE variables and the full set of input samples
    multiindices = p.EvalMulti_totdeg(vars,k) # must be a better way for this
    P = multiindices.shape[1]
    delta_N = [20]*L
    N_opt = [20]*L
    samples = []; samples_on_ab = []
    sampled_polynomials = []
    polynomial_norm = p.get_polynomial_norms(multiindices[:,k], MLM.basis)    
    function_evals = []
    var_sampled = np.zeros((L,))
    iters = 1
    
    # the main iteration that allocates samples and evaluates the MLM
    while max(delta_N) > 0 and iters < max_iters:
        samples, samples_on_ab = generate_samples_orig(delta_N, MLM, samples, samples_on_ab)
        sampled_polynomials = sample_polynomial(delta_N, MLM, samples, sampled_polynomials, multiindices[:,k])
        function_evals = evaluate_ML_functions(delta_N, MLM, samples_on_ab, function_evals)
        N_opt_old = N_opt
        var_sampled, N_opt = sample_allocation_giles(function_evals, sampled_polynomials, MLM, eps2, N_opt_old, polynomial_norm)
        delta_N_raw = [N_opt[i] - N_opt_old[i] for i in range(L)]
        delta_N = np.array([N_opt[i] - N_opt_old[i] for i in range(L)]).clip(min=0)
        print('Iteration: ',iters)
        print('Sampled variance: ', var_sampled)
        print('Sampling by level: ',N_opt)
        print('Change in sample allocation: ', delta_N_raw)
        iters += 1

    # Post-processing: this is where we sample the ML QoI and compute PCE projections
    numerator = 0
    for i in range(MLM.levels):
        numer_update = np.sum(function_evals[i][:N_opt[i]] * sampled_polynomials[i][:N_opt[i]])
        numerator += ((numer_update) / N_opt[i])
    
    beta_k = numerator / polynomial_norm
    # compute the variance of beta_k one final time, with optimal samples
    for i in range(MLM.levels):
        var_sampled[i] = np.var(function_evals[i][:N_opt[i]] * sampled_polynomials[i][:N_opt[i]]) / polynomial_norm
    total_cost = np.sum(C*N_opt) + np.sum(C[1:]*N_opt[1:]); print('total cost: ', total_cost) # this is an option
    
    # Need to fix this section, the optimal sample does not include the basis contribution
    if 'n/a' in MLM.var_analytical:
        print('cannot determine optimal sampling analytically')
    else: 
        optimal_sampling = np.sum(np.sqrt(MLM.var_analytical * C))/eps2 * np.sqrt(MLM.var_analytical / C)
        #print('optimal sampling ratio: ',optimal_sampling)
     
    estim_var = np.sum(var_sampled / N_opt)
    print('estimated variance of beta k: ', estim_var)
 
    return function_evals, beta_k, sampled_polynomials, N_opt, total_cost


# This function estimates an ensemble of PCE coefficients, using a variety of methods to handle sampling.
# Once completed, this will be the main function for this module, referring to all sub-functions. It's a bit out of date now.
def estimate_pce_ensemble(K, eps2, MLM, max_iters, method='individual'):
    '''This code estimates a set of PCE coefficients K using an L level MLMC scheme. 
    The essential inputs are the index of the coefficient k, the target accuracy eps2,
    a and multi level function class with the other relevant pieces.
    
    The method specified can be either individual, worst-case, or global, each enatailing
    a different method for sample allocation across PCE modes.
    
    The indexing of function evaluations and such follows the convention that the first index
    refers to the PCE coefficient and the second refers to function level.
    Implemented by Michael Merritt, July 2020.'''
    # Initialization
    vars = MLM.variables # this needs to be variable, determined by function
    L = MLM.levels
    C = MLM.costs
    P = len(K)
    BETA = np.zeros((P,))
    COST = np.zeros((P,))
    multiindices_full = p.EvalMulti_totdeg(vars,max(K)) # must be a better way for this
    multiindices = multiindices_full[:,K] # the multi-indices we need
    polynomial_norms = p.get_polynomial_norms(multiindices, MLM.basis)
    samples = [[]]*P; samples_on_ab = [[]]*P
    sampled_polynomials = [[]]*P; function_evals = [[]]*P
    
    if method=='individual':
        N_opt = np.zeros((P,L))
        for i in range(P):
            print('----------------Estimating coefficient', K[i],'----------------')
            function_evals[i], BETA[i], sampled_polynomials[i], N_opt[i,:], COST[i] = \
            estimate_kth_pce_coef(K[i], eps2, MLM, max_iters)
            
    if method=='worst-case':
        delta_N = [20]*L   # start with 20 pilot samples
        N_by_level = [20]*L
        N_opt = np.zeros((P,L))
        var_sampled = np.zeros((P,L))
        estim_var = np.zeros((P,L))
        iters = 1
        while max(delta_N) > 0 and iters < max_iters:
            print('Iteration: ',iters)
            N_opt_old = np.copy(N_by_level)
            for j in range(P):
                samples[j], samples_on_ab[j] = generate_samples(delta_N, MLM, samples[j], samples_on_ab[j])
                sampled_polynomials[j] = sample_polynomial(delta_N, MLM, samples[j], sampled_polynomials[j], multiindices[:,j])
                function_evals[j] = evaluate_ML_functions(delta_N, MLM, samples_on_ab[j], function_evals[j])
                var_sampled[j,:], N_opt[j,:] = sample_allocation_giles(function_evals[j], sampled_polynomials[j], MLM, eps2, N_opt_old, polynomial_norms[j])
                estim_var = var_sampled / N_opt_old  # new variance estimate computed with old sampling
            # Determine global sample allocation by finding the worst case, highest variance, PCE coefficient
            print('Variance of coefficient estimators: ', np.sum(estim_var,1))
            wc_idx = np.argmax(estim_var, axis=0)
            N_by_level = N_opt[wc_idx,np.arange(L)].astype(int)
            print('Worst case sampling by level: ',N_by_level)
            print('Worst case coefficient index by level: ',wc_idx)
            delta_N_raw = N_by_level - N_opt_old
            delta_N = np.array(N_by_level - N_opt_old).clip(min=0)
            print('Change in sample allocation: ', delta_N_raw)
            iters += 1

        #Post-processing
        numerator = np.zeros((P,))
        for j in range(P):
            for i in range(MLM.levels):
                numer_update = np.sum(function_evals[j][i][:N_by_level[i]] * sampled_polynomials[j][i][:N_by_level[i]])
                numerator[j] += ((numer_update) / N_by_level[i])
        BETA = numerator / polynomial_norms
        # the cost is the same for each coefficient, so it is repeated P times
        COST = [np.sum(C*N_by_level) + np.sum(C[1:]*N_by_level[1:])]*P  # does not count oversampling cost
        N_opt = N_by_level
        
        if method == 'global':
            print('still working on this option...')

    return function_evals, BETA, sampled_polynomials, N_opt, COST

#
# This next section of codes handle MLMC-PCE computations for targeting GSA indices. They use the function and 
# polynomial evaluations to estimate the necessary ML variance and covariance terms required to build 
# an estimator for the variance of a particular Sobol' index. The goal is to merge these codes with the above one.
#

 # Now we implement the ML variance and covariance expression
def evaluate_ml_variance(k, moments, MLM, N, polynomial_norms):
    '''Computes the ML variance of (\hat{beta}_k)^2 using moments stored in the dictionary moments.
    Form of variance estimator is derived from the assumption that \hat{\beta}_k is normally distributed.
    Implemented by Michael Merritt, September 2020.'''
    L = MLM.levels
    EP = moments['EP']   # unpack the moments dictionary
    EPP = moments['EPP']
    if type(N[0]) != type(polynomial_norms): # implies we are doing a symbolic computation
        pass
    else:
        N = np.clip(N,1,None).tolist()
    
    #Var_betak = np.sum(estim_var[k,:])/ (polynomial_norms[k]**2) # no cancellation, but needs an estimate of V[P_{l,k}]
    Exp_betak = np.sum([EP[l,k] for l in range(L)])/polynomial_norms[k]
    Var_betak = np.sum([(EPP[l,k,k] - EP[l,k]**2) / N[l] for l in range(L)])/ (polynomial_norms[k]**2) # cancellation?
    ML_var_k = 4 * (Exp_betak**2) * Var_betak + 2 * (Var_betak**2)
    
    return ML_var_k
    
def evaluate_ml_covariance(k, z, moments, MLM, N, polynomial_norms):
    '''Computes the ML covariance of (\hat{beta}_k)^2 and (\hat{beta}_z)^2 using the moments of Q.
    The moments dictionary stores the necessary moments of Q and Psi from the evaluate_moments function. 
    Using moments will allow us to avoid recomputing them each time, just look at relevant indices.
    The following covariance formula is derived fully in the appendix to the proceedings article.
    Implemented by Michael Merritt, September 2020.'''
    L = MLM.levels
    Term_one = np.zeros((L,)); Term_two = np.zeros((L,)); Term_three = np.zeros((L,)); Term_four = np.zeros((L,))
    EP = moments['EP']   # unpack the moments dictionary
    EPP = moments['EPP']
    EP2P2 = moments['EP2P2']
    EP2P = moments['EP2P']
    
    for l in range(L):
        N_l = N[l]
        
        T1a = EP2P2[l,k,z] - (EPP[l,k,k]*EPP[l,z,z])
        T1b = (EP2P[l,k,z]*EP[l,z]) - (EPP[l,k,k]*(EP[l,z]**2))
        T1c = (EP2P[l,z,k]*EP[l,k]) - (EPP[l,z,z]*(EP[l,k]**2))
        T1d = EPP[l,k,z]**2
        T1e = EPP[l,k,z]*EP[l,k]*EP[l,z]
        T1f = (EP[l,k]**2)*(EP[l,z]**2)
        
        Term_one[l] =  (1/N_l**3) * ( T1a + (2*N_l-2)*T1b + (2*N_l-2)*T1c + (2*N_l-2)*T1d + \
                                     4*(N_l-1)*(N_l-2)*T1e - (4*N_l**2 - 10*N_l + 6)*T1f )
        
        T2n = np.zeros((L,)) # reset the Term 2 and 3 vectors
        T3n = np.zeros((L,))
        T4n = np.zeros((L,)) # this term 4 vector will include another L-2 size vector
        for n in range(L):
            N_n = N[n]
            if n == l:
                pass # this ensures we use n != l
            else:
                T2n[n] = (2/(N_l**2)) * ( (EP2P[l,k,z]* EP[n,z] - EPP[l,k,k]*EP[l,z]*EP[n,z]) + \
                                         2*(N_l - 1) * (EPP[l,z,k]*EP[l,k]*EP[n,z] - EP[l,k]*EP[l,z]*EP[l,k]*EP[n,z]))
                T3n[n] = (2/(N_l**2)) * ( (EP2P[l,z,k]* EP[n,k] - EPP[l,z,z]*EP[l,k]*EP[n,k]) + \
                                         2*(N_l - 1) * (EPP[l,k,z]*EP[l,z]*EP[n,k] - EP[l,z]*EP[l,k]*EP[l,z]*EP[n,k]))
                
                T4j = np.zeros((L,))
                for j in range(L):  # we cover the j != l,n loop first
                    if j == l:
                        pass
                    elif j == n:
                        pass
                    else:
                        T4j[j] = (2/N_l) * (EPP[l,k,z]*EP[n,k]*EP[j,z] - EP[l,k]*EP[l,z]*EP[n,k]*EP[j,z]) + \
                        (2/N_n) * (EPP[n,k,z]*EP[l,k]*EP[j,z] - EP[n,k]*EP[n,z]*EP[l,k]*EP[j,z])
                
                T4n[n] = (2/N_n/N_l) * ((N_n - 1)*(EPP[l,k,z]*EP[n,k]*EP[n,z] - EP[l,k]*EP[l,z]*EP[n,k]*EP[n,z]) + \
                                        (N_l - 1)*(EPP[n,k,z]*EP[l,k]*EP[l,z] - EP[l,k]*EP[l,z]*EP[n,k]*EP[n,z]) + \
                                        (EPP[l,k,z] * EPP[n,k,z] - EP[l,k]*EP[l,z]*EP[n,k]*EP[n,z])) + np.sum(T4j)
        
        Term_two[l] = np.sum(T2n)
        Term_three[l] = np.sum(T3n)
        Term_four[l] = np.sum(T4n)
        
    #ML_cov_kz = (1 / (polynomial_norms[k]**2 * polynomial_norms[z]**2) )  * \
    #( np.sum(Term_one) + np.sum(Term_two) + np.sum(Term_three) +np.sum(Term_four) ) 
    
    # or you can separate the variances into contributions by level
    ML_cov_kz = (1 / (polynomial_norms[k]**2 * polynomial_norms[z]**2) ) * (Term_one + Term_two + Term_three +Term_four) 
        
    return ML_cov_kz

def evaluate_ml_covariance_symbolic(k, z, moments, MLM, N, polynomial_norms):
    '''Computes the ML covariance of (\hat{beta}_k)^2 and (\hat{beta}_z)^2 using the moments of Q.
    The moments dictionary stores the necessary moments of Q and Psi from the evaluate_moments function. 
    Using moments will allow us to avoid recomputing them each time, just look at relevant indices.
    The following covariance formula is derived fully in the appendix to the proceedings article.
    Implemented by Michael Merritt, September 2020.'''
    L = MLM.levels
    Term_one = 0; Term_two = 0; 
    Term_three = 0; Term_four = 0;
    EP = moments['EP']   # unpack the moments dictionary
    EPP = moments['EPP']
    EP2P2 = moments['EP2P2']
    EP2P = moments['EP2P']
    for l in range(L):
        N_l = N[l]
        
        T1a = EP2P2[l,k,z] - (EPP[l,k,k]*EPP[l,z,z])
        T1b = (EP2P[l,k,z]*EP[l,z]) - (EPP[l,k,k]*(EP[l,z]**2))
        T1c = (EP2P[l,z,k]*EP[l,k]) - (EPP[l,z,z]*(EP[l,k]**2))
        T1d = EPP[l,k,z]**2
        T1e = EPP[l,k,z]*EP[l,k]*EP[l,z]
        T1f = (EP[l,k]**2)*(EP[l,z]**2)
        
        Term_one =  Term_one + (1/N_l**3) * ( T1a + (2*N_l-2)*T1b + (2*N_l-2)*T1c + \
                       (2*N_l-2)*T1d + 4*(N_l-1)*(N_l-2)*T1e - (4*N_l**2 - 10*N_l + 6)*T1f )
    
        T2n = 0 # reset the Term 2 and 3 vectors
        T3n = 0
        T4n = 0 # this term 4 vector will include another L-2 size vector
        for n in range(L):
            N_n = N[n]
            if n == l:
                pass # this ensures we use n != l
            else:
                T2n = T2n + (2/(N_l**2)) * ( (EP2P[l,k,z]* EP[n,z] - EPP[l,k,k]*EP[l,z]*EP[n,z]) + \
                                         2*(N_l - 1) * (EPP[l,z,k]*EP[l,k]*EP[n,z] - EP[l,k]*EP[l,z]*EP[l,k]*EP[n,z]))
                T3n = T3n + (2/(N_l**2)) * ( (EP2P[l,z,k]* EP[n,k] - EPP[l,z,z]*EP[l,k]*EP[n,k]) + \
                                         2*(N_l - 1) * (EPP[l,k,z]*EP[l,z]*EP[n,k] - EP[l,z]*EP[l,k]*EP[l,z]*EP[n,k]))
                
                T4j = 0
                for j in range(L):  # we cover the j != l,n loop first
                    if j == l:
                        pass
                    elif j == n:
                        pass
                    else:
                        T4j = T4j + (2/N_l) * (EPP[l,k,z]*EP[n,k]*EP[j,z] - EP[l,k]*EP[l,z]*EP[n,k]*EP[j,z]) + \
                            (2/N_n) * (EPP[n,k,z]*EP[l,k]*EP[j,z] - EP[n,k]*EP[n,z]*EP[l,k]*EP[j,z])
                
                T4n = T4n + (2/N_n/N_l) * ((N_n - 1)*(EPP[l,k,z]*EP[n,k]*EP[n,z] - EP[l,k]*EP[l,z]*EP[n,k]*EP[n,z]) + \
                                        (N_l - 1)*(EPP[n,k,z]*EP[l,k]*EP[l,z] - EP[l,k]*EP[l,z]*EP[n,k]*EP[n,z]) + \
                                        (EPP[l,k,z] * EPP[n,k,z] - EP[l,k]*EP[l,z]*EP[n,k]*EP[n,z])) + T4j
        
        Term_two = Term_two + T2n
        Term_three = Term_three + T3n
        Term_four = Term_four + T4n

    ML_cov_kz = (1 / (polynomial_norms[k]**2 * polynomial_norms[z]**2) ) * (Term_one + Term_two + Term_three +Term_four) 
        
    return ML_cov_kz

def evaluate_moments(function_evals, sampled_polynomials, MLM):
    '''Uses the difference function evaluations and polynomial evaluations to estimate the quantities
    necessary for computing the ML variance and covariance terms.
    Implemented by Michael Merritt, September 2020.'''
    L = MLM.levels
    P = len(sampled_polynomials) 
    
    EP = np.zeros((L,P)) # matrix of first moments where EP_{lk} = E[P_{lk}]
    EPP = np.zeros((L,P,P)) # tensor where EPP_{lkz} = E[P_{lk} P_{lz}]
    EP2P2 = np.zeros((L,P,P)) # tensor where EP2P2_{lkz} = E[P_{lk}^2 P_{lz}^2]
    EP2P = np.zeros((L,P,P)) # tensor where EP2P_{lkz} = E[P_{lk}^2 P_{lz}]
    
    for l in range(L):
        for k in range(P):
            EP[l,k] = np.mean(function_evals[l]*sampled_polynomials[k][l])
            
            for z in range(P):
                EPP[l,k,z] = np.mean(function_evals[l]**2*sampled_polynomials[k][l]*sampled_polynomials[z][l])
                EP2P2[l,k,z] = np.mean(function_evals[l]**4*sampled_polynomials[k][l]**2*sampled_polynomials[z][l]**2)
                EP2P[l,k,z] = np.mean(function_evals[l]**3*sampled_polynomials[k][l]**2*sampled_polynomials[z][l])
    
    
    moments = {"EP": EP, "EPP": EPP, "EP2P2":EP2P2, "EP2P":EP2P}
    
    return moments

# Here we define a function that extracts the necessary PCE coefficients

def extract_coefs_4gsa(subset, multiindices):
    '''Given a subset of indices in the the parameter vector ,u , this routine extracts the relevant
    PCE coefficients needed to approximate the variance of the Sobol index w.r.t u, i.e. Var[S_u]
    and Var[T_u]. The indices of these PCE coefficients are then returned.
    Implemented by Michael Merritt, September 2020.'''
    #multiindices = p.EvalMulti_totdeg(variables,total_order)
    # We want any index that contains just parameters in u, a main effect
    P = multiindices.shape[1]
    main_u = [[]]*P
    for i in range(P):
        main_u[i] = (np.flatnonzero(multiindices[:,i]).tolist() == subset)
    main_u_idx = np.squeeze(np.nonzero(main_u))  # numerical indices of PCE terms we want
    # then the relevant multiindices are:
    #print('first order MIs: '); print(multiindices[:,main_u])

    # Then we want any index that contains parameters in u, the total effect
    total_u = (np.prod(multiindices[subset,:] > 0, 0) == 1)
    total_u_idx = np.squeeze(np.nonzero(np.prod(multiindices[subset,:] > 0, 0) == 1))
    # then the relevant multiindices are:
    #print('total order MIs: '); print(multiindices[:,total_u]) # numerical indices of PCE terms we want
    
    return main_u_idx, total_u_idx


def estimate_sobol_var(subset_idx, VAR, COV, polynomial_norms):
    '''Given the ML variance vector and covariance matrix, this code sums together the 
    appropriate var and covar values, with the appropriate polynomial norms, to complete the 
    estimation of the variance of a particular Sobol index, say Var[S_u] or Var[T_u].
    You can either specify the coefficient indices necessary for a main effect or total index.
    Implemented by Michael Merritt, September 2020.'''
    Var_subset_u = 0
    #print(subset_idx, 'with ',subset_idx.shape, ' terms: ',)
    if subset_idx.size > 1:
        for i in subset_idx:
            Var_subset_u += VAR[i]*polynomial_norms[i]**2
            for j in subset_idx:
                if i == j:
                    pass
                else:
                    Var_subset_u += COV[i,j]*(polynomial_norms[i] * polynomial_norms[j]) 
    elif subset_idx.size == 1:
        Var_subset_u = VAR[subset_idx]*polynomial_norms[subset_idx]**2
    else:
        print('invalid subset of parameters')
    
    return Var_subset_u

def estimate_sobol_var_wcovar(subset_idx, COV, polynomial_norms):
    '''Given just the ML covariance matrix, this code sums together the 
    appropriate terms, with the appropriate polynomial norms, to complete the 
    estimation of the variance of a particular Sobol index, say Var[S_u] or Var[T_u].
    You can either specify the coefficient indices necessary for a main effect or total index.
    Retutns the results level by level, for use in a worst-case optimization approach.
    DISCLAIMER: here, there is no guarantee the diagonals of the covariance equal variance for small N.
    Implemented by Michael Merritt, December 2020.'''
    Var_subset_u_bylevel = np.zeros((COV.shape[0],)) # variance by level
    #print(subset_idx, 'with ',subset_idx.shape, ' terms: ',)
    for k in range(COV.shape[0]):
        VAR = np.diag(COV[k,:,:])
        if subset_idx.size > 1:
            for i in subset_idx:
                Var_subset_u_bylevel[k] += VAR[i]*polynomial_norms[i]**2
                for j in subset_idx:
                    if i == j:
                        pass
                    else:
                        Var_subset_u_bylevel[k] += COV[k,i,j]*(polynomial_norms[i] * polynomial_norms[j]) 
        elif subset_idx.size == 1:
            Var_subset_u_bylevel[k] = VAR[subset_idx]*polynomial_norms[subset_idx]**2
        else:
            print('invalid subset of parameters')
    
    return Var_subset_u_bylevel

# THIS FUNCTION IS NOT COMPLETED, but here is a first implementation with a few extra bells and whistles
def evaluate_var_sobol(N_by_level, *args):
    '''Evaluates the function that estimates the variance of a particular Sobol index, given sampling 
    conditions. This is the objective function to be passed to an optimization routine, resulting
    in optimal sampling conditions, targeting a variance reduction for a particular Sobol index.
    The subset argument must either be in a list or list of lists format. 
    Originally implemented by Michael Merritt, September 2020. Updated version, *date*.'''
    # Here we unpack the arguments and initialize variable
    MLM = args[0]; moments = args[1] # dictionary of all moment estimations
    subsets = args[2] # the variable or set of variables we target for GSA
    multiindices = args[3]
    targets = len(subsets)
    #if subsets == 'full variance':
    #    targets = 1
    
    #P = int(np.math.factorial(MLM.variables+total_order)/np.math.factorial(MLM.variables)/np.math.factorial(total_order))
    P = moments['EP'].shape[1]; L = len(N_by_level)
    #multiindices = p.EvalMulti_totdeg(MLM.variables,total_order)
    polynomial_norms = p.get_polynomial_norms(multiindices, MLM.basis)
    VAR = np.zeros((P,)); COV_bylevel = np.zeros((L,P,P))
    Costs = np.append(0,MLM.costs[:-1]) + MLM.costs
    Var_Sobol = np.zeros((L,targets)) # gives the variance of each GSA target by level
    
    # Now we build out the ML variance and covariance estimators
    for i in range(P):
        VAR[i] = evaluate_ml_variance(i, moments, MLM, N_by_level, polynomial_norms)
        for j in range(P):
            # computes the covariances over levels
            COV_bylevel[:,i,j] = evaluate_ml_covariance(i, j, moments, MLM, N_by_level, polynomial_norms)
    # Now we extract the proper PCE coefficents for GSA, or groups of coefficients for multiple GSA targets
    pce_indices_S, pce_indices_T = extract_multiple_gsa_coefs(subsets, multiindices)
    #print('PCE terms: ',pce_indices_S)
    # Now we build the ML varinace and covariance estimators

    # Evaluate the Var[S_u] term for each given subset, WE PROVIDE BELOW 2 OPTIONS: WC and Total
    for i in range(targets):
        # THIS LINE IS TEMPORARY! We need a variance that can be decomposed by levels
        #Var_Sobol[:,i] = estimate_sobol_var_wcovar(pce_indices_S[i], COV_bylevel, polynomial_norms)
        # This option does not break up the variance by level, but gives the entire thing
        Var_Sobol[0,i] = estimate_sobol_var(pce_indices_S[i], VAR, np.sum(COV_bylevel,0), polynomial_norms)
    #print('Current objective value: ',Var_Sobol)

    return np.max(Var_Sobol,0) #np.log10(np.max(Var_Sobol,0)) # this helps with the optimization performance

def extract_multiple_gsa_coefs(idx, multiindices):
    '''Set of multiple GSA target indices must be passed as a nested list. A list of arrays will be returned, 
    containing the PCE coefficents relevant for each Sobol index, both main effect and total.
    Implemented by Michael Merritt, December 2020.'''
    #variables = MLM.variables
    ndim, P = multiindices.shape
    pce_indices_S = [[]]*len(idx)
    pce_indices_T = [[]]*len(idx)

    for i in range(len(idx)):
        if idx[i] == 'full variance':
            pce_indices_S[i] = np.arange(1,P)
            pce_indices_T[i] = np.arange(1,P)
        else:
            pce_indices_S[i], pce_indices_T[i] = extract_coefs_4gsa(idx[i], multiindices)
    
    return pce_indices_S, pce_indices_T

def constraint_var_sobol(N, *args):
    '''The constraint function for optimizing cost, constrained by a target variance. This version requires an 
    accuracy, given by epsilon, which will then my apportioned over the possibly-multiple GSA targets. See the
    documentation for the formal definition of the constrained optimization problem.
    Implememted by Michael Merritt, June 2021.'''
    eps = args[0]; MLM = args[1]; moments = args[2]; gsa_targets = args[3]; multiindices = args[4]
    method = args[5]
    var = evaluate_var_sobol(N, MLM, moments, gsa_targets, multiindices)
    if method == 'partial var': # each component of eps is scaled by its contribution to V[V[Q]]
        constraint = var * (eps / np.sum(var) - 1) # percentage of variance to each index for splitting epsilon
    elif method == 'variance': # each component of the variance must be reduced below eps
        constraint = eps - var
    elif method == 'norm var': # each component of the normalized variance (by the mean) must be reduced below eps
        Sobol_main, Sobol_total = get_sobol_indices(MLM, moments, gsa_targets, multiindices)
        constraint = eps - var/Sobol_main 
    elif method == 'sobol norm': # each component of eps is scaled by its expected value for that Sobol index
        Sobol_main, Sobol_total = get_sobol_indices(MLM, moments, gsa_targets, multiindices)
        delta = var / Sobol_main
        constraint = delta * (eps / np.sum(delta) - 1) # percentage of scaled variance to each index for splitting epsilon
    elif method == 'cons_1':
        constraint = eps - var # each target must be less than eps
    elif method == 'cons_2':
        constraint = np.sum(eps - var) # sum of targets must be less than eps
        
    
    return constraint

def get_sobol_indices(MLM, moments, gsa_targets, multiindices, norms):
    '''Given estimated moment dictionary, a set of multiindices, a set of GSA targets, and the necessary 
    polynomial norms, this code will compute the Sobol indices (NORMALIZED BY VARIANCE) and variance.
    Implemented by Michael Merritt, June 2021.'''
    #norms = p.get_polynomial_norms(multiindices, MLM.basis)
    pce_indices_S, pce_indices_T = extract_multiple_gsa_coefs(gsa_targets, multiindices)
    beta_estim = np.sum(moments['EP'],0) / norms # the estimated PCE coefficients from sampling
    Sobol_estim_main = np.zeros((len(gsa_targets),)); Sobol_estim_total = np.zeros((len(gsa_targets),)); idx = 0
    for i in pce_indices_S:
        Sobol_estim_main[idx] = np.sum(beta_estim[i]**2 * norms[i])
        idx += 1
    idx = 0
    for j in pce_indices_T:
        Sobol_estim_total[idx] = np.sum(beta_estim[j]**2 * norms[j])
        idx += 1
    Variance = np.sum(beta_estim[1:]**2 * norms[1:])
          
    return Sobol_estim_main/Variance, Sobol_estim_total/Variance, Variance

def get_sobol_frombeta(beta, gsa_targets, multiindices, norms):
    '''Given estimated PCE coefficients, a set of GSA targets, multiindices and norms, this code will 
    compute the Sobol indices (NORMALIZED BY VARIANCE) and the variance. 
    Implemented by Michael Merritt, June 2021.'''
    pce_indices_S, pce_indices_T = extract_multiple_gsa_coefs(gsa_targets, multiindices)
    Sobol_S = np.zeros((len(gsa_targets),)); Sobol_T = np.zeros((len(gsa_targets),))
    for i in range(len(gsa_targets)):
        Sobol_S[i] = np.sum(beta[pce_indices_S[i]]**2 * norms[pce_indices_S[i]])
        Sobol_T[i] = np.sum(beta[pce_indices_T[i]]**2 * norms[pce_indices_T[i]])
    Variance = np.sum(beta[1:]**2 * norms[1:])
    
    return Sobol_S/Variance, Sobol_T/Variance, Variance

def decompose_variance(strategy, multiindices, *args):
    '''Based on specific use cases, this code builds the terms necessary for the variance of variance
    decomposition, neglecting specific covariance terms in order to reduce noise. Cases are not comprehensive.
    Implemented by Michael Merritt, August 2021.'''
    ndim, P = multiindices.shape
    if strategy == 'selected target':
        variance_targets = [[args[0]]]
    elif strategy == 'all first order':
        variance_targets = [[i] for i in list(range(0,ndim))]
    elif strategy == 'interaction order':
        order = args[0]
        variance_targets = [list(k) for k in combinations(list(range(0,ndim)), order)]
    elif strategy == 'all interactions':
        variance_targets = []
        for i in range(2,P):
            variance_targets = variance_targets + [list(k) for k in combinations(list(range(0,ndim)), i)]
    elif strategy == 'all total':
        variance_targets = [[i] for i in list(range(0,ndim))]
        # Need option here to trigger total index coefficients
    elif strategy == 'full variance':
        variance_targets = ['full variance']
    else:
        print('Need valid variance target')
        variance_target = 'NA'
    
    return variance_targets

def sobol_drop_tolerance(tolerance, Sobol, Var_Sobol, Variance, variance_targets):
    '''This conditional function either keeps or drops a variance target depending on the magnitude of 
    the Sobol index and its confidnce interval. Unimportant indices get removed in returned list of targets.
    Implemented by Michael Merritt, September 2021.'''
    variance_targets_new = []
    for i in range(len(variance_targets)):
        if Sobol[i] + 2*np.sqrt(Var_Sobol[i])/np.sqrt(Variance) > tolerance: # keep if 2 sigma CI is above tolerance
            variance_targets_new.append(variance_targets[i])
    
    return variance_targets_new

#
#
# This is an old function, may still be useful to keep it around. - 7/20/20
#
def estimate_kth_pce_coef_OLDVERSION(k, N, MLM):
    '''This code estimates the kth PCE coefficient using a D level MLMC scheme. 
    The essential inputs are the index of the coefficient k, the total number of samples 
    available N, a multi level function class with the other relevant pieces.'''
    
    A, B = MLM.basis_params # these can be uniform bounds or normal mu and sigma
    vars = MLM.variables # this needs to be variable, determined by function
    D = MLM.levels
    C = MLM.costs
    N_by_level = np.zeros(D)
    
    # Initialization: set up the PCE variables and the full set of input samples
    multiindices = p.EvalMulti_totdeg(vars,k) # must be a better way for this
    P = multiindices.shape[1]
    if MLM.basis == 'legendre':
        samples_on_ab = rnd.uniform(A, B, (vars,N))
        samples = p.transf_uniform(samples_on_ab, A, B) 
    elif MLM.basis == 'hermite':
        samples_on_ab = rnd.normal(A, B, (vars,N))
        samples = p.transf_normal(samples_on_ab, A, B)
    sampled_polynomials = np.zeros((N,)) # just sampling the kth PCE polynomial
    for i in range(N):
        sampled_polynomials[i] = p.evaluate_orthogonal_polynomials(multiindices[:,k], samples[:,i], MLM.basis)
    polynomial_norm = p.get_polynomial_norms(multiindices[:,k], MLM.basis)
        
    N_test = int(N/10) # let these samples be used to estimate the necessary variances
    function_evals = np.zeros((N,D))
    var_sampled = np.zeros((D,))
    
    # this evaluates the pilot samples to estimate the variance for sample allocation
    for i in range(MLM.levels):
        if i == 0:
            function_evals[:N_test,i] = MLM.functions[i](samples_on_ab[:,:N_test])
        else: 
            function_evals[:N_test,i] = MLM.functions[i](samples_on_ab[:,-N_test:]) \
            - MLM.functions[i-1](samples_on_ab[:,-N_test:])
        var_sampled[i] = np.var(function_evals[:N_test,i])
    function_evals = np.zeros((N,D))
      
    # the approach here is to work within a fixed computational budget, not a desired variance
    mu = np.sum(np.sqrt(var_sampled * C))
    N_by_level = mu*np.sqrt(var_sampled / C)
    total_N_by_level = (N * np.round_(N_by_level / np.sum(N_by_level), 3)).astype(int)
    # then we partition the sampling indices according to variance estimation
    idx = np.zeros((MLM.levels, 2)) # 
    for i in range(MLM.levels):
        idx[i,:] = [np.sum(total_N_by_level[:i]), np.sum(total_N_by_level[:i+1])]
    idx = idx.astype(int)
    
    # this is where we sample the ML QoI and compute PCE projections
    numerator = 0
    for i in range(MLM.levels):
        if i == 0:
            function_evals[idx[i,0]:idx[i,1],i] = MLM.functions[i](samples_on_ab[:,idx[i,0]:idx[i,1]])
            numer_update = np.sum(function_evals[:,i] * sampled_polynomials)
        else:
            function_evals[idx[i,0]:idx[i,1],i] = MLM.functions[i](samples_on_ab[:,idx[i,0]:idx[i,1]]) \
            - MLM.functions[i-1](samples_on_ab[:,idx[i,0]:idx[i,1]])                     
            numer_update = np.sum(function_evals[:,i] * sampled_polynomials)
        numerator += ((numer_update) / total_N_by_level[i])
    
    # Postprocessing steps
    beta_k = numerator / polynomial_norm
    total_cost = np.sum(C*total_N_by_level)  # this is currently unused, can be useful though
    if 'n/a' in MLM.var_analytical:
        print('cannot determine optimal sampling analytically')
    else: 
        optimal_sampling = np.sum(np.sqrt(MLM.var_analytical * C)) * np.sqrt(MLM.var_analytical / C)
        print('optimal sampling ratio: ',optimal_sampling)
 
    return function_evals, beta_k, sampled_polynomials, total_N_by_level
