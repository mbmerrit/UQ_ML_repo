import numpy as np
import numpy.matlib as npmat
import numpy.random as nprand


__all__ = ['get_sobol_indices','compute_indices']


# The main function for the SI calculator, calls to other lower level functions
def get_sobol_indices(f, x_nom, N, percent):
    """ 
    Given a scalar-valued function f, nominal parameters, number or samples, 
    and width of uniform sampling interval. Computation progress will be given during function
    evaluations. Computes Sobol' indices using the Saltelli sampling procedure with Monte Carlo samples.
    
    Make sure the function f can handle vector valued inputs in matrix form (multiple samples).
    
    N must be a integer, percent should be kept in the interval (0,100]
    """
    
    p = len(x_nom) # number of parameters
    a = x_nom - (percent/100)*x_nom
    b = x_nom + (percent/100)*x_nom             # this block computes the matrices of parameter samples
    XA = -1 + 2*nprand.rand(N, p)    
    XB = -1 + 2*nprand.rand(N, p)
    A = npmat.repmat(0.5*(a+b),N,1) + 0.5*(b-a)*XA
    B = npmat.repmat(0.5*(a+b),N,1) + 0.5*(b-a)*XB
    
    # Now we evaluate the function at the parameter samples
    qA = f(A)
    print('progress: ',(1/(p+2))*100,'%')
    qB = f(B)
    print('progress: ',(2/(p+2))*100,'%')
    qC = np.zeros([N,p])
    for k in range(p):  # this substitutes one column of B into each A, then evaluates
        C = np.copy(A);
        C[:,k] = B[:,k]  # here you are varying one parameter at a time
        qC[:,k] = f(C) 
        print('progress: ',((k+3)/(p+2))*100,'%')
        
    [S, T] = compute_indices(qA, qB, qC)
    
    return S, T   # Outputs are vectors of first order and total indices


def compute_indices(qA, qB, qC):
    """
    This function takes parameter samples from function evalutations and uses them 
    to estimate Sobol' indices (first order and total)
    """
    
    [N,p] = qC.shape
    mu_qA = np.mean(qA)
    mu_qB = np.mean(qB)
    Var_Y = (1/2)*np.mean( (qA-mu_qA)**2 + (qB-mu_qB)**2 )

    # Estimate Sobol' indices
    S = np.zeros(p)
    T = np.zeros(p)
    for j in range(p):
        S[j] = np.mean( qB*(qC[:,j]-qA) )/Var_Y
        T[j] = (1/2)*np.mean( (qA - qC[:,j])**2 )/Var_Y 
    
    return S, T
