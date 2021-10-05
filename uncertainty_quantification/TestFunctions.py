import numpy as np

__all__ = ['Ishigami', 'rosenbrock','Ishigami_Dakota','Mike_fun', 'Anisotropic_fun','Twolevel_Ani_fun']

def Ishigami(x,a=7.0,b=0.1):
    "optional arguments for parameters a and b, and dimension is ndim = 3"
    return np.sin(x[0]) + a*np.sin(x[1])**2 + b*x[2]**4*np.sin(x[0])

def Mike_fun(x,a=1.0,b=1.0):
    "optional arguments for parameters a and b, and dimension is ndim = 3"
    # a*x[0]*x[2] - 1.5*b*x[1]**2 + 3.0*x[1] + 3.0*x[2]**2 - 2.0*x[2] + 4.5 # original version 
    #7.5*x_1**2 - 4.0*x_1*x_2 + 3.0*x_1*x_3 - 8.0*x_1 - 3.0*x_2**2 + 1.0*x_2*x_3 + 7.0*x_2 - 1.5*x_3**2 - 6.0*x_3 + 8.0 #new ver
    #return a*(1.5*x[0]**2 + x[0]*x[1] + x[0]*x[2]) + x[0] + b*(-1.5*x[1]**2 + x[1]*x[2] + 3.0*x[1]) + 3.0*x[2]**2 - 2.0*x[2]+4.0
    return a*(7.5*x[0]**2 - 4*x[0]*x[1] + 3*x[0]*x[2] - 8*x[0] - 3*x[1]**2 + 1*x[1]*x[2] + 7*x[1]) - 1.5*x[2]**2 - 6*x[2] + 8

def Anisotropic_fun(x, level=0):
    "Three level anisotropic function built from a PCE with known coefficients. Variable contributions are added by level"
    if level == 0:
        ani_fun = 6.75*x[0]**2-1.8*x[0]*x[1]+1.35*x[0]*x[2]-7.2*x[0]-1.5*x[1]**2+.25*x[1]*x[2]+3.5*x[1]-.75*x[2]**2-3*x[2]+7.5
    elif level == 1:
        ani_fun = 6.75*x[0]**2-3.24*x[0]*x[1]+1.35*x[0]*x[2]-7.2*x[0]-2.7*x[1]**2 +.45*x[1]*x[2]+6.3*x[1]-.75*x[2]**2-3*x[2]+7.9
    elif level == 2:
        ani_fun = 7.5*x[0]**2 - 4*x[0]*x[1] + 3*x[0]*x[2] - 8*x[0] - 3*x[1]**2 + 1*x[1]*x[2] + 7*x[1] - 1.5*x[2]**2 - 6*x[2] + 8

    return ani_fun

def Twolevel_Ani_fun(x, b1, b2, b3, b4, b5, level=0):
    "Anisotropic function of 2 variables and terms up to total order 2 with 2 levels"
    if level == 0:
        fun = b1 * x[0] + b3 * x[0] * x[1] + b4 * (3*x[0]**2 - 1)/2
    elif level == 1:
        fun = b1 * x[0] + b2 * x[1] + b3 * x[0] * x[1] + b4 * (3*x[0]**2 - 1)/2 + b5 * (3*x[1]**2 - 1)/2
    else:
        print('give a valid level')
    return fun

def Twolevel_Ani_fun2(x, b1, b2, b3, b4, b5, level=0):
    "Anisotropic function of 2 variables and only linear terms with 2 levels"
    if level == 0:
        fun = b1*x[0]
    elif level == 1:
        fun = b1*x[0] + b2*x[1]
    else:
        print('give a valid level')
    return fun
    
def Sobol_Ishigami(a=7.0,b=0.1):
    "returns analytical values for Ishigami Sobol indices for U[-pi, pi]"
    variance = a**2/8 + b*np.pi**4/5 + b**2*np.pi**8/18 + 1/2
    S1 = (b*np.pi**4/5 + b**2*np.pi**8/50 + 1/2) / variance
    S2 = a**2/8/variance
    first_order = [S1, S2, 0]
    T1 = S1 + (b**2*np.pi**8/18 - b**2*np.pi**8/50)/variance
    total_indices = [T1, S1, (b**2*np.pi**8/18 - b**2*np.pi**8/50)/variance]
    return variance, first_order, total_indices

def rosenbrock(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

def Ishigami_Dakota(x):
    return np.sin(2*np.pi*x[0] - np.pi) + 7*np.sin(2*np.pi*x[1] - np.pi)**2 + 0.1*(2*np.pi*x[2] - np.pi)**4*np.sin(2*np.pi*x[0] - np.pi)

# Both of these functions assume C and C' are 1
def var_ishigami(a, b):
    return b**2*np.pi**8/18 + b*np.pi**4/5 + a**2/8 + 1/2

def var_correction(a, b, a_p, b_p):
    term1 = (3*np.pi**3*(a - a_p)**2 + 4*np.pi**11*(b - b_p)**2/9) / (8*np.pi**3)
    term2 = 16*np.pi**6*(a - a_p)**2 / (64*np.pi**6)
    return term1 - term2
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
