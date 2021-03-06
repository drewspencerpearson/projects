# name this file 'solutions.py'
"""Volume II Lab 13: Optimization Packages I (scipy.optimize)
Drew Pearson
Math 323
January 7th
"""
import numpy as np
from scipy import optimize as opt
from matplotlib import pyplot as plt

def prob1():
    """Use the minimize() function in the scipy.optimize package to find the
    minimum of the Rosenbrock function (scipy.optimize.rosen) using the
    following methods:
        Nelder-Mead
        Powell
        CG
        BFGS
        Newton-CG (test with and without the hessian)
        Anneal
        L-BFGS-B
        TNC
        COBYLA
        SLSQP
    Use x0 = np.array([4., -2.5]) for the initial guess for each test.
    
    Print a statement answering the following questions:
        Which algorithm(s) take(s) the least number of iterations?
        Which algorithm(s) fail to find the (correct) minimum?
    """
    x0 = np.array([4.,-2.5])
    my_list = ["nelder-mead", "powell", "cg", "bfgs", "l-bfgs-b", 'tnc', 'cobyla', 'slsqp']
    #for i in my_list:
        #print "for " + i
        #print opt.minimize(opt.rosen, x0, method = i, options= {'xtol':1e-8})
    #print "for newton-cg"
    #print opt.minimize(opt.rosen, x0, jac=opt.rosen_der, method = "newton-cg", hess = opt.rosen_hess, options= {'xtol':1e-8})
    print 'Powell algorithm used the least amount of iterations (19)'
    print 'cobyla algorithm failed to find the correct minimum'

def prob2():
    """Explore the documentation on the function scipy.optimize.basinhopping()
    online or via IPython. Use it to find the global minimum of the multmin()
    function given in the lab, with initial point x0 = np.array([-2, -2]) and
    the Nelder-Mead algorithm. Try it first with stepsize=0.5, then with
    stepsize=0.2.

    Return the minimum value of the function with stepsize=0.2.
    Print a statement answering the following question:
        Why doesn't scipy.optimize.basinhopping() find the minimum the second
        time (with stepsize=0.2)?
    """
    x0 = np.array([-2,-2])
    def multmin(x):
        r = np.sqrt((x[0]+1)**2+x[1]**2)
        return r**2 *(1+np.sin(4*r)**2)
    print "for stepsize 0.2, the algorithm fails to find the minimum because the step size is not large enough. Meaning the step size does not bring the function outside of it's current trough. So it keeps finding the same minimum, which is just a local min. Not a global min."
    #print opt.basinhopping(multmin, x0, stepsize = 0.5, minimizer_kwargs={'method':'nelder-mead'})
    #print "for stepsize 0.2"
    solution=opt.basinhopping(multmin, x0, stepsize = 0.2, minimizer_kwargs={'method':'nelder-mead'})
    return solution.fun    

def prob3():
    """Find the roots of the system
    [       -x + y + z     ]   [0]
    [  1 + x^3 - y^2 + z^3 ] = [0]
    [ -2 - x^2 + y^2 + z^2 ]   [0]

    Returns the values of x,y,z as an array.
    """
    def func(x):
        return [-x[0]+x[1]+x[2], 1+x[0]**3-x[1]**2+x[2]**3, -2-x[0]**2+x[1]**2+x[2]**2]
    def jac(x):
        A = np.array([[-1,3*x[0]**2,-2*x[0]],[1,-2*x[1],2*x[1]],[1,3*x[2]**2,2*x[2]]])
        return A.T
    sol = opt.root(func, [0,0,0], jac = jac, method = 'hybr')
    return sol.x

def prob4():
    """Use the scipy.optimize.curve_fit() function to fit a heating curve to
    the data found in `heating.txt`. The first column of this file is time, and
    the second column is temperature in Kelvin.

    The fitting parameters should be gamma, C, and K, as given in Newton's law
    of cooling.

    Plot the data from `heating.txt` and the curve generated by curve_fit.
    Return the values gamma, C, K as an array.
    """
    data = np.loadtxt("heating.txt")
    time = data[:,0]
    temp = data[:,1]

    def func(time, gamma, C, K):
        return 290 + 59.43/gamma +K*np.exp(-gamma*time/C)

    popt, pcov = opt.curve_fit(func, time, temp)

    plt.scatter(time, temp, s=2)
    x = np.linspace(0,400)
    y = func(x,popt[0],popt[1],popt[2])
    plt.plot(x,y)
    plt.show()

    return popt
    #return pcov





if __name__ == '__main__':

    print prob2()
    #prob3()
    #print prob4()
    #prob1()


