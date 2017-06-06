# name this file 'solutions.py'.
"""Volume II Lab 15: Line Search Algorithms
Drew
Pearson
Jan 21, 2015
"""

import numpy as np
from scipy import linalg as la
import scipy.optimize as opt
from scipy.optimize import line_search as ls
from scipy.optimize import leastsq
from matplotlib import pyplot as plt


# Problem 1
def newton1d(f, df, ddf, x, niter=10):
    """
    Perform Newton's method to minimize a function from R to R.

    Parameters:
        f (function): The twice-differentiable objective function.
        df (function): The first derivative of 'f'.
        ddf (function): The second derivative of 'f'.
        x (float): The initial guess.
        niter (int): The number of iterations. Defaults to 10.
    
    Returns:
        (float) The approximated minimizer.
    """
    i = 1
    old = x
    while i <= niter:
        new = old - df(old)/float(ddf(old))
        #if abs(new-old) <=tol:
            #return (new, True, i)
        old = new
        i +=1
    return new

def test_newton():
    """Use the newton1d() function to minimixe f(x) = x^2 + sin(5x) with an
    initial guess of x_0 = 0. Also try other guesses farther away from the
    true minimizer, and note when the method fails to obtain the correct
    answer.

    Returns:
        (float) The true minimizer with an initial guess x_0 = 0.
        (float) The result of newton1d() with a bad initial guess.
    """
    f = lambda x: x**2 + np.sin(5*x)
    df = lambda x: 2*x + 5*np.cos(5*x)
    ddf = lambda x: 2 - 25*np.sin(5*x)
    answer = newton1d(f,df,ddf,0)
    bad_answer = newton1d(f,df,ddf,10)
    return answer,bad_answer


# Problem 2
def backtracking(f, slope, x, p, a=1, rho=.9, c=10e-4):
    """Perform a backtracking line search to satisfy the Armijo Conditions.

    Parameters:
        f (function): the twice-differentiable objective function.
        slope (float): The value of grad(f)^T p.
        x (ndarray of shape (n,)): The current iterate.
        p (ndarray of shape (n,)): The current search direction.
        a (float): The intial step length. (set to 1 in Newton and
            quasi-Newton methods)
        rho (float): A number in (0,1).
        c (float): A number in (0,1).
    
    Returns:
        (float) The computed step size satisfying the Armijo condition.
    """
    def funct(a):
        return f(x + a*p)
    def comparison(a):
        comp = f(x)+c*a*slope
        return comp   

    while funct(a) > comparison(a):
        a = rho*a
    return a



# Problem 3    
def gradientDescent(f, df, x, niter=10):
    """Minimize a function using gradient descent.

    Parameters:
        f (function): The twice-differentiable objective function.
        df (function): The gradient of the function.
        x (ndarray of shape (n,)): The initial point.
        niter (int): The number of iterations to run.
    
    Returns:
        (list of ndarrays) The sequence of points generated.
    """

    x_list = []
    i = 1
    while i <= niter:
        slope = np.dot(df(x), -df(x))
        p = -df(x)
        a = backtracking(f,slope,x,p)
        x_new = x + a*p
        x_list.append(x_new)
        x = x_new
        i +=1
    return x_list
def newtonsMethod(f, df, ddf, x, niter=10):
    """Minimize a function using Newton's method.

    Parameters:
        f (function): The twice-differentiable objective function.
        df (function): The gradient of the function.
        ddf (function): The Hessian of the function.
        x (ndarray of shape (n,)): The initial point.
        niter (int): The number of iterations.
    
    Returns:
        (list of ndarrays) The sequence of points generated.
    """
    i = 1
    x_list = []
    while i <= niter:
        p = np.dot(la.inv(-ddf(x)),df(x))
        slope = np.dot(df(x), p)
        a = backtracking(f,slope,x,p)
        x_new = x+a*p
        x_list.append(x_new)
        x = x_new
        i+=1
    return x_list


# Problem 4
def gaussNewton(f, df, jac, r, x, niter=10):
    """Solve a nonlinear least squares problem with Gauss-Newton method.

    Parameters:
        f (function): The objective function.
        df (function): The gradient of f.
        jac (function): The jacobian of the residual vector.
        r (function): The residual vector.
        x (ndarray of shape (n,)): The initial point.
        niter (int): The number of iterations.
    
    Returns:
        (ndarray of shape (n,)) The minimizer.
    """
    for i in xrange(niter):
        try:
            p = la.solve(np.dot(jac(x).T,jac(x)),np.dot(-jac(x).T,r(x)))
            a = ls(f,df,x,p)[0]
            x = x +a*p
        except:
            true 
    return x

# Problem 5
def census():
    """Generate two plots: one that considers the first 8 decades of the US
    Census data (with the exponential model), and one that considers all 16
    decades of data (with the logistic model).
    """

    # Start with the first 8 decades of data.
    years1 = np.arange(8)
    pop1 = np.array([3.929,  5.308,  7.240,  9.638,
                    12.866, 17.069, 23.192, 31.443])

    # Now consider the first 16 decades.
    years2 = np.arange(16)
    pop2 = np.array([3.929,   5.308,   7.240,   9.638,
                    12.866,  17.069,  23.192,  31.443,
                    38.558,  50.156,  62.948,  75.996,
                    91.972, 105.711, 122.775, 131.669])

    def model(x,t):
        return x[0]*np.exp(x[1]*(t+x[2]))
    def residual(x):
        return model(x, years1)-pop1

    x0 = np.array([150, .4,2.5])
    x = opt.leastsq(residual, x0)[0]

    dom = np.linspace(0,8,1000)
    y = x[0]*np.exp(x[1]*(dom+x[2]))
    plt.plot(years1, pop1, '*')
    plt.plot(dom, y)
    plt.show()

    def model2(x,t):
        return float(x[0]) / (1+np.exp(-x[1]*(t +x[2])))
    def residual2(x):
        return model2(x,years2)-pop2
    x0 = np.array([150,.4,-15])
    x2 = opt.leastsq(residual2, x0)[0]

    d2 = np.linspace(0,16,1000)
    y2 = model2(x2,d2)
    plt.plot(years2,pop2, '*')
    plt.plot(d2, y2)
    plt.show()


def test_3():
    f = lambda x: x**2 + np.sin(5*x)
    df = lambda x: 2*x + 5*np.cos(5*x)
    ddf = lambda x: 2 - 25*np.sin(5*x)
    print gradientDescent(f,df,0)

def test_3b():
    f = lambda x: x[0]**2 + np.sin(5*x[1])
    df = lambda x: np.array([2*x[0],5*np.cos(5*x[1])])
    ddf = lambda x: np.array([[2,0],[0,-25*np.sin(5*x[1])]])
    print newtonsMethod(f, df, ddf, np.array([0,np.pi]), niter=10)

if __name__ == '__main__':
    #print test_newton()
    #test_3b()
    census()




