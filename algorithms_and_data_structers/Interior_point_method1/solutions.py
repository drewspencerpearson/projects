# Name this file 'solutions.py'.
"""Volume 2 Lab 19: Interior Point 1 (Linear Programs)
Drew Pearson
Math 322

"""

import numpy as np
from scipy import linalg as la
from scipy.stats import linregress
from matplotlib import pyplot as plt
from scipy.stats import linregress


# Auxiliary Functions ---------------------------------------------------------
def startingPoint(A, b, c):
    """Calculate an initial guess to the solution of the linear program
    min c^T x, Ax = b, x>=0.
    Reference: Nocedal and Wright, p. 410.
    """
    # Calculate x, lam, mu of minimal norm satisfying both
    # the primal and dual constraints.
    B = la.inv(A.dot(A.T))
    x = A.T.dot(B.dot(b))
    lam = B.dot(A.dot(c))
    mu = c - A.T.dot(lam)

    # Perturb x and s so they are nonnegative.
    dx = max((-3./2)*x.min(), 0)
    dmu = max((-3./2)*mu.min(), 0)
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    # Perturb x and mu so they are not too small and not too dissimilar.
    dx = .5*(x*mu).sum()/mu.sum()
    dmu = .5*(x*mu).sum()/x.sum()
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    return x, lam, mu

# Use this linear program generator to test your interior point method.
def randomLP(m):
    """Generate a 'square' linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add slack variables.
    Inputs:
        m -- positive integer: the number of desired constraints
             and the dimension of space in which to optimize.
    Outputs:
        A -- array of shape (m,n).
        b -- array of shape (m,).
        c -- array of shape (n,).
        x -- the solution to the LP.
    """
    n = m
    A = np.random.random((m,n))*20 - 10
    A[A[:,-1]<0] *= -1
    x = np.random.random(n)*10
    b = A.dot(x)
    c = A.sum(axis=0)/float(n)
    return A, b, -c, x

# This random linear program generator is more general than the first.
def randomLP2(m,n):
    """Generate a linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add
    slack variables to convert it into the above form.
    Inputs:
        m -- positive integer >= n, number of desired constraints
        n -- dimension of space in which to optimize
    Outputs:
        A -- array of shape (m,n+m)
        b -- array of shape (m,)
        c -- array of shape (n+m,), with m trailing 0s
        v -- the solution to the LP
    """
    A = np.random.random((m,n))*20 - 10
    A[A[:,-1]<0] *= -1
    v = np.random.random(n)*10
    k = n
    b = np.zeros(m)
    b[:k] = A[:k,:].dot(v)
    b[k:] = A[k:,:].dot(v) + np.random.random(m-k)*10
    c = np.zeros(n+m)
    c[:n] = A[:k,:].sum(axis=0)/k
    A = np.hstack((A, np.eye(m)))
    return A, b, -c, v


# Problems --------------------------------------------------------------------
def interiorPoint(A, b, c, niter=20, tol=1e-16, verbose=False):
    """Solve the linear program min c^T x, Ax = b, x>=0
    using an Interior Point method.

    Parameters:
        A ((m,n) ndarray): Equality constraint matrix with full row rank.
        b ((m, ) ndarray): Equality constraint vector.
        c ((n, ) ndarray): Linear objective function coefficients.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """
    def create_F(x,lam, mu):
        M = np.diag(mu)
        #F = np.hstack((np.dot(A.T, lam)+mu-c,np.dot(A,x)-b,np.dot(M,x)))
        F = np.bmat((np.dot(A.T, lam)+mu-c,np.dot(A,x)-b,np.dot(M,x)))
        #print F
        return F
    def derivative(mu,x):
        M = np.diag(mu)
        #print A
        #print A.T
        m,n = np.shape(A)
        top = [np.zeros((n,n)), A.T, np.eye(n)]
        #print np.shape(top)
        middle = [A, np.zeros((m,m)), np.zeros((m,n))]
        #print np.shape(middle)
        bottom = [M, np.zeros((n,m)), np.diag(x)]
        #print np.shape(bottom)
        #print np.vstack((top,middle,bottom))
        return np.bmat([top,middle,bottom])
        #return np.vstack((top,middle,bottom))
    def calc_nu(x,mu):
        return np.dot(x.T,mu)/len(x)
    def search_direction(x, lam, mu):
        estimate = -1*create_F(x,lam,mu)
        v = calc_nu(x,mu)
        extra_piece = np.hstack((np.zeros(len(x)),np.zeros(len(lam)),v*.1*np.ones(len(mu))))
        b = (estimate+extra_piece).T
        soln = la.solve(derivative(mu,x),b)
        del_x = soln[:len(x)]
        del_lam = soln[len(x):len(x)+len(lam)]
        del_mu = soln[len(x)+len(lam):]
        return del_x, del_lam, del_mu
    def step_size(x, mu, del_x, del_mu):
        #del_x, del_lam, del_mu = search_direction(x,lam,mu)
        #print x, mu, del_x, del_mu
        coutner_x = x.reshape(len(x), 1)
        counter_mu = mu.reshape(len(mu), 1)
        mu_mask = del_mu < 0
        new_mu = counter_mu[mu_mask]
        new_del_mu = del_mu[mu_mask]

        x_mask = del_x < 0
        new_x = coutner_x[x_mask]
        new_del_x = del_x[x_mask]

        try:
            min_mues = min(-new_mu/new_del_mu)
        except:
            min_mues = 0
        try:
            min_xs = min(-new_x/new_del_x)
        except:
            min_xs = 0

        max_alpha = min(1,min_mues)
        max_delta = min(1,min_xs)

        alpha = min(1,.95*max_alpha)
        delta = min(1,.95*max_delta)
        return alpha, delta


    x, lam, mu = startingPoint(A,b,c)
    nu = calc_nu(x,mu)
    i = 0
    while i < niter and nu > tol:

        del_x,del_lam,del_mu = search_direction(x,lam,mu)
        #print del_x, del_lam, del_mu
        alpha, delta = step_size(x,mu, del_x, del_mu)
        del_x = del_x.reshape(len(x),)
        del_lam = del_lam.reshape(len(lam),)
        del_mu = del_mu.reshape(len(mu),)
        x = x+delta*del_x
        lam = lam+alpha*del_lam
        mu = mu+alpha*del_mu
        i +=1
        nu = calc_nu(x,mu)


    return x, np.dot(c.T,x)








def leastAbsoluteDeviations(filename='simdata.txt'):
    """Generate and show the plot requested in the lab."""
    data = np.loadtxt('simdata.txt')
    slope, intercept = linregress(data[:,1], data[:,0])[:2]
    domain = np.linspace(0,10,200)
    plt.plot(data[:, 1], data[:,0],"ro")
    plt.plot(domain, domain*slope + intercept)
    #plt.show()
    m = data.shape[0]
    n = data.shape[1] - 1
    c = np.zeros(3*m + 2*(n + 1))
    c[:m] = 1
    y = np.empty(2*m)
    y[::2] = -data[:, 0]
    y[1::2] = data[:, 0]
    x = data[:, 1:]
    A = np.ones((2*m, 3*m + 2*(n + 1)))
    A[::2, :m] = np.eye(m)
    A[1::2, :m] = np.eye(m)
    A[::2, m:m+n] = -x
    A[1::2, m:m+n] = x
    A[::2, m+n:m+2*n] = x
    A[1::2, m+n:m+2*n] = -x
    A[::2, m+2*n] = -1
    A[1::2, m+2*n+1] = -1
    A[:, m+2*n+2:] = -np.eye(2*m, 2*m)
    sol = interiorPoint(A, y, c, niter=10)[0]
    beta = sol[m:m+n] - sol[m+n:m+2*n]
    b = sol[m+2*n] - sol[m+2*n+1]
    plt.plot(domain, domain*beta+b)
    plt.show()


def test():
    #data = np.load('for_luke.npz')
    #A, b, c, x = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    #print A 
    #A,b,c,x = randomLP2(5,3)
    A,b,c,x = randomLP2(5,3)
    point, value = interiorPoint(A,b,c)
    print point, x
    print np.allclose(x,point[:3])
    #print np.allclose(x,value)


if __name__ == '__main__':
    test()
    #leastAbsoluteDeviations()
