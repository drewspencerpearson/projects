# Name this file 'solutions.py'.
"""Volume II: Interior Point II (Quadratic Optimization).
Drew Pearson
Math 3223
March 31, 2015
"""

import numpy as np
from scipy import linalg as la
from scipy.sparse import spdiags
import cvxopt as opt

# Auxiliary function for problem 2
def startingPoint(G, c, A, b, guess):
    """
    Obtain an appropriate initial point for solving the QP
    .5 x^T Gx + x^T c s.t. Ax >= b.
    Inputs:
        G -- symmetric positive semidefinite matrix shape (n,n)
        c -- array of length n
        A -- constraint matrix shape (m,n)
        b -- array of length m
        guess -- a tuple of arrays (x, y, mu) of lengths n, m, and m, resp.
    Returns:
        a tuple of arrays (x0, y0, l0) of lengths n, m, and m, resp.
    """
    m,n = A.shape
    x0, y0, l0 = guess

    N = np.zeros((n+m+m, n+m+m))
    N[:n,:n] = G
    N[:n, n+m:] = -A.T
    N[n:n+m, :n] = A
    N[n:n+m, n:n+m] = -np.eye(m)
    N[n+m:, n:n+m] = np.diag(l0)
    N[n+m:, n+m:] = np.diag(y0)
    rhs = np.empty(n+m+m)
    rhs[:n] = -(G.dot(x0) - A.T.dot(l0)+c)
    rhs[n:n+m] = -(A.dot(x0) - y0 - b)
    rhs[n+m:] = -(y0*l0)

    sol = la.solve(N, rhs)
    dx = sol[:n]
    dy = sol[n:n+m]
    dl = sol[n+m:]

    y0 = np.maximum(1, np.abs(y0 + dy))
    l0 = np.maximum(1, np.abs(l0+dl))

    return x0, y0, l0

# Problems 1-2
def qInteriorPoint(Q, c, A, b, guess, niter=20, tol=1e-16):
    """Solve the Quadratic program min .5 x^T Q x +  c^T x, Ax >= b
    using an Interior Point method.

    Parameters:
        Q ((n,n) ndarray): Positive semidefinite objective matrix.
        c ((n, ) ndarray): linear objective vector.
        A ((m,n) ndarray): Inequality constraint matrix.
        b ((m, ) ndarray): Inequality constraint vector.
        guess (3-tuple of arrays of lengths n, m, and m): Initial guesses for
            the solution x and lagrange multipliers y and eta, respectively.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """
    m, n = A.shape
    def F(A, b, c, x, mu, Q, y):
        #print -np.dot(A.T, mu) + c +Q.dot(x)
        #print np.dot(A, x)
        #print y
        #print np.dot(A, x) - y -b
        return np.bmat((-np.dot(A.T, mu) + c +Q.dot(x), np.dot(A, x) - y -b, np.dot(np.diag(y),np.diag(mu)).dot(np.ones(m)))).T

    #create top half of our DF matrix
    top_matrix = np.bmat([[Q,np.zeros((n,m)), -A.T], [A, -np.eye(m), np.zeros((m,m))]])

    def find_nu(y, mu):
        #print "Y SHAPE: ", y.shape
        #print "MU SHAPE: ", mu.shape
        #print (y.dot(mu)) / float(m)
        return (y.dot(mu)) / float(m)

    def direction(A, b, c, x, mu, Q,y, sigma = .1):
        #create a variable for the negative F function
        negF = -F(A, b, c, x, mu,Q, y)
        #print negF.shape
        #print x

        #create the bottom half of our DF matrix and stack with the top
        bottom = np.bmat([[np.zeros((m,n)), np.diag((mu)), np.diag(y)]])
        DF = np.vstack((top_matrix, bottom))
        #print DF.shape 
        #print DF
        #print DF.shape

        #find nu to create our vector of 0, 0, and simga * nu * e
        nu = find_nu(y, mu)
        
        sig_nu_e = sigma * nu * np.ones(m)
        extra = np.vstack((np.hstack((np.zeros(m+n), sig_nu_e))))
        f = negF + extra
        #print f.shape
        

        factor = la.solve(DF,f)
        #print factor.shape
        #print factor

        #return 3 components for the delx, dellambda, delmu
        return factor[:n], factor[n:n+m], factor[n+m:]

    def step(mu, dmu, y, dy):
        #create a mask for mu
        mask = dmu < 0
        #print dmu
        try:
            mask = mask.flatten()
            dmu = dmu.flatten()
            beta = min(1, min(-mu[mask]/dmu[mask]))
            beta = min(1, .95 * beta)
        except:
            beta = .95

        masky = dy < 0
        try:
            deltamax = min(1, min(-y[masky]/dy[masky]))
            deltamax = min(1, .95 * deltamax)
        except:
            deltamax = .95
        alpha  = min(beta, deltamax)
        return alpha, beta, deltamax

    #choose starting point using auxiliary function
    x, y, mu = startingPoint(Q, c, A, b, guess)
    #print y
    #print x,y,mu

    t = 1
    while t < niter and find_nu(y, mu) > tol:
        #print find_nu(y,mu)
        dx, dy, dmu = direction(A, b, c, x, mu, Q, y)
        #print dx, dy, dmu
        #print dy.shape, dx.shape, dmu.shape
        #dy = dy.flatten()
        #dx.flatten()
        #print dx.shape
        alpha, beta, delta = step(mu, dmu, x, dx)
        #print alpha, beta, delta 
        #print beta 
        #print x
        #print dx 
        #calculate new x/lambda/mu values using step function
        #print x
        #print dx
        dx = dx.flatten()
        #print alpha*dx
        x = x + alpha * dx
        #print x
        dy = dy.flatten()
        y = y + alpha * dy
        dmu = dmu.flatten()
        mu = mu + alpha * dmu
        #print x.shape, y.shape, mu.shape
        #print x, y, mu
        t += 1
    return x, c.T.dot(x)

# Auxiliary function for problem 3
def laplacian(n):
    """Construct the discrete Dirichlet energy matrix H for an n x n grid."""
    data = -1*np.ones((5, n**2))
    data[2,:] = 4
    data[1, n-1::n] = 0
    data[3, ::n] = 0
    diags = np.array([-n, -1, 0, 1, n])
    return spdiags(data, diags, n**2, n**2).toarray()

# Problem 3
def circus(n=15):
    """Solve the circus tent problem for grid size length 'n'.
    Plot and show the solution.
    """
    H = laplacian(n)
    c = np.ones(n**2)*-1*(n-1)**(-2)
    A = np.eye(n**2)
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
    L = np.zeros((n,n))
    L[n//2-1:n//2+1,n//2-1:n//2+1] = .5
    m = [n//6-1, n//6, int(5*(n/6.))-1, int(5*(n/6.))]
    mask1, mask2 = np.meshgrid(m, m)
    L[mask1, mask2] = .3
    L = L.ravel()
    x = np.ones((n,n)).ravel()
    y = np.ones(n**2)
    mu = np.ones(n**2)
    # Calculate the solution.
    z = qInteriorPoint(H, c, A, L, (x,y,mu))[0].reshape((n,n))
    # Plot the solution.
    domain = np.arange(n)
    X, Y = np.meshgrid(domain, domain)
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_surface(X, Y, z, rstride=1, cstride=1, color='r')
    plt.show()


# Problem 4
def portfolio(filename="portfolio.txt"):
    """Use the data in the specified file to estimate a covariance matrix and
    expected rates of return. Find the optimal portfolio that guarantees an
    expected return of R = 1.13, with and then without short selling.

    Returns:
        An array of the percentages per asset, allowing short selling.
        An array of the percentages per asset without allowing short selling.
    """
    data = np.loadtxt(filename)
    data = data[:,1:]
    m, n = data.shape
    c = opt.matrix(np.zeros(n))
    mu = np.sum(data, axis =0)/m
    A = opt.matrix(np.vstack((np.ones(n),mu)))
    b = opt.matrix((np.hstack((1.,1.13))))
    h = opt.matrix((np.zeros(n)))
    Q = opt.matrix(np.cov(data.T))

    #allow short-selling
    G = opt.matrix(np.zeros((n,n)))
    soln = opt.solvers.qp(Q,c, G, h, A,b) 

    #dont allow short-selling
    G = opt.matrix(-np.eye(n))
    soln2 = opt.solvers.qp(Q,c,G,h,A,b)

    return np.array(soln['x']).T, np.array(soln2["x"]).T


def test():
    Q = np.array([[1, -1], [-1, 2]])
    c = np.array([-2, -6])
    A = np.array([[-1, -1], [1, -2], [-2, -1], [1, 0], [0, 1]])
    b = np.array([-2, -2, -3, 0, 0])
    x = np.array([.5, .5])
    y = np.array([1, 1, 1, 1, 1])
    mu = np.array([1, 1, 1, 1, 1])
    guess = (x, y, mu)
    #print qInteriorPoint(Q, c, A, b, guess)
    circus()
#test()

if __name__ == '__main__':
    #qInteriorPoint()
    #test()
    print portfolio()



