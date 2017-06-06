# name this file solutions.py
"""Volume 2 Lab 14: Optimization Packages II (CVXOPT)
Drew Pearson
Math 323
January 14, 2015
"""
from cvxopt import matrix 
from cvxopt import solvers
import numpy as np

def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x + 2y          >= 3
                    2x + y + 3z     >= 10
                    x               >= 0
                    y               >= 0
                    z               >= 0

    Returns (in order):
        The optimizer (sol['x'])
        The optimal value (sol['primal objective'])
    """
    c = matrix([2.,1.,3.])
    G = matrix([[-1.,-2.,-1.,0.,0.],[-2.,-1.,0.,-1.,0.],[0.,-3.,0.,0.,-1.]])
    h = matrix([-3.0,-10.0,0.,0.,0.])
    sol = solvers.lp(c,G,h)
    return sol['x'], sol['primal objective']


def prob2():
    """Solve the transportation problem by converting all equality constraints
    into inequality constraints.

    Returns (in order):
        The optimizer (sol['x'])
        The optimal value (sol['primal objective'])
    """
    c = matrix([4.,7.,6.,8.,8.,9.])
    G = matrix([[1.,-1.,0.,0.,0.,0.,1.,-1.,0.,0.,-1.,0.,0.,0.,0.,0.],[1.,-1.,0.,0.,0.,0.,0.,0.,1.,-1.,0.,-1.,0.,0.,0.,0.],[0.,0.,1.,-1.,0.,0.,1.,-1.,0.,0.,0.,0.,-1.,0.,0.,0.],[0.,0.,1.,-1.,0.,0.,0.,0.,1.,-1.,0.,0.,0.,-1.,0.,0.],[0.,0.,0.,0.,1.,-1.,1.,-1.,0.,0.,0.,0.,0.,0.,-1.,0.],[0.,0.,0.,0.,1.,-1.,0.,0.,1.,-1.,0.,0.,0.,0.,0.,-1.]])
    h = matrix([7.,-7.,2.,-2.,4.,-4.,5.,-5.,8.,-8.,0.,0.,0.,0.,0.,0.])
    sol = solvers.lp(c,G,h)
    return sol['x'], sol['primal objective']

def prob3():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer (sol['x'])
        The optimal value (sol['primal objective'])
    """
    Q = matrix([[3.,2.,1.],[2.,4.,2.],[1.,2.,3.]])
    p = matrix([3.,0.,1.])
    sol = solvers.qp(Q,p)
    return np.array(sol['x']), sol['primal objective']


def prob4():
    """Solve the allocation model problem in 'ForestData.npy'.
    Note that the first three rows of the data correspond to the first
    analysis area, the second group of three rows correspond to the second
    analysis area, and so on.

    Returns (in order):
        The optimizer (sol['x'])
        The optimal value (sol['primal objective']*-1000)
    """
    data = np.load("ForestData.npy")
    c = -data[:,3]
    t = -data[:,4]
    g = -data[:,5]
    w = -data[:,6]
    acres = data[::3,1]

    K = np.array([[1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                [0.,0.,0.,0.,0.,0.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,1.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,1.,0.,0.,0.],
                [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,1.],[-1.,-1.,-1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                [0.,0.,0.,-1.,-1.,-1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,-1.,-1.,-1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                [0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,-1.,-1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,-1.,-1.,0.,0.,0.,0.,0.,0.],
                [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,-1.,-1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,-1.,-1.]])
    a = -np.eye(21)
    G = np.vstack((K,t,g,w,a))
    #G = np.vstack((K,t))
    #G = np.vstack((G,g))
    #G = np.vstack((G,w))
    #G = np.vstack((G,a)) 
    h = np.array([acres[0],acres[1],acres[2],acres[3],acres[4],acres[5],acres[6],-acres[0],-acres[1],-acres[2],-acres[3],-acres[4],-acres[5],-acres[6],-40000.,-5.,(-70.0*788)])
    h = np.hstack((h,np.zeros(21)))

    G = matrix(G)
    h = matrix(h)

    sol = solvers.lp(matrix(c),G,h)
    return sol['x'], -1000*sol['primal objective']



if __name__ == '__main__':
    #print prob1()
    #print prob2()
    #print prob3()
    print prob4()

