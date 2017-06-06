import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg
from scipy import sparse
from scipy.sparse import linalg as sl
from time import time 

'''Functions for use in problem 1.'''
# Run through a single for loop.
def func1(n):
    start_time = time()
    n = 500*n
    sum(xrange(n))
    end_time = time()
    return (end_time - start_time)

# Run through a double for loop.
def func2(n):
    start_time = time()
    n = 3*n
    t = 0
    for i in xrange(n):
        for j in xrange(i):
            t += j
    end_time = time()
    return (end_time - start_time)
# Square a matrix.
def func3(n):
    start_time = time()
    n = int(1.2*n)
    A = np.random.rand(n, n)
    np.power(A, 2)
    end_time = time()
    return (end_time - start_time)
# Invert a matrix.
from scipy import linalg as la
def func4(n):
    start_time = time()
    A = np.random.rand(n, n)
    la.inv(A)
    end_time = time()
    return (end_time - start_time)
# Find the determinant of a matrix.
from scipy import linalg as la
def func5(n):
    start_time = time()
    n = int(1.25*n)
    A = np.random.rand(n, n)
    la.det(A)
    end_time = time()
    return (end_time - start_time)

def Problem1():
    """Create a plot comparing the times of func1, func2, func3, func4, 
    and func5. Time each function 4 times and take the average of each.
    """
    x = [100,200,400,800]
    i = 0
    y = []
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    while i < 4: 
        y.append(func1(x[i]))
        y1.append(func2(x[i]))
        y2.append(func3(x[i]))
        y3.append(func4(x[i]))
        y4.append(func5(x[i]))    
        i +=1
    plt.plot(x,y, label = 'Function 1')
    plt.plot(x,y1, label = 'Function 2')
    plt.plot(x,y2, label = 'Function 3')
    plt.plot(x,y3, label = 'Function 4')
    plt.plot(x,y4, label = 'Function 5')
    plt.legend(loc = 'upper left')
    plt.show()
#Problem1()
    

def Problem2(n):
    """takes an integer argument n and returns a sparse n n 
    tri-diagonal array with along the diagonal and along
    the two sub-diagonals above and below the diagonal.
    """
    x = np.ones(n) * 2
    y = np.ones(n) * (-1)
    diag_entries = np.vstack((y,x,y))
    
    A = sparse.spdiags(diag_entries, [-1,0,1], n, n, format = 'csr')
    
    return A
#print Problem2(20).todense()

def Problem3(n):
    """Generate an nx1 random array b and solve the linear system Ax=b
    where A is the tri-diagonal array in Problem 2 of size nxn
    """
    A = Problem2(n)
    b = np.random.rand(n, 1)
    return sl.spsolve(A, b)
    
#print Problem3(9)
def Problem4(n, sparse=False):
    """Write a function that accepts an integer argument n and returns
    (lamba)*n^2 where (lamba) is the smallest eigenvalue of the sparse 
    tri-diagonal array you built in Problem 2.
    """
    A = Problem2(n)
    E, W  = sl.eigs(A.asfptype(), which = 'SM')
    e = min(E)
    return e*(n**2)
    
