# spec.py
"""Volume I Lab 5: Invertible Affine Transformations and Linear Systems.
Name: Drew Pearson
Date: Sep 29, 2015
"""

# include your import statements here.
import numpy as np
from matplotlib import pyplot as plt
import math
my_points = np.load('pi.npy')
import time 
from scipy import linalg as la

# Helper Functions
def plot_transform(original, new):
    """Display a plot of points before and after a transform.
    
    Inputs:
        original (array) - Array of size (2,n) containing points in R2 as columns.
        new (array) - Array of size (2,n) containing points in R2 as columns.
    """
    window = [-5,5,-5,5]
    plt.subplot(1, 2, 1)
    plt.title('Before')
    plt.gca().set_aspect('equal')
    plt.scatter(original[0], original[1])
    plt.axis(window)
    plt.subplot(1, 2, 2)
    plt.title('After')
    plt.gca().set_aspect('equal')
    plt.scatter(new[0], new[1])
    plt.axis(window)
    plt.show()

def type_I(A, i, j):  
    """Swap the i-th and j-th rows of A."""
    A[i], A[j] = np.copy(A[j]), np.copy(A[i])
    
def type_II(A, i, const):  
    """Multiply the i-th row of A by const."""
    A[i] *= const
    
def type_III(A, i, j, const):  
    """Add a constant of the j-th row of A to the i-th row."""
    A[i] += const*A[j]


# Problem 1
def dilation2D(A, x_factor, y_factor):
    """Scale the points in A by x_factor in the x direction and y_factor in
    the y direction. Returns the new array.
    
    Inputs:
        A (array) - Array of size (2,n) containing points in R2 stored as columns.
        x_factor (float) - scaling factor in the x direction.
        y_factor (float) - scaling factor in the y direction.
    """
    B = np.matrix([[x_factor, 0], [0, y_factor]])
    return np.dot(B, A)

#plot_transform(my_points, dilation2D(my_points, 1.5, 1))

# Problem 2
def rotate2D(A, theta):
    """Rotate the points in A about the origin by theta radians. Returns 
    the new array.
    
    Inputs:
        A (array) - Array of size (2,n) containing points in R2 stored as columns.
        theta (float) - number of radians to rotate points in A.
    """
    B = np.matrix([[math.cos(theta), -math.sin(theta)],[math.sin(theta), math.cos(theta)]])
    return np.dot(B, A)
#plot_transform(my_points, rotate2D(my_points, math.pi/3.0))
# Problem 3
def translate2D(A, b):
    """Translate the points in A by the vector b. Returns the new array.
    
    Inputs:
        A (array) - Array of size (2,n) containing points in R2 stored as columns.
        b (2-tuple (b1,b2)) - Translate points by b1 in the x direction and by b2 
            in the y direction.
    """
    b = np.vstack([b[0],b[1]]) 
    return A+b


#plot_transform(my_points, translate2D(my_points, [2, 0]))
# Problem 4
def rotatingParticle(time, omega, direction, speed):
    """Display a plot of the path of a particle P1 that is rotating 
    around another particle P2.
    
    Inputs:
     - time (2-tuple (a,b)): Time span from a to b seconds.
     - omega (float): Angular velocity of P1 rotating around P2.
     - direction (2-tuple (x,y)): Vector indicating direction.
     - speed (float): Distance per second.
    """

    t = np.linspace(time[0], time[1], 500)
    p1 = (1,0)
    xpos = np.empty(500)
    ypos = np.empty(500)
    counter  = 0
    for i in t:    
        p2=np.array((speed*i/np.linalg.norm(direction)) * np.vstack(direction))
        p1_rotate = rotate2D(np.vstack(p1),i*omega)
        p1_translate = translate2D(p1_rotate, p2)
        xpos[counter] = p1_translate[0]
        ypos[counter] = p1_translate[1]
        counter +=1
    plt.plot(xpos, ypos)
    plt.show()
#rotatingParticle((0,10), math.pi,(1,1),2.0)
    
# Problem 5
def REF(A):
    """Reduce a square matrix A to REF. During a row operation, do not
    modify any entries that you know will be zero before and after the
    operation. Returns the new array."""
    n, m = np.shape(A)
    U = A.copy().astype(float)
    for i in xrange(1, n):
        for j in xrange(0, i):
            U[i,j:] = U[i,j:] - (U[i,j]/U[j,j])*U[j,j:]
    return U

#print REF(np.array([[1,2,3],[4,5,6],[7,8,9]]))    
# Problem 6
def LU(A):
    """Returns the LU decomposition of a square matrix."""
    m, n = np.shape(A)
    U = A.copy().astype(float)
    L = np.identity(n)
    for i in xrange(1, n):
        for j in xrange(0,i):
            L[i,j] = U[i,j]/U[j,j]
            U[i,j:] = U[i,j:] - L[i,j]*U[j,j:]
    return L, U

#print LU(np.array([[1,2,3],[4,5,6],[7,8,9]])) 

# Problem 7
def time_LU():
    """Print the times it takes to solve a system of equations using
    LU decomposition and (A^-1)B where A is 1000x1000 and B is 1000x500."""
    
    A = np.random.rand(1000,1000)
    B = np.random.rand(1000, 500)
    start_time = time.time()
    C = la.lu_factor(A)
    end_time = time.time()

    start_time2 = time.time()
    D = la.inv(A)
    end_time2 = time.time()

    start_time3 = time.time()
    la.lu_solve(C , B)
    end_time3 = time.time()

    start_time4 = time.time()
    np.dot(D, B)
    end_time4 = time.time()

    time_lu_factor =  (end_time-start_time)  # set this to the time it takes to perform la.lu_factor(A)
    time_inv = (end_time2-start_time2) # set this to the time it takes to take the inverse of A
    time_lu_solve = (end_time3-start_time3)  # set this to the time it takes to perform la.lu_solve()
    time_inv_solve = (end_time4 - start_time4) # set this to the time it take to perform (A^-1)B


    print "LU solve: " + str(time_lu_factor + time_lu_solve)
    print "Inv solve: " + str(time_inv + time_inv_solve)
    
    # What can you conclude about the more efficient way to solve linear systems?
    print "LU solve was faster than Inv Solve so that is the more efficient way to solve linear systems"# print your answer here."""

#time_LU()

