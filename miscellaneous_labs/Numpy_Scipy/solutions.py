"""Volume 1 Lab 2: NumPy and SciPy
Written Summer 2015 (Tanner Christensen)
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# student's file should be called solutions.py

# Problem 1: Perform matrix multiplication
"""Perform matrix-matrix multiplication on A and B.
Set the varibale 'product' to your answer.
"""
A = np.array([[2,4,0],[-3,1,-1],[0,3,2]])
B = np.array([[3,-1,2],[-2,-3,0],[1,0,-2]])
product = np.dot(A,B)  # set product equal to the result of AB.

# Problem 2: Return an array with all nonnegative numbers

def nonnegative(my_array):
    """Changes all negative entries in the inputed array to 0 then returns
    the new array.

    Example:
    >>> my_array = np.array([-3,-1,3])
    >>> nonnegative(my_array)
    array([0,0,3])
    """
    my_array = np.array(my_array)
    my_array[my_array<0] = 0
    return my_array



# Problem 3: nxn array of floats and operations on that array

def normal_var(n):
    """Creates nxn array with values from the normal distribution, computes 
    the mean of each row and computes the variance of these means. Return this
    final value.
    """
    A = np.random.randn(n,n)
    C = A.mean(axis=1)
    B = C.var()
    return B
#print normal_var(4)
   
# Problem 4: Solving Laplace's Equation using the Jacobi method and array slicing

def laplace(A, tolerance):
    """Solve Laplace's Equation using the Jacobi method and array slicing."""
    copy_A = np.copy(A) 
    max_diff = 100

    while max_diff >= tolerance:
        A[1:-1,1:-1] = (copy_A[:-2,1:-1] + copy_A[2:,1:-1] + copy_A[1:-1,2:] + copy_A[1:-1,:-2])/4.0
        diff = abs(A - copy_A)
        max_diff = np.max(diff)
        copy_A[1:-1,1:-1] = np.copy(A[1:-1,1:-1])
def laplace_plot():    
    """Visualize your solution to Laplace equation"""
    n = 100
    tol = .0001
    U = np.ones ((n, n))
    U[:,0] = 100 # sets north boundary condition
    U[:,-1] = 100 # sets south boundary condition
    U[0] = 0 # sets west boundary condition
    U[-1] = 0 # sets east boundary condition
    # U has been changed in place (note that laplace is the name of
    # the function in this case).
    laplace(U, tol)
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    ax.plot_surface (X, Y, U, rstride=5)
    plt.show()
laplace_plot()
# Problem 5: Blue shift an RGB image

def blue_shift():
    """Create a 100x100x3 array and perform a blue shift. Returns the original
    array and the blue-shifted array
    """
    A = np.random.random_integers(0,256, (100,100,3))
    B = np.array([.5,.5,1])

    return A, np.round(A*B)



def blue_shift_plot():
    """Visualize the original and the blue_shift image"""
    original, blue = blue_shift()
    original = 255 - original
    blue = 255 - blue
    plt.subplot(1,2,1)
    plt.imshow(original)
    plt.subplot(1,2,2)
    plt.imshow(blue)
    plt.show()

blue_shift_plot() 