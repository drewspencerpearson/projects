# name this file 'solutions.py'
"""Volume I Lab 3: Plotting with matplotlib
Drew Pearson
10/15/15
"""

# Add your import statements here.
from matplotlib import pyplot as plt 
import numpy as np
import math
from mayavi import mlab 
from scipy import linalg as la
# Problem 1
def curve():
    """Plot the curve 1/(x-1) on [-2,6]. Plot the two sides of the curve separately
    (still with a single call to plt.plot()) so that the graph looks discontinuous 
    at x = 1.
    """

    x1 = np.linspace(-2, .99, 100)
    x2 = np.linspace(1.01,6, 100)
    y1 = 1.0/(x1-1)
    y2 = 1.0/(x2-1)

    plt.plot(x1,y1, "m--", x2,y2, "m--", linewidth = 5)
    plt.ylim([-6,6])
    #plt.plot(x2,y2)
    plt.show()
#curve()
# Problem 2
def colormesh():
    """Plot the function f(x,y) = sin(x)sin(y)/(xy) on [-2*pi, 2*pi]x[-2*pi, 2*pi].
    Include the scale bar in your plot.
    """
    p = math.pi
    x = np.linspace(-2*p, 2*p, 300)
    y = np.linspace(-2*p, 2*p, 300)
    X,Y = np.meshgrid(x,y)
    f = (np.sin(X) * np.sin(Y))/(X*Y)
    plt.pcolormesh(X,Y,f, cmap = 'seismic')
    plt.ylim([-2*p, 2*p])
    plt.xlim([-2*p, 2*p])
    plt.colorbar()
    plt.gca().set_aspect('equal')
    plt.show()
#colormesh()

# Problem 3
def histogram():
    """Plot a histogram and a scatter plot of 50 random numbers chosen in the
    interval [0,1)
    """
    x = np.random.rand(50)
    plt.subplot(1,2,1)
    plt.hist(x, bins =5, range = [0,1])

    plt.subplot(1,2,2)
    t = np.linspace(1,50)
    plt.scatter(t,x,s=100)
    plt.plot(t,np.ones(50)*x.mean(), 'r')

    plt.show()
    

#histogram()

# Problem 4 
def ripple():
    """Plot z = sin(10(x^2 + y^2))/10 on [-1,1]x[-1,1] using Mayavi."""
    x,y = np.mgrid[-1:1:0.025, -1:1:0.025]

    z = np.sin(10*(x**2 + y**2))/10
    mlab.surf(x,y,z)
    mlab.show()

def test():
    A = np.array([[-2,1,0,1],[1,-2,1,0],[0,1,-2,1],[1,0,1,-2]])
    print la.eig(A)
test()
#ripple()
