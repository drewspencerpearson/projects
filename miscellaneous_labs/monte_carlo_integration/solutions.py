# spec.py
"""Volume I: Monte Carlo Integration
Drew Pearson
Math 347
Jan 19, 2015
"""
import numpy as np
from numpy import linalg 
# Problem 1
def prob1(numPoints = 10**5):
    """Return an estimate of the volume of the unit sphere using Monte
    Carlo Integration.
    
    Inputs:
        numPoints (int, optional) - Number of points to sample. Defaults
            to 10^5.
    Returns:
        volume (int) - Approximate value of the area of the unit sphere.
    """
    points = np.random.rand(3,numPoints)
    points = points*2-1
    sphere_mask = (points[0,:])**2 + (points[1,:])**2 + (points[2,:])**2 <=1
    numincircle = np.count_nonzero(sphere_mask)
    return 8.0*numincircle/numPoints

# Problem 2
def prob2(numPoints = 10**5):
    """Return an estimate of the area under the curve,
    f(x) = |sin(10x)cos(10x) + sqrt(x)*sin(3x)| from 1 to 5.
    
    Inputs:
        numPoints (int, optional) - Number of points to sample. Defautls
            to 10^5.
    Returns:
        area (int) - Apprimate value of the area under the 
            specified curve.
    """
    points = np.random.uniform(1,5,numPoints)
    f = lambda x: np.abs(np.sin(10*x)*np.cos(10*x) +np.sqrt(x)*np.sin(3*x))

    area = (4./numPoints)*sum(f(points))
    return area

# Problem 3
def mc_int(f, mins, maxs, numPoints=500, numIters=100):
    """Use Monte-Carlo integration to approximate the integral of f
    on the box defined by mins and maxs.
    
    Inputs:
        f (function) - The function to integrate. This function should 
            accept a 1-D NumPy array as input.
        mins (1-D np.ndarray) - Minimum bounds on integration.
        maxs (1-D np.ndarray) - Maximum bounds on integration.
        numPoints (int, optional) - The number of points to sample in 
            the Monte-Carlo method. Defaults to 500.
        numIters (int, optional) - An integer specifying the number of 
            times to run the Monte Carlo algorithm. Defaults to 100.
        
    Returns:
        estimate (int) - The average of 'numIters' runs of the 
            Monte-Carlo algorithm.
                
    Example:
        >>> f = lambda x: np.hypot(x[0], x[1]) <= 1
        >>> # Integral over the square [-1,1] x [-1,1]. Should be pi.
        >>> mc_int(f, np.array([-1,-1]), np.array([1,1]))
        3.1290400000000007
    """
    my_list = []
    def monte_carlo():
        points = np.random.rand(numPoints, len(mins))
        points = points*(maxs-mins)+mins
        f_vals = np.apply_along_axis(f,1,points)
        return np.prod(maxs-mins) * (1./numPoints)*sum(f_vals)
    for i in xrange(numIters):
        my_list.append(monte_carlo())
    return sum(my_list)/numIters
        
       
# Problem 4
def prob4(numPoints=[500]):
    """Calculates an estimate of the integral of 
    f(x,y,z,w) = sin(x)y^5 - y^5 + zw + yz^3
    
    Inputs:
        numPoints (list, optional) - a list of the number of points to 
            use in the approximation. Defaults to [500].
    Returns:
        errors (list) - a list of the errors when calculating the 
            approximation using 'numPoints' points.
    Example:
    >>> prob4([100,200,300])
    [-0.061492011289160729, 0.016174426377108819, -0.0014292910207835802]
    """
    f = lambda x: np.sin(x[0])*x[1]**5 -x[1] **3 + x[2]*x[3]+x[1]*x[2]**3
    mins = np.array([-1,-1,-1,-1])
    maxs = np.array([1,1,1,1])
    points = [100,1000,10000]
    for i in numPoints:
        print "Error for " + str(i) + "sample point: " +str(abs(mc_int(f, mins, maxs, i, 100)))

def test3():
    f = lambda x: np.hypot(x[0], x[1]) <= 1
    # Integral over the square [-1,1] x [-1,1]. Should be pi.
    print mc_int(f, np.array([-1,-1]), np.array([1,1]))

def test():
    A = np.array([[4,2,0],[2,2,2],[0,2,13]])
    print np.linalg.det(A)


if __name__ == '__main__':
    #print prob1()
    test()
    #print prob2()
    #test3()
