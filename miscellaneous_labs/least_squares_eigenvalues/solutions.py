"""Volume 1 Lab 7
Drew Pearson
Math 345"""

import numpy as np
from numpy import linalg as lq 
from scipy import linalg as la
from numpy.lib import scimath
from matplotlib import pyplot as plt


# Problem 1
def least_squares(A,b):
    """Return the least squares solutions to Ax = b using QR decomposition."""
    Q, R = lq.qr(A)
    #print R
    C = np.dot(lq.inv(R), np.dot(Q.T, b)) 
    return C 

# Problem 2
def line_fit():
    """Plot linepts and its best-fit line on the same plot."""
    linepts = np.load('data.npz')['linepts']
    n = np.ones((len(linepts)))
    x_linefit = linepts[:,0]
    y_linefit = linepts[:,1]
    A = np.vstack((x_linefit,n)).T
    b = np.vstack(y_linefit)
    xhat = la.lstsq(A,y_linefit)[0]
    #xhat = la.lstsq(A, y_linefit)
    #print xhat
    x0 = np.linspace(0,5000,100)
    y0 = xhat[0]*x0
    #plt.plot(A,y_linefit,'*', x0,y0)
    plt.plot(x_linefit,y_linefit, '*', x0,y0)
    plt.show()


# Problem 3
def ellipse_fit():
    """Plot ellipsepts and its best-fit line on the same plot."""
    ellipsepts = np.load('data.npz')['ellipsepts']
    x_ellipse = ellipsepts[:,0]
    y_ellipse = ellipsepts[:,1]
    A = np.vstack((x_ellipse**2,x_ellipse,x_ellipse*y_ellipse, y_ellipse, y_ellipse**2)).T
    b = np.ones((len(x_ellipse),1))
    least = least_squares(A,b)

    plot_ellipse(x_ellipse,y_ellipse,least[0], least[1], least[2], least[3],least[4])

def plot_ellipse(X, Y, a, b, c, d, e):
    """Plots an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1.

    Input:
      X (array) - x-coordinates of all the data points.
      Y (array) - y-coordinates of all the data points.
      a,b,c,d,e (float) - the coefficients from the equation of an 
                    ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1.
    """
    def get_r(a, b, c, d, e):
        theta = np.linspace(0,2*np.pi,200)
        A = a*(np.cos(theta)**2) + c*np.cos(theta)*np.sin(theta) + e*(np.sin(theta)**2)
        B = b*np.cos(theta) + d*np.sin(theta)
        r = (-B + np.sqrt(B**2 + 4*A))/(2*A)
        return r, theta
        
    r,theta = get_r(a,b,c,d,e)
    plt.plot(r*np.cos(theta), r*np.sin(theta), color = "r")
    plt.plot(X,Y,".", color = "b")
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()

# Problem 4
def power_method(A,tol):
    """Return the dominant eigenvalue of A and its corresponding eigenvector."""
    n, m = np.shape(A)
    diff = tol
    b = np.random.rand(n)
    while diff >= tol:
        c = np.copy(b)
        norm = la.norm(np.dot(A,c))
        b = np.dot(A, b)/norm
        diff = la.norm(c-b)
    D = np.dot(A, b)
    return b, np.inner(D,b)/(la.norm(b)**2)


    
# Problem 5
def QR_algorithm(A,niter,tol):
    """Return the eigenvalues of A using the QR algorithm."""
    A = la.hessenberg(A).astype(float)
    eigenvalues = []
    n = 0
    for i in xrange(1, niter):
        Q, R = la.qr(A)
        A = np.dot(np.dot(la.inv(Q),A),Q)
    while n < np.shape(A)[1]:   
        if n == np.shape(A)[1]-1:
            eigenvalues.append(A[n][n])
        elif abs(A[n+1][n]) < tol:
            eigenvalues.append(A[n][n])
        else:
            two_two = A[n:n+2, n:n+2]
            a = 1
            b = -1*(two_two[0][0] + two_two[1][1])
            c = la.det(two_two)
            x = (-b+(scimath.sqrt(b**2-4*a*c)))/2
            x2 = (-b-(scimath.sqrt(b**2-4*a*c)))/2
            eigenvalues.append(x)
            eigenvalues.append(x2)
            n+=1
        n+=1
    return eigenvalues



def Test_least():
    A = np.array([[1,3,4],[3,4,7],[9,10,11]])
    b = np.array([1,4,5])
    print least_squares(A, b)
def test_2():
    A = np.array([[1,2,3,4],[5,6,7,8],[9,10,15,12],[13,14,11,19]])
    #A = np.array([[4,5],[6,5]])
    tol = .001
    print power_method(A,tol)
def test_3():
    A = np.array([[4,12,17,-2],[-5.5,-30.5,-45.5,9.5],[3,20,30,-6],[1.5,1.5,1.5,1.5]])
    print QR_algorithm(A,100,.01)

#ellipse_fit()
#test_3()
#line_fit()
#test_2()