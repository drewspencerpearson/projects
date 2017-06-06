'''
Lab 14 - Newton's Method.
'''

import numpy as np
from matplotlib import pyplot as plt

def Newtons_method(f, x0, Df=None, iters=15, tol=.002):
    '''Use Newton's method to approximate a zero of a function.
    
    INPUTS:
    f     - A function handle. Should represent a function from 
            R to R.
    x0    - Initial guess. Should be a float.
    Df    - A function handle. Should represent the derivative 
            of `f`.
    iters - Maximum number of iterations before the function 
            returns. Defaults to 15.
    tol   - The function returns when the difference between 
            successive approximations is less than `tol`.
    
    RETURN:
    A tuple (x, converged, numiters) with
    x           - the approximation to a zero of `f`
    converged   - a Boolean telling whether Newton's method 
                converged
    numiters    - the number of iterations the method computed
    '''
    if Df is None:
        return Newtons_2(f,x0,iters,tol)
    i = 1
    old = x0
    while i <= iters:
        new = old - f(old)/float(Df(old))
        if abs(new-old) <=tol:
            return (new, True, i)
        old = new
        i +=1
    return (new, False, i)

    
def prob2():
    '''
    Print the answers to the questions in problem 2.
    '''
    f1 = lambda x: np.cos(x)
    df = lambda x: -np.sin(x)
    #print Newtons_method(f,1,df,15,10**(-5))
    x = np.linspace(-4,4,100)
    f2 = lambda x: (np.sin(x)/x)-x
    plt.plot(f2(x),x)
    plt.show()
    df2 = lambda x: -(x**2 + np.sin(x)-x*np.cos(x))/(x**2)

    f3 = lambda x: x**9
    df3 = lambda x:9*(x**8)
    #print Newtons_method(f3,1.0,df3,1000,10**(-7))[0]
    #print Newtons_method(f2,1.0,df2,15,10**(-7))[0]
    f4 = lambda x: np.sign(x)*np.power(np.abs(x), 1./3)
    df4 = lambda x: (1/3.0)*np.sign(x)*np.power(np.abs(x), -2./3)
    print Newtons_method(f4,.01,df4,1000,10**(-5))
    print '1.' +"4 iterations for x0 = 1. " + "3 iterations for x0 = 2"  
    print '2.' + str(Newtons_method(f2,1.0,df2,15,10**(-7))[0])
    print '3.' + str(Newtons_method(f3,1.0,df3,1000,10**(-5))[2]) + "This is slower because x^9 is a lot flatter near the zeros so it has to go through more iterations."
    print '4.' + "the Newtons Method diverges to -infinity because the first derivative is not continuous."
    
def Newtons_2(f, x0, iters=15, tol=.002):
    '''
    Optional problem.
    Re-implement Newtons method, but without a derivative.
    Instead, use the centered difference method to estimate the derivative.
    '''
    h = 1e-5
    Cent_diff = lambda pts: .5*(f(pts+h)-f(pts-h))/h
    i = 1
    old = x0
    while i <= iters:
        new = old - f(old)/float(Cent_diff(old))
        if abs(new-old) <=tol:
            return (new, True, i)
        old = new
        i +=1
    return (new, False, i)

def plot_basins(f, Df, roots, xmin, xmax, ymin, ymax, numpoints=100, iters=15, colormap='brg'):
    '''Plot the basins of attraction of f.
    
    INPUTS:
    f       - A function handle. Should represent a function 
            from C to C.
    Df      - A function handle. Should be the derivative of f.
    roots   - An array of the zeros of f.
    xmin, xmax, ymin, ymax - Scalars that define the domain 
            for the plot.
    numpoints - A scalar that determines the resolution of 
            the plot. Defaults to 100.
    iters   - Number of times to iterate Newton's method. 
            Defaults to 15.
    colormap - A colormap to use in the plot. Defaults to 'brg'. 
    
    RETURN:
    Returns nothing, but should display a plot of the basins of attraction.
    '''
    xreal = np.linspace(xmin, xmax, numpoints)
    ximag = np.linspace(ymin, ymax, numpoints)
    Xreal, Ximag = np.meshgrid(xreal, ximag)
    Xold = Xreal+1j*Ximag
    
  
    Xnew = Xold
    for i in xrange(iters):
        Xold = Xnew
        Xnew = Xold - f(Xold)/Df(Xold)
    def match(x):
        return np.argmin(abs(roots-[x]*len(roots)))
    Xnew = np.vectorize(match)(Xnew)
    plt.pcolormesh(Xreal, Ximag, Xnew)
    plt.show()


def prob5():
    '''
    Using the function you wrote in the previous problem, plot the basins of
    attraction of the function x^3 - 1 on the interval [-1.5,1.5]X[-1.5,1.5]
    (in the complex plane).
    '''
    roots = np.array([0,-1j**(1./3),1j**(2./3)])
    f = lambda x: (x**3)-1
    Df = lambda x: 3*x**2
    plot_basins(f,Df,roots,-1.5,1.5,-1.5,1.5,1000,100)


def test_prob1():
    f = lambda x: x**2 - 1
    Df= lambda x: 2*x
    print Newtons_method(f,1.5, Df = None)
    print Newtons_method(f,1.5,Df)
def test_ec():
    f = lambda x: x**2 - 1
    Df= lambda x: 2*x
    print Newtons_2(f,1.5)
def test4():
    roots = np.array([0,1,-1])
    f = lambda x : x**3-x
    Df = lambda x : 3*x**2 - 1
    plot_basins(f,Df,roots,-1.5,1.5,-1.5,1.5,1000,100)

if __name__ == '__main__':
    #test_prob1()
    #prob2()
    #Newtons_2()
    #test_ec()
    test4()
    #prob5()
    #plot_basins()

