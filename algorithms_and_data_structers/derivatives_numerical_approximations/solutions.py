#spec.py
"""Drew Pearson
    Math 345"""
import numpy as np 
from matplotlib import pyplot as plt
import numpy.linalg as la


# Problem 1: Implement this function.
def centered_difference_quotient(f,pts,h = 1e-5):
    '''
    Compute the centered difference quotient for function (f)
    given points (pts).
    Inputs:
        f (function): the function for which the derivative will be approximated
        pts (array): array of values to calculate the derivative
    Returns:
        centered difference quotient (array): array of the centered difference
            quotient
    '''
    Cent_diff = lambda pts: .5*(f(pts+h)-f(pts-h))/h
    return Cent_diff(pts)


# Problem 2: Implement this function.
def jacobian(f,n,m,pt,h = 1e-5):
    '''
    Compute the approximate Jacobian matrix of f at pt using the centered
    difference quotient.
    Inputs:
        f (function): the multidimensional function for which the derivative
            will be approximated
        n (int): dimension of the domain of f
        m (int): dimension of the range of f
        pt (array): an n-dimensional array representing a point in R^n
        h (float): a float to use in the centered difference approximation
    Returns:
        Jacobian matrix of f at pt using the centered difference quotient.
    '''
    my_list = []
    for j in range(n):
        h_e = np.zeros(n)
        h_e[j] = h
        my_list.append(.5*(f(pt+h_e)-f(pt-h_e))/h)
    return np.vstack(my_list).T
    #return np.concatenate(my_list, axis = 1)    


# Problem 3: Implement this function.
def findError():
    '''
    Compute the maximum error of your jacobian function for the function
    f(x,y)=[(e^x)*sin(y)+y^3,3y-cos(x)] on the square [-1,1]x[-1,1].
    Returns:
        Maximum error of your jacobian function.
    '''
    f = lambda x:np.array([[np.exp(x[0])*np.sin(x[1])+x[1]**3], [3*x[1]-np.cos(x[0])]])
    real_f_prime = lambda x:np.array([[np.exp(x[0])*np.sin(x[1]),np.exp(x[0])*np.cos(x[1])+3*x[0]**2], [np.sin(x[0]),3]])
    x_space = np.linspace(-1,1,100)
    numerical = []
    actual = [] 
    for x in x_space:
        for y in x_space:
            numerical.append(jacobian(f,2,2,np.array([x,y]), h=1e-10).reshape(2,2))
            actual.append(real_f_prime([x,y]))
    numerical = np.array(numerical)
    #print numerical.shape
    #print numerical[0]
    actual = np.array(actual)
    #print actual[0]
    #print actual.shape
    diff = numerical - actual
    max_diff = 0
    for i in diff:
        j = la.norm(i)
        if j>max_diff:
            max_diff = j
    return max_diff

        
# Problem 4: Implement this function.
def Filter(image,F):
    '''
    Applies the filter to the image.
    Inputs:
        image (array): an array of the image
        F (array): an nxn filter to be applied (a numpy array).
    Returns:
        The filtered image.
    '''
    m, n = image.shape
    print image.shape
    h, k = F.shape

    image_pad = np.zeros((m+(h-1), n+(k-1)))
    print image_pad[((h-1)/2):((h-1)/2)+m, ((k-1)/2):((k-1)/2)+n].shape
    image_pad[((h-1)/2):((h-1)/2)+m, ((k-1)/2):((k-1)/2)+n] = image
    # Make the interior of image_pad equal to image
    C = np.zeros(image.shape)
    for i in range(m):
        for j in range(n):
            C[i,j] = sum([F[x][y]*image_pad[i+x][j+y] for x in range((-h+1)/2, (h-1)/2) for y in range((-k+1)/2, (k-1)/2)])
    return C


# Problem 5: Implement this function.
def sobelFilter(image):
    '''
    Applies the Sobel filter to the image
    Inputs:
        image(array): an array of the image in grayscale
    Returns:
        The image with the Sobel filter applied.
    '''
    raise NotImplementedError("Problem 5 Incomplete")


def test_one():
    f = lambda x: x**2
    pts = np.array([1,2,3,4])
    print centered_difference_quotient(f,pts)
def test_two():
    f = lambda (x,y): np.array([[(x**2)*y,5*x+np.sin(y)]])
    pt = np.array([1,0])
    print jacobian(f,2,2,pt)
def test_filter():
    G = 1/159.*np.array([[2,4,5,4,2],[4,9,12,9,4],[5,12,15,12,5],[4,9,12,9,4],[2,4,5,4,2]])
    image = plt.imread('cameraman.png')
    plt.subplot(1,2,1)
    plt.imshow(image, cmap = 'gray')
    plt.subplot(1,2,2)
    plt.imshow(Filter(image,G), cmap = 'gray')
    plt.show()



if __name__ == '__main__':
    #test_one()
    #test_two()
    print findError()
    #test_filter()


