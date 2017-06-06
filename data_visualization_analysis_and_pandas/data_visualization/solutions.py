"""Lab 11 data visualization
Drew Pearson
November 24th"""

from matplotlib import pyplot as plt
import numpy as np
from numpy.random import normal
from numpy.polynomial import Chebyshev as Cheb
from mpl_toolkits.mplot3d import axes3d, axes3d
from scipy.misc import comb

#Implement this function
def problemOne():
    '''
    Order and plot the data as a horizontal bar chart.
    '''
    male = np.array([179.2,160.0,177.8,178.9,178,176,172.5,165.6,170.8,183.2,182.4,164,163.6,175.4,174,176.1,165.7])
    female = np.array([167.6,142.2,164.5,165.3,165,164,158,154.9,157.4,168.4,168,151,151.4,164,158.9,162.1,155.2])
    country = 'Austria', 'Bolivia', 'England', 'Finland', 'Germany', 'Hungary', 'Japan','North Korea', 'South Korea', 'Montenegro', 'Norway', 'Peru','Sri Lanka', 'Switzerland', 'Turkey', 'U.S.', 'Vietnam'
    male2 = [179.2,160.0,177.8,178.9,178,176,172.5,165.6,170.8,183.2,182.4,164,163.6,175.4,174,176.1,165.7]
    female2 = [167.6,142.2,164.5,165.3,165,164,158,154.9,157.4,168.4,168,151,151.4,164,158.9,162.1,155.2] 
    
    width = 0.3
    pos = np.arange(17)+width
    #plt.subplot(1,2,1)
    #plt.title("Male")
   
    fig, ax = plt.subplots()
    ax.barh(pos, male, width, align = 'center', label = 'Male')
    ax.barh(pos-width, female, width, align = 'center',color = 'r', label = 'Female')
    ax.set(yticks = pos, yticklabels = country[::-1], ylim = [2*width - 1, len(country)])
    ax.legend(loc = 'center left', bbox_to_anchor = (.85, 1.05))
    #plt.yticks(pos, country[::-1])
    #plt.subplot(1,2,2)
    #plt.title("Female")
    #plt.barh(pos-.5, female, color = 'r', align = 'center')
    #plt.yticks(pos, country[::-1])

    plt.show()

# Implement this function
def problemTwo():
    '''
    Plot some histograms with white reference lines.  Do this for 20 bins,
    10 bins, 5 bins, and 3 bins
    '''
    """bins = [3,5,10,20]
    for i in bins:
        normal_numbers = normal(size=1000)
        plt.hist(normal_numbers, bins = i)
        plt.grid(True, color = 'w', linestyle = '-')
        plt.title("histogram for " +str(i) + " bins")
        plt.show()"""
    plt.subplot(2,2,1)
    normal_numbers = normal(size=1000)
    plt.hist(normal_numbers, bins = 3)
    plt.grid(True, color = 'w', linestyle = '-')
    plt.title("histogram for " +str(3) + " bins")

    plt.subplot(2,2,2)
    normal_numbers = normal(size=1000)
    plt.hist(normal_numbers, bins = 5)
    plt.grid(True, color = 'w', linestyle = '-')
    plt.title("histogram for " +str(5) + " bins")

    plt.subplot(2,2,3)
    normal_numbers = normal(size=1000)
    plt.hist(normal_numbers, bins = 10)
    plt.grid(True, color = 'w', linestyle = '-')
    plt.title("histogram for " +str(10) + " bins")

    plt.subplot(2,2,4)
    normal_numbers = normal(size=1000)
    plt.hist(normal_numbers, bins = 20)
    plt.grid(True, color = 'w', linestyle = '-')
    plt.title("histogram for " +str(20) + " bins")


    plt.show()
    

# Implement this function
def problemThree():
    '''
    Plot y = x^2 * sin(x) using 1000 data points and x in [0,100]
    '''
    x = np.linspace(0,100,1000)
    y = x**2*np.sin(x)
    plt.plot(x,y, linewidth = 2)
    plt.show()


# Implement this function
def problemFour():
    '''
    Plot a scatter plot of the average heights of men against women
    using the data from problem 1.
    '''
    male = np.array([179.2,160.0,177.8,178.9,178,176,172.5,165.6,170.8,183.2,182.4,164,163.6,175.4,174,176.1,165.7])
    female = np.array([167.6,142.2,164.5,165.3,165,164,158,154.9,157.4,168.4,168,151,151.4,164,158.9,162.1,155.2])
    plt.scatter(female, male)
    plt.show()
# Implement this function
def problemFive():
    '''
    Plot a contour map of z = sin(x) + sin(y) where x is in [0,12*pi] and
    y is in [0,12*pi]
    '''
    x = np.linspace(0, 12*np.pi, 400)
    y = np.linspace(0, 12*np.pi, 400)
    a, b = np.meshgrid(x,y)
    z = np.sin(a) + np.sin(b)
    plt.contourf(a, b, z, cmap = plt.get_cmap('afmhot'))
    plt.show()
    
# Implement this function
def problemSix():
    '''
    Plot each data set.
    '''
    dataI = np.array([[10,8.04],[8.,6.95],[13.,7.58],[9,8.81],[11.,8.33],[14.,9.96],[6.,7.24],[4.,4.26],[12.,10.84],[7.,4.82],[5.,5.68]])
    dataII = np.array([[10,9.14],[8.,8.14],[13.,8.74],[9,8.77],[11.,9.26],[14.,8.10],[6.,6.13],[4.,3.10],[12.,9.13],[7.,7.26],[5.,4.74]])
    dataIII = np.array([[10,7.46],[8.,6.77],[13.,12.74],[9,7.11],[11.,7.81],[14.,8.84],[6.,6.08],[4.,5.39],[12.,8.15],[7.,6.42],[5.,5.73]])
    dataIV = np.array([[8.,6.58],[8.,5.76],[8.,7.71],[8.,8.84],[8.,8.47],[8.,7.04],[8.,5.25],[19.,12.50],[8.,5.56],[8.,7.91],[8.,6.89]])
    plt.subplot(2,2,1)
    plt.scatter(dataI[:,0],dataI[:,1])
    plt.title('DataI')
    plt.subplot(2,2,2)
    plt.scatter(dataII[:,0],dataII[:,1])
    plt.title('DataII')
    plt.subplot(2,2,3)
    plt.title('DataIII')
    plt.scatter(dataIII[:,0],dataIII[:,1])
    plt.subplot(2,2,4)
    plt.title('DataIV')
    plt.scatter(dataIV[:,0],dataIV[:,1])
    plt.show()

# Implement this function
def problemSeven():
    '''
    Change the surface to a heatmap or a contour plot.  Return a string
    of the benefits of each type of visualization.
    '''
    x = np.linspace(-2*np.pi, 2*np.pi, num = 400)
    y = np.linspace(-2*np.pi, 2*np.pi, num = 400)
    X, Y = np.meshgrid(x,y)
    Z = np.exp(np.cos(np.sqrt(X**2 + Y**2)))
    #plt.subplot(1,2,1)
    plt.contourf(X,Y,Z, cmap = plt.get_cmap('afmhot'))
    #plt.subplot(1,2,2)
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #ax.plot_surface(X,Y,Z)
    plt.show()
    my_string = 'The 3d plot can show how much taller some points are in compared to other points, however, the backside is blocked So the contour is nice because it can show every single point. Plotting them together makes a good team. Because then you can see how tall the tallest points are from the 3d and then use the contour to find out the height and values of points on the backside of the 3d map. '
    return my_string

# Implement this function
def problemEight():
    '''
    Plot y = x^2 * sin(x) where x is in [0,100] and adjust to y limit to be
    [-10^k,10^k] for k = 0,1,2,3,4.
    '''
    x = np.linspace(0,100, 1000)
    y = (x**2) * (np.sin(x))
    k = [0,10,100,1000,10000]
    for i in k:
        plt.plot(x,y,lw = 2)
        plt.ylim(-i,i)
        plt.show() 

# Implement this function
def problemNine():
    '''
    Simplify one of your previous graphs.
    '''
    #Modify problem Twobins = [3,5,10,20]
    #for i in bins:
    plt.subplot(2,2,1)
    normal_numbers = normal(size=1000)
    plt.hist(normal_numbers, edgecolor = 'none', bins = 3)
    plt.grid(True, color = 'w', linestyle = '-')
    plt.title("histogram for " +str(3) + " bins")
    axis = plt.gca()
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.yaxis.set_ticks_position('left')
    axis.xaxis.set_ticks_position('bottom')

    plt.subplot(2,2,2)
    normal_numbers = normal(size=1000)
    plt.hist(normal_numbers, edgecolor = 'none', bins = 5)
    plt.grid(True, color = 'w', linestyle = '-')
    plt.title("histogram for " +str(5) + " bins")
    axis = plt.gca()
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.yaxis.set_ticks_position('left')
    axis.xaxis.set_ticks_position('bottom')
    plt.subplot(2,2,3)
    normal_numbers = normal(size=1000)
    plt.hist(normal_numbers, edgecolor = 'none', bins = 10)
    plt.grid(True, color = 'w', linestyle = '-')
    plt.title("histogram for " +str(10) + " bins")
    axis = plt.gca()
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.yaxis.set_ticks_position('left')
    axis.xaxis.set_ticks_position('bottom')
    plt.subplot(2,2,4)
    normal_numbers = normal(size=1000)
    plt.hist(normal_numbers, edgecolor = 'none', bins = 20)
    plt.grid(True, color = 'w', linestyle = '-')
    plt.title("histogram for " +str(20) + " bins")
    axis = plt.gca()
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.yaxis.set_ticks_position('left')
    axis.xaxis.set_ticks_position('bottom')
    plt.show()


# Implement this function
def problemTen():
    '''
    Plot the Bernstein polynomials for v,n in [0,3] as small multiples
    and as the cluttered version.
    '''
    def Bern_poly(v,n,x):
        return comb(n, v) *x**v*(1-x)**(n-v)
    n = 3
    x = np.linspace(0,1,1000)
    fig = plt.figure()
    fig.set_size_inches(5,5)
    fig.suptitle('Bernstein polynomials aka Berenstain Bears', fontsize = 15)
    for i in xrange(n+1):
        for j in xrange(i+1):
            k = 4*i + j + 1
            plt.subplot(4,4,k)
            plt.plot(x,Bern_poly(j, i, x))
    plt.show()






if __name__ == '__main__':
    #problemOne()
    problemTwo()
    #problemThree()
    #problemFour()
    #problemFive()
    #problemSix()
    #problemSeven()
    #problemEight()
    #problemTen()
    problemNine()


    """Problem 8, 9, 10"""


