# Name this file 'solutions.py'.
"""Volume II Lab 22: Dynamic Optimization (Value Function Iteration).
Drew Pearson
Math 323
April 7, 2016
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg as la 

def eatCake(beta, N, W_max=1, T=None, finite=True, plot=False):
    """Solve the finite- or infinite-horizon cake-eating problem using
    Value Function iteration.
    
    Inputs:
        beta (float): Discount factor.
        N (int): The number of discrete cake values.
        W_max (int): The original size of the cake.
        T (int): The final time period. Defaults to None.
        finite (bool): If True, solve the finite-horizon problem. If False,
            solve the infinite-horizon problem.
        plot (bool): If True, plot the value function surface and policy
            function.

    Returns:
        values ((N, T+2) ndarray if finite=True, (N,) ndarray if finite=False):
            The value function at each time period for each state (this is also
            called V in the lab).
        psi ((N, T+1) ndarray if finite=True, (N,) ndarray if finite=False):
            The policy at each time period for each state.
    """
    if finite:
        w = np.linspace(0,W_max,N)
        utility_values = np.empty((N,N))
        for i in xrange(N):
            for j in xrange(N):
                if i >= j:
                    utility_values[i][j] = np.sqrt(w[i]-w[j])
                else:
                    utility_values[i][j] = -10**10

        values = np.zeros((N,T+2))
        policy = np.zeros((N,T+1))
        for x in xrange(T,-1,-1):
            choices = utility_values+beta*values[:,x+1]
            best_collumn = np.max(choices, axis=1)
            values[:,x] = best_collumn
            policy[:,x] = w[np.argmax(choices,axis = 1)]

        if plot:
            W = np.linspace(0, W_max, N)
            x = np.arange(0, N)
            y = np.arange(0, T+2)
            X, Y = np.meshgrid(x, y)
            fig1 = plt.figure()
            ax1 = Axes3D(fig1)
            ax1.plot_surface(W[X], Y, np.transpose(values), cmap=cm.coolwarm)
            plt.show()
            fig2 = plt.figure()
            ax2 = Axes3D(fig2)
            y = np.arange(0,T+1)
            X, Y = np.meshgrid(x, y)
            ax2.plot_surface(W[X], Y, np.transpose(policy), cmap=cm.coolwarm)
            plt.show()
        return values, policy
    else:
        w = np.linspace(0,W_max,N)
        utility_values = np.empty((N,N))
        for i in xrange(N):
            for j in xrange(N):
                if i >= j:
                    utility_values[i][j] = np.sqrt(w[i]-w[j])
                else:
                    utility_values[i][j] = -10**10
        V0 = np.zeros((N,1))
        V1 = np.max((beta*V0+utility_values),axis = 1)
        best = np.argmax(beta*V0+utility_values,axis = 1)
        delta = np.inf
        while delta > 10**-9:
            V0 = np.copy(V1)
            V1 = np.max(beta*V0+utility_values,axis = 1)
            best = np.argmax(beta*V0+utility_values,axis = 1)
            delta = la.norm(V1-V0)
        policy = w[best]
        if plot:
            plt.plot(w,policy)
            plt.show()
        return V1, policy  




def prob2():
    """Call eatCake() with the parameters specified in the lab."""
    eatCake(.9,100,T=1000,plot = True)


def prob3():
    """Modify eatCake() to deal with the infinite case.
    Call eatCake() with the parameters specified in part 6 of the problem.
    """
    return eatCake(.9,100,finite=False, plot=True)



if __name__ == '__main__':
    #print eatCake(.9,100,T=10,plot=True)
    #prob2()
    print prob3()

