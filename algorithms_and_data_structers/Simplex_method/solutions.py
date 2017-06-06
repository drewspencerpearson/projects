# name this file 'solutions.py'.
"""Volume II Lab 16: Simplex
Drew Pearson
Math 323
Jan 28, 2015

Problems 1-6 give instructions on how to build the SimplexSolver class.
The grader will test your class by solving various linear optimization
problems and will only call the constructor and the solve() methods directly.
Write good docstrings for each of your class methods and comment your code.

prob7() will also be tested directly.
"""

import numpy as np
# Problems 1-6
class SimplexSolver(object):
    """Class for solving the standard linear optimization problem

                        maximize        c^Tx
                        subject to      Ax <= b
                                         x >= 0
    via the Simplex algorithm.
    """
    
    def __init__(self, c, A, b):
        """

        Parameters:
            c (1xn ndarray): The coefficients of the linear objective function.
            A (mxn ndarray): The constraint coefficients matrix.
            b (1xm ndarray): The constraint vector.

        Raises:
            ValueError: if the given system is infeasible at the origin.
        """
        #x = np.zeros(len(b))
        for i in b:
            if i < 0:
                raise ValueError('origin is not a feasible starting point.')

        self.c = c
        self.A = A
        self.b = b
        self.L = range(A.shape[1], A.shape[1]+A.shape[0])+range(0,A.shape[1])


        index_list = []
        for i in xrange(len(self.b)):
            index_list.append(i+len(self.c))
        for i in xrange(len(self.c)):
            index_list.append(i)
        m = np.shape(self.A)[0]
        col1 = np.hstack((0,self.b))
        middle_top = np.hstack((-self.c.T,np.zeros(m)))
        middle_bottom = np.hstack((self.A,np.eye(m)))
        middle = np.vstack([middle_top,middle_bottom])
        front = np.column_stack([col1.T,middle])
        end = np.hstack((1,np.zeros(m)))
        self.Tableau = np.column_stack([front,end.T])

    def chose_pivot(self):
        col = np.argmax(self.Tableau[0,:] <0)
        if np.all(self.Tableau[:,col] <0 ):
            raise ValueError("The problem is unbounded")
        else:
            divisor = np.copy(self.Tableau[1:,col])
            divisor[divisor<0] = 0
            ratios = self.Tableau[1:,0] /divisor
            row = np.argmin(abs(ratios)) + 1
        return row, col
    def pivot(self):
        row, col = self.chose_pivot()
        basic, nonbasic = self.L.index(col-1), self.L.index(row+1)
        self.L[basic], self.L[nonbasic] = self.L[nonbasic], self.L[basic]
        self.Tableau[row, :] = self.Tableau[row,:]/ self.Tableau[row][col]
        for i in xrange(row):
            self.Tableau[i,:] = -self.Tableau[i,col]*self.Tableau[row, :] +self.Tableau[i, :]
        for i in xrange(row+1, self.Tableau.shape[0]):
            self.Tableau[i,:] = -self.Tableau[i,col]*self.Tableau[row, :] +self.Tableau[i, :]

    def solve(self):
        """Solve the linear optimization problem.

        Returns:
            (float) The maximum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """
        while np.any(self.Tableau[0,:]<0):
            self.pivot()

        maximum = self.Tableau[0][0]
        basic = {}
        nonbasic = {}
        m, n = np.shape(self.Tableau)
        for i in xrange(1,n-1):
            if self.Tableau[0,i] != 0:
                nonbasic[i-1] = 0
            else:
                for j in xrange(m):
                    if self.Tableau[j,i] ==1:
                        basic[i-1] = self.Tableau[j,0]

        """basic_optimizers = self.Tableau[1:,0]
        basic_index = self.L[:len(self.b)]
        basic_dictionary = dict(zip(basic_index,basic_optimizers))
        nonbasic_index = np.zeros(len(self.c))
        nonbasic_dictionary = dict(zip(self.L[len(self.b):], nonbasic_index))
        maximum = self.Tableau[0][0]"""

        #print self.Tableau
        return maximum, basic, nonbasic


# Problem 7
def prob7(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.

    Parameters:
        filename (str): the path to the data file.

    Returns:
        The minimizer of the problem (as an array).
    """
    product = np.load('productMix.npz')
    A = product['A']
    P = product['p']
    M = product['m']
    D = product['d']
    A = np.vstack((A, np.eye(A.shape[1])))
    b = np.hstack((M,D))
    simplex = SimplexSolver(P,A,b)
    maximum, basic, nonbasic = simplex.solve()

    n = len(P)
    final = np.zeros(n)
    for i in xrange(n):
        if basic.has_key(i):
            final[i] = basic[i]
        else:
            final[i] = nonbasic[i]

    return final




def test():
    b = np.array([1,2,-4,4])
    for i in b:
        if i <0:
            raise ValueError("it works")

def test2():
    A = np.array([[1,-1],[3,1],[4,3]])
    b = np.array([2,5,7])
    b = b.T
    c = np.array([3,2])
    c = c.T
    #print A
    #print b
    #print c
    m = np.shape(A)[0]
    col1 = np.hstack((0,b))
    middle_top = np.hstack((-c.T,np.zeros(m)))
    middle_bottom = np.hstack((A,np.eye(m)))
    middle = np.vstack([middle_top,middle_bottom])
    print col1
    print middle_bottom
    print middle
    front = np.column_stack([col1.T,middle])
    end = np.hstack((1,np.zeros(m)))
    final = np.column_stack([front,end.T])
    print final


def test7():
    A = np.array([[1,3,7,-5,-6]])
    print np.argmax(A[0,:] <0)

def test():
    c = np.array([3, 2])
    A = np.array([[1, -1], [3, 1], [4, 3]])
    b = np.array([2, 5, 7])
    simplex = SimplexSolver(c, A, b)
    #print simplex.L
    #print simplex.pivot_choose()
    print simplex.solve()

def test2():
    c = np.array([3, 1])
    A = np.array([[1, 3], [2, 3], [1, -1]])
    b = np.array([15, 8, 4])
    simplex = SimplexSolver(c, A, b)
    print simplex.solve()

def test3():
    c = np.array([3., 1.])
    b = np.array([15., 18., 4.])
    A = np.array([[1., 3.], [2., 3.], [1., -1]])
    solver = SimplexSolver(c, A, b)
    sol = solver.solve()
    print(sol)

def test4():
    c = np.array([4., 6.])
    b = np.array([11., 27., 90.])
    A = np.array([[-1., 1.], [1., 1.], [2., 5.]])
    solver = SimplexSolver(c, A, b)
    #print solver.Tableau
    sol = solver.solve()
    print(sol)

def test9():
    print prob7()
    

if __name__ == '__main__':
    #test()
    #test2()
    #test3()
    #test4()
    test9()

