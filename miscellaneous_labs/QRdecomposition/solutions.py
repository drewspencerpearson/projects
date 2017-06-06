# spec.py
"""Volume I Lab 6: QR Decomposition.
Name: Drew Pearson  
Date: 10/15/15
"""
import numpy as np
from scipy import linalg as la
import math
import scipy
from scipy import stats

def QR(A):
    '''
    Compute the QR decomposition of a matrix.
    Accept an m by n matrix A of rank n. 
    Return Q, R
    '''
    Q, R = la.qr(A)
    return Q, R 
    
def prob2(A):
    '''
    Use your QR decomposition from the previous problem to compute 
    the determinant of A.
    Accept a square matrix A of full rank.
    Return |det(A)|.
    '''
    Q, R = QR(A)
    return abs(np.linalg.det(R))

def householder(A):
    '''
    Use the Householder algorithm to compute the QR decomposition
    of a matrix.
    Accept an m by n matrix A of rank n. 
    Return Q, R
    '''
    A = A.astype(float)
    m, n = np.shape(A)
    R = np.copy(A)
    Q = np.eye(m)
    for k in range (0,n):
        u = np.copy(R[k:,k])
        u = np.reshape(u,(m-k,1))
        u[0] = u[0]+(u[0]/(abs(u[0])))*np.linalg.norm(u)
        u = u/(np.linalg.norm(u))
        R[k:,k:] = R[k:,k:] - 2*np.dot(u,np.dot(np.transpose(u),R[k:,k:]))
        Q[k:] = Q[k:] - 2*np.dot(u,np.dot(np.transpose(u),Q[k:]))
    return np.transpose(Q), R 


def hessenberg(A):
    '''
    Compute the Hessenberg form of a matrix. Find orthogonal Q and upper
    Hessenberg H such that A = QtHQ.
    Accept a non-singular square matrix A.
    Return Q, H
    '''
    A = A.astype(float)
    m, n = np.shape(A)
    H = np.copy(A)
    Q = np.eye(m)
    for k in range (0,(n-2)):
        u = np.copy(H[k+1:,k])
        u = np.reshape(u, (m-k-1,1))
        u[0]= u[0] + (u[0]/(abs(u[0])))*np.linalg.norm(u)
        u = u/(np.linalg.norm(u))
        H[k+1:,k:] = H[k+1:,k:] -   2*np.dot(u,np.dot(np.transpose(u),H[k+1:,k:]))
        H[:,k+1:] = H[:,k+1:] - 2*(np.dot(np.dot(H[:,k+1:],u),np.transpose(u)))
        Q[k+1:] = Q[k+1:] - 2*np.dot(u,np.dot(np.transpose(u),Q[k+1:]))
    return Q, H
def givens(A):
    '''
    EXTRA 20% CREDIT
    Compute the Givens triangularization of matrix A.
    Assume that at the ijth stage of the algorithm, a_ij will be nonzero.
    Accept A
    Return Q, R
    '''
    pass

def prob6(H):
    '''
    EXTRA 20% CREDIT
    Compute the Givens triangularization of an upper Hessenberg matrix.
    Accept upper Hessenberg H.
    
    '''
    pass

def test_qr():
    A = np.array([[2,4,5],[3,2,1],[4,5,7],[5,6,9]])
    #print QR(A)
    Q,R = QR(A)
    Z = np.dot(Q,R)
    return Z
    #print np.allclose(Q.dot(R),A)

def prob_2():
    A = np.array([[3,2,1],[1,2,3],[2,1,1]])
    print A
    print prob2(A)

def prob3():
    A = np.array([[1,-2,3.5],[1,3,-0.5],[1,3,2.5],[1,-2,0.5]])
    #print A
    Q,R = householder(A)
    Z2 = np.dot(Q,R)
    print np.allclose(np.dot(np.transpose(Q),Q),np.eye(4))
    print Q
    print R
    print np.allclose(Z2, A)
def test_matrices():
    A = np.array([[1,2]])
    B = np.array([[1,2]])
    C = np.array([[1,2]])
    print np.dot(C,np.dot(np.transpose(B),A))
def prob4():
    A = np.array([[-1,-4,3], [4,5,6], [10,11,12]])
    Q,H= hessenberg(A)
    print Q 
    print H
    print np.allclose(A,np.dot(np.dot(np.transpose(Q),H),Q))


def test():
    import math 
    x = 2000-10*176
    y = 50/(math.sqrt((900*5)/36.0))
    print y 
    
    print scipy.stats.norm.cdf(y) - scipy.stats.norm.cdf(0)
    #x/()

def testifft():
    A = np.array([0,1,0,0])
    A = A.T
    print np.fft.ifft(A)



if __name__ == '__main__':
    
    testifft()
    #test()
    #Z = test_qr()
    #prob_2()
    #prob3()
    #test_matrices()
    #prob4()
#print np.allclose(Z, Z2)
#print np.allclose(R,R1)



