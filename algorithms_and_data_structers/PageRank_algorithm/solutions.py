"""Drew Pearson
    Math 347"""

import numpy as np
import scipy.sparse as spar
import scipy.linalg as la
from scipy.sparse import linalg as sla

def to_matrix(filename,n):
    '''
    Return the nxn adjacency matrix described by datafile.
    INPUTS:
    datafile (.txt file): Name of a .txt file describing a directed graph. 
        Lines describing edges should have the form '<from node>\t<to node>\n'.
        The file may also include comments.
    n (int): The number of nodes in the graph described by datafile
    RETURN:
        Return a SciPy sparse `dok_matrix'.
    '''
    A = np.zeros((n,n))
    with open(filename, 'r') as myfile:
        for line in myfile:
            try:
                p = line.strip().split()
                A[p[0],p[1]] += 1
            except:
                pass
    return spar.dok_matrix(A)


def calculateK(A,N):
    '''
    Compute the matrix K as described in the lab.
    Input:
        A (array): adjacency matrix of an array
        N (int): the datasize of the array
    Return:
        K (array)
    '''
    d = A.sum(axis=1)
    for i in xrange(N):
        if d[i] ==0:
            A[i,:] = 1
    d = A.sum(axis=1)
    K = (A/d).T
    return K

def iter_solve(adj, N=None, d=.85, tol=1E-5):
    '''
    Return the page ranks of the network described by `adj`.
    Iterate through the PageRank algorithm until the error is less than `tol'.
    Inputs:
    adj - A NumPy array representing the adjacency matrix of a directed graph
    N (int) - Restrict the computation to the first `N` nodes of the graph.
            Defaults to N=None; in this case, the entire matrix is used.
    d     - The damping factor, a float between 0 and 1.
            Defaults to .85.
    tol  - Stop iterating when the change in approximations to the solution is
        less than `tol'. Defaults to 1E-5.
    Returns:
    The approximation to the steady state.
    '''
    if N== None:
        N = adj.shape[0]
    else:
        adj = adj[:N,:N]
    k = calculateK(adj,N)
    p_t = 1./N*np.ones((N,1))
    p_t_1 = d*np.dot(k,p_t)+((1.-d)/N)*np.ones((N,1))

    while la.norm(p_t_1-p_t) > tol:
        p_t = p_t_1
        p_t_1 = d*np.dot(k,p_t)+((1.-d)/N)*np.ones((N,1))

    return p_t_1.T



def eig_solve( adj, N=None, d=.85):
    '''
    Return the page ranks of the network described by `adj`. Use the
    eigenvalue solver in scipy.linalg to calculate the steady state
    of the PageRank algorithm
    Inputs:
    adj - A NumPy array representing the adjacency matrix of a directed graph
    N - Restrict the computation to the first `N` nodes of the graph.
            Defaults to N=None; in this case, the entire matrix is used.
    d     - The damping factor, a float between 0 and 1.
            Defaults to .85.
    Returns:
    The approximation to the steady state.
    '''
    if N== None:
        N = adj.shape[0]
    else:
        adj = adj[:N,:N]
    B = d*calculateK(adj,N)+((1-d)/N)*np.ones((N,N))
    e_val, e_vec = la.eig(B)
    rank = e_vec[:,np.argmax(e_val)]
    rank = rank/np.sum(rank)
    return rank


    
def problem5(filename='ncaa2013.csv'):
    '''
    Create an adjacency matrix from the input file.
    Using iter_solve with d = 0.7, run the PageRank algorithm on the adjacency 
    matrix to estimate the rankings of the teams.
    Inputs:
    filename - Name of a .txt file containing data for basketball games. 
        Should contain a header row: 'Winning team,Losing team",
        after which each row contains the names of two teams,
        winning and losing, separated by a comma
    Returns:
    sorted_ranks - The array of ranks output by iter_solve, sorted from highest
        to lowest.
    sorted_teams - List of team names, sorted from highest rank to lowest rank.   
    '''
    game_list = []
    teams = set()
    i = 0
    with open(filename, "r") as myfile:
        for line in myfile:
            if i != 0:
                game = line.strip().split(',')
                game_list.append(game)
                teams.add(game[0])
                teams.add(game[1])
            i+=1 
    game_list = game_list[1:]
    teams = sorted(list(teams))

    adj = np.zeros((len(teams),len(teams)))
    for game in game_list:
        adj[teams.index(game[1]),teams.index(game[0])]=1
    ranks = iter_solve(adj, d=.7).flatten()
    sorted_ranks = np.argsort(ranks)
    winners_list = [teams[j] for j in sorted_ranks]
    winners_list.reverse()
    print "Wichita St pulled the biggest upset"
    return ranks[sorted_ranks], winners_list
    
def problem6():
    '''
    Optional problem: Load in and explore any one of the SNAP datasets.
    Run the PageRank algorithm on the adjacency matrix.
    If you can, create sparse versions of your algorithms from the previous
    problems so that they can handle more nodes.
    '''
    pass

def test():
    A = to_matrix('datafile.txt',8)
    print calculateK(A,8)
def test2():
    A = np.array([[2,4,6,8],[0,2,4,6],[0,0,2,4],[0,0,0,4]])
    print la.eigvals(A)

def test3():
    A = to_matrix("datafile.txt", 8)
    print iter_solve(A)
if __name__ == '__main__':
    #print to_matrix("datafile", 8)
    #test()
    #test2()  
    #test3()
    print problem5()

