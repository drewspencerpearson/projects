# name this file solutions.py
"""Volume II Lab 6: Nearest Neighbor Search
Drew Pearson
Math 320
10/15/15
"""

import numpy as np
from Trees import BST
from Trees import BSTNode
from scipy.spatial import distance
import math
from sklearn import neighbors  

# Problem 1: Implement this function.
def euclidean_metric(x, y):
    """Return the euclidean distance between the vectors 'x' and 'y'.

    Raises:
        ValueError: if the two vectors 'x' and 'y' are of different lengths.
    
    Example:
        >>> print(euclidean_metric([1,2],[2,2]))
        1.0
        >>> print(euclidean_metric([1,2,1],[2,2]))
        ValueError: Incompatible dimensions.
    """

    if len(x) != len(y):
        raise ValueError("the vectors do not have the same dimensions")
    else:
            return math.sqrt(sum((x-y)**2))



# Problem 2: Implement this function.
def exhaustive_search(data_set, target):
    """Solve the nearest neighbor search problem exhaustively.
    Check the distances between 'target' and each point in 'data_set'.
    Use the Euclidean metric to calculate distances.
    
    Inputs:
        data_set (mxk ndarray): An array of m k-dimensional points.
        target (1xk ndarray): A k-dimensional point to compare to 'dataset'.
        
    Returns:
        the member of 'data_set' that is nearest to 'target' (1xk ndarray).
        The distance from the nearest neighbor to 'target' (float).
    """
    i = 0
    while i < len(data_set):
        x = data_set[i]
        if i ==0:
            dst = euclidean_metric(x, target)
            answer = data_set[i]
            i +=1
        elif euclidean_metric(x,target)  < dst:
            dst = euclidean_metric(x,target)
            answer = data_set[i]
            i+=1
        else:
            i+=1
    return answer, dst



# Problem 3: Finish implementing this class by modifying __init__()
#   and adding the __sub__, __eq__, __lt__, and __gt__ magic methods.
class KDTNode(BSTNode):
    """Node class for K-D Trees. Inherits from BSTNode.

    Attributes:
        left (KDTNode): a reference to this node's left child.
        right (KDTNode): a reference to this node's right child.
        parent (KDTNode): a reference to this node's parent node.
        data (ndarray): a coordinate in k-dimensional space.
        axis (int): the 'dimension' of the node to make comparisons on.
    """
    

    def __init__(self, data):
        """Construct a K-D Tree node containing 'data'. The left, right,
        and prev attributes are set in the constructor of BSTNode.

        Raises:
            TypeError: if 'data' is not a a numpy array (of type np.ndarray).
        """
        if type(data) != np.ndarray:
            raise TypeError("data must be an np.ndarray")
        BSTNode.__init__(self, data)
        self.axis  = 0
    def __sub__(self, other):
        return euclidean_metric(self.data, other.data)
    def __eq__(self, other):
        return np.allclose(self.data, other.data)
    def __lt__(self, other): 
        return self.data[other.axis] < other.data[other.axis]
    def __gt__(self, other):
        return self.data[other.axis] > other.data[other.axis]

# Problem 4: Finish implementing this class by overriding
#   the insert() and remove() methods.
class KDT(BST):
    """A k-dimensional binary search tree object.
    Used to solve the nearest neighbor problem efficiently.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other
            nodes in the tree, the root houses data as a numpy array.
        k (int): the dimension of the tree (the 'k' of the k-d tree).
    """
    
    def find(self, data):
        """Return the node containing 'data'.

        Raises:
            ValueError: if there is node containing 'data' in the tree,
                or the tree is empty.
        """

        # First check that the tree is not empty.
        if self.root is None:
            raise ValueError(str(data) + " is not in the tree.")
        
        # Define a recursive function to traverse the tree.
        def _step(current, target):
            """Recursively approach the target node."""
            
            if current is None:             # Base case: target not found.
                return current
            if current == target:            # Base case: target found!
                return current
            if target < current:            # Recursively search to the left.
                return _step(current.left, target)
            else:                           # Recursively search to the right.
                return _step(current.right, target)
            
        # Create a new node to use the KDTNode comparison operators.
        n = KDTNode(data)

        # Call the recursive function, starting at the root.
        found = _step(self.root, n)
        if found is None:                  # Report the data was not found.
            raise ValueError(str(data) + " is not in the tree.")
        return found                       # Otherwise, return the target node.

    def insert(self, data):
        new_node = KDTNode(data) 
        current = self.root

        def _find_parent(current, other):
            if current is None:
                self.root = other
                self.root.axis = 0
                other.axis = 0
            elif other < current:
                if current.left is None:
                    current.left = other
                    other.prev = current
                    other.axis = (current.axis + 1)%len(data) 
                else:
                    _find_parent(current.left, other)
            else:
                if current.right is None:
                    current.right = other
                    other.prev = current
                    other.axis = (current.axis+1)%len(data)
                else:
                    _find_parent(current.right, other)

        _find_parent(current, new_node)
    def remove(self, *args):
        raise NotImplementedError("remove() has been disabled")


# Problem 5: Implement this function.
def nearest_neighbor(data_set, target):
    """Use the KDT class to solve the nearest neighbor problem.

    Inputs:
        data_set (mxk ndarray): An array of m k-dimensional points.
        target (1xk ndarray): A k-dimensional point to compare to 'dataset'.

    Returns:
        The point in the tree that is nearest to 'target' (1xk ndarray).
        The distance from the nearest neighbor to 'target' (float).
    """
    m, n = np.shape(data_set)

    tree = KDT()
    for i in range(m):
        tree.insert(data_set[i])

    current = tree.root
    target = KDTNode(target)
    neighbor = tree.root 
    distance = euclidean_metric(tree.root.data, target.data)
    def KDTsearch(current, target, neighbor, distance):
        if current is None:
            return neighbor, distance 
        index = current.axis 
        if euclidean_metric(current.data, target.data) < distance:
            neighbor = current
            distance = euclidean_metric(current.data, target.data)
        if target.data[index] < current.data[index]:
            neighbor, distance = KDTsearch(current.left, target, neighbor, distance)
            if target.data[index]+ distance >= current.data[index]:
                neighbor, distance = KDTsearch(current.right, target, neighbor, distance)
        else:
            neighbor, distance = KDTsearch(current.right, target, neighbor, distance)
            if target.data[index] - distance <= current.data[index]:
                neighbor, distance = KDTsearch(current.left, target, neighbor, distance)

        return neighbor, distance

    neighbor, distance = KDTsearch(current, target, neighbor, distance)
    return neighbor.data, distance 



# Problem 6: Implement this function.
def postal_problem():
    """Use the neighbors module in sklearn to classify the Postal data set
    provided in 'PostalData.npz'. Classify the testpoints with 'n_neighbors'
    as 1, 4, or 10, and with 'weights' as 'uniform' or 'distance'. For each
    trial print a report indicating how the classifier performs in terms of
    percentage of misclassifications.

    Your function should print a report similar to the following:
    n_neighbors = 1, weights = 'distance':  0.903
    n_neighbors = 1, weights =  'uniform':  0.903       (...and so on.)
    """
    labels, points, testlabels, testpoints = np.load('PostalData.npz').items()

    y = [1,4,10]
    x = ['uniform','distance']
    for i in y:
        for j in x:  
            nbrs = neighbors.KNeighborsClassifier(n_neighbors = i, weights = j)
            nbrs.fit(points[1], labels[1])
            prediction = nbrs.predict(testpoints[1])
            percent = np.average(prediction/testlabels[1])
            print "n_neighbors = " + str(i) + ", weights = " + str(j) + ": " + str(percent) + "\n"

#postal_problem()
    

    

def test_euclidean():
        x = np.array([1,2,3])
        y = np.array([3,2,1])
        print euclidean_metric(x,y)
def test_exhaust():
    x = np.array([[1,2,3],[2,2,2],[2,3,4],[1,1,1],[3,1,1],[3,2,1]])
    y = np.array([3,2,1])
    print exhaustive_search(x,y)

def test_3():
    x = KDTNode(np.array([3,2,1]))
    y = KDTNode(np.array([3,2,1]))
    print x == y
    print x < y
    print x > y 

def test_insert():
    b = KDT()
    b.insert('super')
    b.insert(np.array([8,4,3]))
    b.insert(np.array([3,2,1]))
    b.insert(np.array([7,7,6]))
    b.insert(np.array([2,6,11]))
    b.insert(np.array([9,2,14]))
    b.insert(np.array([7,7,8]))
    b.insert(np.array([8,8,5]))
    #b.remove()
    print b
def test_kdtsearch():
    a = np.array([[1,1],[10,10],[11,11]])
    b = np.array([-2,-2])
    print nearest_neighbor(a,b)

if __name__ == '__main__':
    
    #test_exhaust()
    #test_euclidean()
    #test_3()
    postal_problem()
    #test_insert()
    #test_kdtsearch()
# ============================== END OF FILE =============================== #