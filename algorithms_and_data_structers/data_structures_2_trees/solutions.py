# name this file 'solutions.py'
"""Volume II Lab 5: Data Structures II (Trees)
Drew Pearson
Math 321
10/1/15
"""

from Trees import BST
from Trees import AVL
from LinkedLists import LinkedList
from WordList import create_word_list
import numpy as np
import matplotlib.pyplot as plt
import time

def iterative_search(linkedlist, data):
    """Find the node containing 'data' using an iterative approach.
    If there is no such node in the list, or if the list is empty,
    raise a ValueError with error message "<data> is not in the list."
    
    Inputs:
        linkedlist (LinkedList): a linked list object
        data: the data to find in the list.
    
    Returns:
        The node in 'linkedlist' containing 'data'.
    """
    # Start the search at the head.
    current = linkedlist.head
    
    # Iterate through the list, checking the data of each node.
    while current is not None:
        if current.data == data:
            return current
        current = current.next
    
    # If 'current' no longer points to a Node, raise a value error.
    raise ValueError(str(data) + " is not in the list.")


# Problem 1: rewrite iterative_search() using recursion.
def recursive_search(linkedlist, data):
    """Find the node containing 'data' using a recursive approach.
    If there is no such node in the list, raise a ValueError with error
    message "<data> is not in the list."
    
    Inputs:
        linkedlist (LinkedList): a linked list object
        data: the data to find in the list.
    
    Returns:
        The node in 'linkedlist' containing 'data'.
    """
    def recurse(current, data):
        """current is a node"""
        if current is None:
            raise ValueError(str(data) + " is not in the list")

        if current.data == data:
            return current 

        return recurse(current.next, data)
    return recurse(linkedlist.head, data)

def test():
    l = LinkedList()
    l.add(1)
    l.add(2)
    l.add(3)
    print iterative_search(l, 3)
    print recursive_search(l, 3)
    try:
        recursive_search(l, "vector space")
        print "Failure!"
    except ValueError as e:
        print "Success!" + str(e)
#test()


# Problem 2: Implement BST.insert() in Trees.py.


# Problem 3: Implement BST.remove() in Trees.py


# Problem 4: Test build and search speeds for LinkedList, BST, and AVL objects.
def plot_times(filename="English.txt", start=500, stop=5500, step=500):
    """Vary n from 'start' to 'stop', incrementing by 'step'. At each
    iteration, use the create_word_list() from the 'WordList' module to
    generate a list of n randomized words from the specified file.
    
    Time (separately) how long it takes to load a LinkedList, a BST, and
    an AVL with the data set.
    
    Choose 5 random words from the data set. Time how long it takes to
    find each word in each object. Calculate the average search time for
    each object.
    
    Create one plot with two subplots. In the first subplot, plot the
    number of words in each dataset against the build time for each object.
    In the second subplot, plot the number of words against the search time
    for each object.
    
    Inputs:
        filename (str): the file to use in creating the data sets.
        start (int): the lower bound on the sample interval.
        stop (int): the upper bound on the sample interval.
        step (int): the space between points in the sample interval.
    
    Returns:
        Show the plot, but do not return any values.
    """

    def get_average_time_linked_list(to_search, linked_list, times_left, current_time = 0):
        while times_left > 0:
            start = time.time()
            iterative_search(linked_list, to_search[times_left-1])
            end =time.time()
            current_time +=(end-start)
            times_left -=1
        return current_time/len(to_search)

    def get_average_time_BST(to_search, BST_list, times_left, current_time =0):
        while times_left >0:
            start = time.time()
            BST_list.find(to_search[times_left-1])
            end = time.time()
            current_time +=(end-start)
            times_left -= 1 
        return current_time/len(to_search)
    def get_average_time_AVL(to_search, AVL_list, times_left, current_time = 0):
        while times_left > 0:
            start = time.time()
            AVL_list.find(to_search[times_left-1])
            end = time.time()
            current_time +=(end-start)
            times_left -= 1
        return current_time/len(to_search)


    word_list = create_word_list(filename)
    if (stop-start)%step!=0:
        raise ValueError("Your steps won't get you from start to stop")
    current = start
    time_linked_list = []
    time_BST_list = []
    time_AVL_list = []

    time_linked_list_search = []
    time_BST_list_search = []
    time_AVL_list_search = []

    set_size = []

    while current < stop:
        current_linked_list = LinkedList()
        current_BST = BST()
        current_AVL = AVL()
        current_list = word_list[:current]
        to_search = np.random.permutation(current_list)
        start_linked_time = time.time()

        for x in current_list:
            current_linked_list.add(x)
        end_linked_time = time.time()

        start_BST_time = time.time()
        for y in current_list:
            current_BST.insert(y)
        end_BST_time = time.time()

        start_AVL_time = time.time()
        for z in current_list:
            current_AVL.insert(z)
        end_AVL_time = time.time()

        time_linked_list.append(end_linked_time - start_linked_time)
        time_BST_list.append(end_BST_time - start_BST_time)
        time_AVL_list.append(end_AVL_time- start_AVL_time)

        time_linked_list_search.append(get_average_time_linked_list(to_search,current_linked_list, len(to_search)))
        time_BST_list_search.append(get_average_time_BST(to_search,current_BST, len(to_search)))
        time_AVL_list_search.append(get_average_time_AVL(to_search,current_AVL, len(to_search)))

        set_size.append(current)

        current+=step
    plt.subplot(2,1,1)
    plt.title('Building Data Structures')
    plt.plot(set_size,time_linked_list, label = 'Linked List', linewidth = 3)
    plt.plot(set_size, time_BST_list, label = "BST", linewidth = 3)
    plt.plot(set_size, time_AVL_list, label = "AVL", linewidth = 3)
    plt.legend(loc = 2)

    plt.subplot(2,1,2)
    plt.title("Searching Data Structures")
    plt.plot(set_size, time_linked_list_search, label = 'Linked list', linewidth = 3)
    plt.plot(set_size, time_BST_list_search, label = 'BST', linewidth = 3)
    plt.plot(set_size, time_AVL_list_search, label = 'AVL', linewidth = 3)
    plt.legend(loc = 2)
    plt.show()

if __name__ == '__main__':
    main()
#plot_times()




    

# =============================== END OF FILE =============================== #