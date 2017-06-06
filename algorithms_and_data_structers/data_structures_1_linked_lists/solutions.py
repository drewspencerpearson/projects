# name this file 'solutions.py'
"""Volume II Lab 4: Data Structures 1 (Linked Lists)
Drew Pearson
Math 321
10/1/15
"""

# Do not modify the first two import statements (for the test driver)
from LinkedLists import LinkedListNode, LinkedList
from LinkedLists import DoublyLinkedList, SortedLinkedList
from WordList import create_word_list


# Problem 1: in LinkedLists.py, add magic methods to the Node class.

# Problems 2, 3, 4: Complete the implementation of the LinkedList class.

# Problem 5: Implement the DoublyLinkedList class.

# Problem 6: Implement the SortedLinkedList class in LinkedLists.py and the
# sort_words() function in this file.

# Conclude problem 6 by implementing this function.
def sort_words(filename = "English.txt"):
    """Use the 'create_word_list' method from the 'WordList' module to generate
    a scrambled list of words from the specified file. Use an instance of
    the SortedLinkedList class to sort the list. Then return the list.
    
    Inputs:
        filename (str, opt): the file to be parsed and sorted.
            Defaults to 'English.txt'.
    
    Returns:
        A SortedLinkedList object containing the sorted list of words.
    """
    my_list = create_word_list(filename)
    my_list_sorted = SortedLinkedList()

    for word in my_list:
        my_list_sorted.add(word)

    return my_list_sorted

def test_sort():
    print sort_words('test.txt')

if __name__ == '__main__':
    
    test_sort()