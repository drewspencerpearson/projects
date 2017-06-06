# LinkedLists.py
"""Volume II Lab 4: Data Structures 1 (Linked Lists)
Auxiliary file. Modify this file for problems 1-5.
Drew Pearson
Math 320
9/24/15
"""


# Problem 1: Add the magic methods __str__, __lt__, __eq__, and __gt__.
class Node(object):
    """A Node class for storing data."""
    def __init__(self, data):
        """Construct a new node that stores some data."""
        self.data = data
    def __str__(self):
        return str(self.data) 

    def __lt__(self, other):
        return self.data < other.data
    def __eq__(self, other):
        return self.data == other.data
    def __gt__(self, other):
        return self.data > other.data


class LinkedListNode(Node):
    """A Node class for linked lists. Inherits from the 'Node' class.
    Contains a reference to the next node in the list.
    """
    def __init__(self, data):
        """Construct a Node and initialize an attribute for
        the next node in the list.
        """
        Node.__init__(self, data)
        self.next = None

# Problems 2-4: Finish implementing this class.
class LinkedList(object):
    """Singly-linked list data structure class.
    The first node in the list is referenced to by 'head'.
    """
    def __init__(self):
        """Create a new empty linked list. Create the head
        attribute and set it to None since the list is empty.
        """
        self.head = None

    def add(self, data):
        """Create a new Node containing 'data' and add it to
        the end of the list.
        
        Example:
            >>> my_list = LinkedList()
            >>> my_list.add(1)
            >>> my_list.head.data
            1
            >>> my_list.add(2)
            >>> my_list.head.next.data
            2
        """
        new_node = LinkedListNode(data)
        if self.head is None:
            self.head = new_node
        else:
            current_node = self.head
            while current_node.next is not None:
                current_node = current_node.next
            current_node.next = new_node
    
    # Problem 2: Implement the __str__ method so that a LinkedList instance can
    #   be printed out the same way that Python lists are printed.
    def __str__(self):
        """String representation: the same as a standard Python list.
        
        Example:
            >>> my_list = LinkedList()
            >>> my_list.add(1)
            >>> my_list.add(2)
            >>> my_list.add(3)
            >>> print(my_list)
            [1, 2, 3]
            >>> str(my_list) == str([1,2,3])
            True
        """
        data_list = []
        current_node = self.head
        while current_node != None:
            data_list.append(current_node.data)
            current_node = current_node.next
        return str(data_list)

    # Problem 3: Finish implementing LinkedList.remove() so that if the node
    #   is not found, an exception is raised.
    def remove(self, data):
        """Remove the node containing 'data'. If the list is empty, or if the
        target node is not in the list, raise a ValueError with error message
        "<data> is not in the list."
        
        Example:
            >>> print(my_list)
            [1, 2, 3]
            >>> my_list.remove(2)
            >>> print(my_list)
            [1, 3]
            >>> my_list.remove(2)
            2 is not in the list.
            >>> print(my_list)
            [1, 3]
        """
        if self.head is None:
            raise ValueError(str(data) +  " is not in the list.")

        if self.head.data == data:
            self.head = self.head.next
        else:
            current_node = self.head
            while current_node.next.data != data:
                if current_node.next.next is None:
                    raise ValueError(str(data) + " is not in the list.")
                current_node = current_node.next

            new_next_node = current_node.next.next
            current_node.next = new_next_node

    # Problem 4: Implement LinkedList.insert().
    def insert(self, data, place):
        """Create a new Node containing 'data'. Insert it into the list before
        the first Node in the list containing 'place'. If the list is empty, or
        if there is no node containing 'place' in the list, raise a ValueError
        with error message "<place> is not in the list."
        
        Example:
            >>> print(my_list)
            [1, 3]
            >>> my_list.insert(2,3)
            >>> print(my_list)
            [1, 2, 3]
            >>> my_list.insert(2,4)
            4 is not in the list.
        """
        new_node = LinkedListNode(data)
        if self.head is None:
            raise ValueError(str(place) +  " is not in the list.")

        if self.head.data == place:
            new_node.next = self.head
            self.head = new_node
        else:
            current_node = self.head
            while current_node.next.data != place:
                if current_node.next.next is None:
                    raise ValueError(str(place) + " is not in the list.")
                current_node = current_node.next
            new_node.next = current_node.next
            current_node.next = new_node

class DoublyLinkedListNode(LinkedListNode):
    """A Node class for doubly-linked lists. Inherits from the 'Node' class.
    Contains references to the next and previous nodes in the list.
    """
    def __init__(self,data):
        """Initialize the next and prev attributes."""
        Node.__init__(self,data)
        self.next = None
        self.prev = None

# Problem 5: Implement this class.
class DoublyLinkedList(LinkedList):
    """Doubly-linked list data structure class. Inherits from the 'LinkedList'
    class. Has a 'head' for the front of the list and a 'tail' for the end.
    """
    def __init__(self):
        self.head = None
        self.tail = None

    def add(self, data):
        new_node = DoublyLinkedListNode(data)

        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
    def remove(self, data):
        if self.head is None:
            raise ValueError(str(data) +  " is not in the list.")
        if self.head.data == data:
            self.head = self.head.next
        else:
            current_node = self.head
            while current_node.next.data != data:
                if current_node.next.next is None:
                    raise ValueError(str(data) + " is not in the list.")
                current_node = current_node.next
            if current_node.next.data == self.tail.data:
                self.tail = current_node
                self.tail.next = None

            else:
                delete_node = current_node.next
                new_next_node = delete_node.next
                new_prev_node = delete_node.prev
                new_next_node.prev = new_prev_node
                new_prev_node.next = new_next_node


    def insert(self, data, place):
        new_node = DoublyLinkedListNode(data)
        if self.head is None:
            raise ValueError(str(place) +  " is not in the list.")
        if self.head.data == place:
            if self.head.data ==self.tail.data:

                new_node.next = self.head
                self.head = new_node
                self.tail = self.head.next
                self.tail.prev = self.head
            else:
                new_node.next = self.head
                self.head = new_node
        else:
            current_node = self.head
            while current_node.next.data != place:
                if current_node.next.next is None:
                    raise ValueError(str(place) + " is not in the list.")
                current_node = current_node.next
            new_next = current_node.next
            current_node.next = new_node
            new_node.next = new_next
            new_next.prev = new_node
            new_node.prev = current_node

# Problem 6: Implement this class. Use an instance of your object to implement
# the sort_words() function in solutions.py.
class SortedLinkedList(DoublyLinkedList):
    """Sorted doubly-linked list data structure class."""

    # Overload add() and insert().
    def add(self, data):
        """Create a new Node containing 'data' and insert it at the
        appropriate location to preserve list sorting.
        
        Example:
            >>> print(my_list)
            [3, 5]
            >>> my_list.add(2)
            >>> my_list.add(4)
            >>> my_list.add(6)
            >>> print(my_list)
            [2, 3, 4, 5, 6]
        """
        new_node = DoublyLinkedListNode(data)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            current_node = self.head
            while data > current_node.data:
                if current_node.next is None:
                    break 
                current_node = current_node.next
            if self.head.data == self.tail.data:
                if data > current_node.data:
                    new_node.prev = current_node
                    self.tail = new_node
                    current_node.next = new_node
                else:
                    current_node.prev = new_node
                    new_node.next = current_node
                    self.head = new_node
            elif current_node.next is None and data > current_node.data:
                current_node.next = new_node
                self.tail = new_node
                new_node.prev = current_node
            elif current_node.prev is None:
                current_node.prev = new_node
                new_node.next = current_node
                self.head = new_node
            else:
                new_node.next = current_node
                new_node.prev = current_node.prev
                current_node.prev.next = new_node
                current_node.prev = new_node
            
    def insert(self, *args):
        raise ValueError("insert() has been disabled for this class.")

        
def add_linked():
    my_list = SortedLinkedList()
    my_list.add(1)
    my_list.add(3)
    my_list.add(2)
    my_list.add(4)
    my_list.add(0)
    my_list.add(6.5)
    my_list.add(7)
    my_list.add(-2)
    my_list.add('super')
    my_list.add('useful')
    my_list.add('hope')
    my_list.add(-2.5)
    #my_list.add(0)
    #my_list.add()
    print my_list.head
    print my_list.tail
    print my_list         
def remove_test():
    my_list = LinkedList()
    my_list.add(1)
    print my_list
    #my_list.add(3)
    #my_list.add("super")
    my_list.remove(1)

    #print my_list

def remove_double_test():
    my_list = DoublyLinkedList()
    my_list.add(1)
    """my_list.add(2)
    my_list.add(3)
    print my_list.head.next.next.prev.prev, " should be 1"
    my_list.remove(3)
    print my_list.head.next.prev, " should be 1"""
    my_list.remove(2)
    print my_list.head
    print my_list.tail
    print my_list
def insert_test():
    my_list = LinkedList()
    my_list.add(1)
    my_list.add(2)
    my_list.add(4)
    my_list.insert(4,5)
    print my_list
def insert_double_test():
    my_list = DoublyLinkedList()
    my_list.add(1)
    my_list.add(4)
    my_list.add(5)
    my_list.insert(0,1)
    my_list.insert(2,5)
    print my_list
if __name__ == '__main__':
    #add_linked()
    #insert_double_test()
    remove_double_test()
    #remove_test()
    #insert_test()
# =============================== END OF FILE =============================== #