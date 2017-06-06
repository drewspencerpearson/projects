# Backpack.py
"""Volume II Lab 2: Object Oriented Programming (Auxiliary file)
Modify this file for problems 1 and 3.
Drew Pearson
321
9/11/15
"""
import math 
# Problem 1: Modify this class. Add 'name' and max_size' attributes, modify
#   the put() method, and add a dump() method. Remember to update docstrings.
class Backpack:
    """A Backpack object class. Has a color, name, max size, and a list of contents.
    
    Attributes:
        color (str): the color of the backpack.
        contents (list): the contents of the backpack.
        name (str): name of the backpack.
        max_size(int) the max amount of contents in a backpack.
    """
    
    def __init__(self, color='black', name = 'backpack', max_size = 5):
        """Constructor for a backpack object.
        Set the color, name, and max size and initialize the contents list.
        
        Inputs:
            color (str, opt): the color of the backpack. Defaults to 'black'.
            name (str, opt): is default to "backpack."
            max_size (int, opt): is default to 5.
        
        Returns:
            A backpack object wth no contents, but with a color, name, and max size. 
        """
        
        self.color = color
        self.contents = []
        self.name = name
        self.max_size = max_size


    def put(self, item):
        """Add 'item' to the backpack's content list."""
        if len(self.contents) == self.max_size:

            print "Backpack Full"
        else:
            self.contents.append(item)
    
    def take(self, item):
        """Remove 'item' from the backpack's content list."""
        self.contents.remove(item)

    def dump(self):
        """removes all the contents from the backpack"""
        self.contents = []

    
    
    # -------------------- Magic Methods (Problem 3) -------------------- #
    
    def __add__(self, other):
        """Add the contents of 'other' to the contents of 'self'.
        Note that the contents of 'other' are unchanged.
        
        Inputs:
            self (Backpack): the backpack on the left-hand side
                of the '+' addition operator.
            other (Backpack): The backpack on the right-hand side
                of the '+' addition operator.
        """
        self.contents = self.contents + other.contents
    
    def __lt__(self, other):
        """Compare two backpacks. If 'self' has fewer contents
        than 'other', return True. Otherwise, return False.
        
        Inputs:
            self (Backpack): the backpack on the left-hand side
                of the '<' comparison operator.
            other (Backpack): The backpack on the right-hand side
                of the '<' comparison operator.
        """    
        return len(self.contents) < len(other.contents)
    
    # Problem 3: write the __str__ and __eq__ magic methods for the Backpack.
    def __str__(self):
        """String Representation: a list of the backpack's attributes.
        
        Examples:                           |
            >>> b = Backpack()              |   Or,
            >>> b.put('something')          |
            >>> b.put('something else')     |   >>> c = Backpack('red','Bob',3)
            >>> print(b)                    |   >>> print(c)
            Name:       backpack            |   Name:       Bob
            Color:      black               |   Color:      red
            Size:       2                   |   Size:       0
            Max Size:   5                   |   Max Size:   3
            Contents:                       |   Contents:   Empty
                        something           |
                        something else      |
        """
        x = "Name:\t\t{}\nColor:\t\t{}\nSize:\t\t{}\nMax Size:\t{}\n".format(self.name,self.color,len(self.contents),self.max_size)
        content = str()
        if len(self.contents)==0:
            content = "\tEmpty"
        else:   
            for i in self.contents:
                content += "\n\t\t\t" + str(i) 

        
        return x + "Contents:" + content
    def __eq__(self,other):
        """method to determine if two backpacks (self and other) are equal"""
        same_both = True
        same_self = True
        same_other = True 
        if self.name != other.name:
            same_self = False
            same_other = False
        if self.color != other.color:
            same_self = False 
            same_other = False
        for i in self.contents:
            if i not in other.contents:
                same_self = False
        for i in other.contents:
            if i not in self.contents:
                same_other = False
        if (same_other == False) or (same_self == False):
            same_both = False        
        return same_both 


# Study this example of inheritance. You are not required to modify it.
class Knapsack(Backpack):
    """A Knapsack object class. Inherits from the Backpack class.
    A knapsack is smaller than a backpack and can be tied closed.
    
    Attributes:
        color (str): the color of the knapsack.
        name (str): the name of the knapsack.
        max_size (int): the maximum number of items that can fit in the
            knapsack.
        contents (list): the contents of the backpack.
        closed (bool): whether or not the knapsack is tied shut.
    """
    
    def __init__(self, color='brown', name='knapsack', max_size=3):
        """Constructor for a knapsack object. A knapsack only holds 3 item by
        default instead of 5. Use the Backpack constructor to initialize the
        name and max_size attributes.
        
        Inputs:
            color (str, opt): the color of the knapsack. Defaults to 'brown'.
            name (str, opt): the name of the knapsack. Defaults to 'knapsack'.
            max_size (int, opt): the maximum number of items that can be
                stored in the knapsack. Defaults to 3.
        
        Returns:
            A knapsack object with no contents.
        """
        
        Backpack.__init__(self, color, name, max_size)
        self.closed = True
    
    def put(self, item):
        """If the knapsack is untied, use the Backpack put() method."""
        if self.closed:
            print "Knapsack closed!"
        else:
            Backpack.put(self, item)
    
    def take(self, item):
        """If the knapsack is untied, use the Backpack take() method."""
        if self.closed:
            print "Knapsack closed!"
        else:
            Backpack.take(self, item)
    
    def untie(self):
        """Untie the knapsack."""
        self.closed = False
    
    def tie(self):
        """Tie the knapsack."""
        self.closed = True

# ============================== END OF FILE ================================ #