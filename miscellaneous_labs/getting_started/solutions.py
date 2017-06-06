# name this file 'solutions.py'
"""Volume II Lab 1: The Standard Library
Drew Pearson
321
9/4/15
"""

# Add import statements here.
# In future labs, do not modify any PROVIDED import statements.
# You may always add others as needed.
import matrix_multiply
import sys
import math
import calculator 
import time 
# Problem 1: Implement this function.
def prob1(l):
    """Accept a list 'l' of numbers as input and return a new list with the
    minimum, maximum, and average of the contents of 'l'.
    """
    minimum = min(l)
    maximum = max(l)
    average = sum(l)*1.0/len(l)
    my_list = [minimum, maximum, average]
    return my_list

# Problem 2: Implement this function.
def prob2():
    """Determine which Python objects are mutable and which are immutable. Test
    numbers, strings, lists, tuples, and dictionaries. Print your results to the
    terminal using the print() function.
    """
    """list_1 = [1,2,3]
    list_2 = list_1
    list_2.append(4)
    print list_2
    print list_1
    number_1 = 3
    number_2 = number_1
    number_2 +=1
    print number_2
    print number_1
    string_1 = 'Hello'
    string_2 = string_1
    string_2 += 'a'
    print string_2
    print string_1
    dictionary_1 = {'dog' : 1, 'cat' : 2, 'cow' : 3}
    dictionary_2 = dictionary_1
    dictionary_2[1] = 'a'
    print dictionary_2
    print dictionary_1"""
    print "lists are mutable, tuples are immutable, numbers are immutable, strings are immutable, and dictionaries are mutable."


# Problem 3: Create a 'calculator' module and use it to implement this function.
def prob3(a,b):
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any methods other than those that are imported from the
    'calculator' module.
    
    Parameters:
        a (float): the length one of the sides of the triangle.
        b (float): the length the other nonhypotenuse side of the triangle.
    
    Returns:
        The length of the triangle's hypotenuse.
    """
    import calculator

    leg_1 = calculator.product(a,a)
    leg_2 = calculator.product(b,b)
    sum_squared = calculator.sum(leg_1,leg_2)
    hypotenuse = calculator.square_root(sum_squared)
    return hypotenuse


# Problem 4: Utilize the 'matrix_multiply' module and 'matrices.npz' file to
#   implement this function.
def prob4():
    """If no command line argument is given, print "No Input."
    If anything other than "matrices.npz is given, print "Incorrect Input."
    If "matrices.npz" is given as a command line argument, use functions
    from the provided 'matrix_multiply' module to load two matrices, then
    time how long each method takes to multiply the two matrices together.
    Print your results to the terminal.
    how do I load the files correctly? how do I print to the terminal?
    """
    if len(sys.argv) <2 : 
        print "No Input."
    elif sys.argv[1] == 'matrices.npz':

        A,B = matrix_multiply.load_matrices('matrices.npz')
        start1 = time.time()
        matrix_multiply.method1(A,B)
        end1 = time.time()
        C =  (end1-start1)
        #time1 = str(end1-start1)

        start2 = time.time()
        matrix_multiply.method2(A,B)
        end2 = time.time()
        #print (end2-start2)
        D = (end2-start2)
        #time2 = str(end2 -time2)

        start3 = time.time()
        matrix_multiply.method3(A,B)
        end3 = time.time()
        E = (end3-start3)
        #print (end3-start3)

        #time3 = str(end3-start3)

        print "time for method 1: " + str(C)
        print "time for method 2: " + str(D)
        print "time for method 3: " + str(E)
    else :
        print "Incorrect Input."




# Everything under this 'if' statement is executed when this file is run from
#   the terminal. In this case, if we enter 'python solutions.py word' into
#   the terminal, then sys.argv is ['solutions.py', 'word'], and prob4() is
#   executed. Note that the arguments are parsed as strings. Do not modify.
if __name__ == "__main__":
    prob4()