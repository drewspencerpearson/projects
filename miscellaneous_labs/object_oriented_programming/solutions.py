# name this file 'solutions.py'
"""Volume II Lab 2: Object Oriented Programming
Drew Pearson
Class 321
9/10/15
"""
import math 
from Backpack import Backpack

my_backpack = Backpack()
your_backpack = Backpack()

my_backpack.put('tape')
my_backpack.put('scissors')
your_backpack.put('tape')
my_backpack = your_backpack


# Problem 1: Modify the 'Backpack' class in 'Backpack.py'.

# Study the 'Knapsack' class in 'Backpack.py'. You should be able to create a 
#   Knapsack object after finishing problem 1.


# Problem 2: Write a 'Jetpack' class that inherits from the 'Backpack' class.
class Jetpack(Backpack):
    """A Jetpack object class. Inherits from the Backpack class.
    However, it's initial color is silver and its initial max size is 2.
    A jetpack class also has an additional attribute - feul (int) which if not given
    is preset to 10. 
    """
    def __init__(self, color='silver', name='jetpack', max_size=2, fuel = 10):
        """constructor for a jetback object
        Inputs: 
        color (str, opt) default is silver
        name(str, opt) default is jetpack
        max_size (int, opt) deafult is 2
        feul (int, opt) default is 10
        returns a jetpack with no contents"""
        Backpack.__init__(self, color, name, max_size)
        """ this calls the backpack object which is used to create the Jetpack object"""
        self.fuel = fuel
    def fly(self, item):
        """ function that takes the amount of fuel a certain trip requires and adjusts the fuel accordingly"""
        if item > self.fuel:
            print "Not enough fuel!"
        else:
            self.fuel = self.fuel-item
    def dump(self):
        """ This function deletes all the items from the jetpack and removes all feul"""
        Backpack.dump(self)
        self.fuel = 0



"""my_jetpack = Jetpack()
print my_jetpack.color
print my_jetpack.name
print my_jetpack.max_size
print my_jetpack.fuel
my_jetpack.fly(11)
my_jetpack.fly(9)
print my_jetpack.fuel
my_jetpack.put('tape')
print my_jetpack.contents
my_jetpack.dump()
print my_jetpack.contents
print my_jetpack.fuel
my_jetpack.fly(3)
my_jetpack.name = 'gold'
print my_jetpack.name"""

# Problem 3: write __str__ and __eq__ for the 'Backpack' class in 'Backpack.py'


# Problem 4: Write a ComplexNumber class.
class ComplexNumber(object):
    """A Complex number object class. Has a real and imaginary part.
    
    Attributes:
        real (int): the real part of the complex number.
        imag (int): the imaginary part of the complex number.
    """
    def __init__(self, real, imag):
        """initiates the comple number object with a real and imaginary part"""
        self.real = real
        self.imag = imag
    def conjugate(self):
        """finds the conjugate of the created complex number and returns a new complex number object"""
        return ComplexNumber(self.real, -self.imag)

    def norm(self):
        """Finds the norm of the Complex number and returns the norm"""
        real_squared = self.real**2
        imag_squared = self.imag**2

        return math.sqrt(real_squared+imag_squared)

    def __add__(self, other):
        """ adds one complex number (self) to another complex number (other) by adding the real and imaginary parts. """
        add_real = self.real + other.real
        add_imag = self.imag + other.imag
        return ComplexNumber(add_real, add_imag)

    def __sub__(self, other):
        """ Subtracts one complex number (self) from another complex number (other) by subtracting the real and imaginary parts"""
        sub_real = self.real - other.real
        sub_imag = self.imag - other.imag
        return ComplexNumber(sub_real, sub_imag)

    def __mul__(self, other):
        """Multiplies two Complex numbers called self and real. First we find the real part by multiplying both real parts 
        together and subtract the imaginary parts multiplied together. Next we find the imaginary part by multiplying the self real and the
        other imaginary + self imaginary and other real. """

        new_real = (self.real * other.real) - (self.imag * other.imag)
        new_imag = (self.real * other.imag) + (self.imag * other.real)
        return ComplexNumber(new_real, new_imag) 

    def __div__(self, other):
        """Divides two Complex Numbers, self and other. First we find the denomonator of the division
        labeled den, by squaring the real of other and adding it to the square of the imaginary of other
        Next we find the coeffecients of the real by using the same idea as the mul magic. Same with the imaginary
        coefficient. then we divide those coeffecients by den (denomonator) and we return a new Complex Number object. """
        den = other.real**2 + other.imag**2
        new_real = (self.real * other.real) - (self.imag * (-other.imag))
        new_imag = (self.real * (-other.imag)) + (self.imag * other.real)
        
        
        new_real = new_real/float(den)
        new_imag = new_imag/float(den)
        return ComplexNumber(new_real, new_imag)
"""CN = ComplexNumber(2.0,3)
CN2 = ComplexNumber(-1,2.0)

CN3 = CN/CN2 #does he want me to save the result as self?
print CN3.real"""

# =============================== END OF FILE =============================== #