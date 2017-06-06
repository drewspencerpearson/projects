# spec.py
"""Volume II Lab 8: Markov Chains
Drew Pearson
Math 321O
October 29, 2015
"""
import numpy as np

# Problem 1: implement this function.
def random_markov(n):
    """Create and return a transition matrix for a random
    Markov chain with 'n' states as an nxn NumPy array.
    """
    A = np.random.rand(n,n)
    collumn_sum = np.sum (A, axis = 0)
    for i in xrange(0,n):
        for j in xrange(0,n): 
            #row_sum = sum(A[i])
            #collumn_sum = sum(A[:,j])
            A[i][j] = (A[i][j])/collumn_sum[j]
    return A 





# Problem 2: modify this function.
def forecast(num_days):
    """Run a simulation for the weather over 'num_days' days, with
    "hot" as the starting state. Return a list containing the day-by-day
    results, not including the starting day.

    Example:
        >>> forecast(5)
        [1, 0, 0, 1, 0]
        # Or, if you prefer,
        ['cold', 'hot', 'hot', 'cold', 'hot']
    """
    transition_matrix = np.array([[0.7,0.6],[0.3,0.4]])
    prev = 0
    n = 0
    my_list = []
    while n < num_days:
        rand = np.random.random()
        if prev == 0:
            if rand < .3:
                my_list.append(1)
                prev = 1
            else:
                my_list.append(0)
                prev = 0
        else:
            if rand < .4:
                my_list.append(1)
                prev = 1
            else:
                my_list.append(0)
                prev = 0
        n+=1
    return my_list


# Problem 3: implement this function.
def four_state_forecast(days=1):
    """Same as forecast(), but using the four-state transition matrix."""
    W = np.array([[.5,.3,.1,0],[.3,.3,.3,.3],[.2,.3,.4,.5],[0,.1,.2,.2]])
    n = 0
    my_array = np.array([1,0,0,0])
    my_list = []

    while n < days:
        if my_array[0] ==1:
            if n != 0:
                my_list.append(0)
            my_array = np.random.multinomial(1, W[:,0])
        elif my_array[1] == 1:
            my_list.append(1)
            my_array = np.random.multinomial(1, W[:,1])
        elif my_array[2] == 1:
            my_list.append(2)
            my_array = np.random.multinomial(1, W[:,2])
        else:
            my_list.append(3)
            my_array = np.random.multinomial(1, W[:,3])
        n +=1
    return my_list




# Problem 4: implement this function.
def analyze_simulation():
    """Analyze the results of the previous two problems. What percentage
    of days are in each state? Print your results to the terminal.
    """
    my_list = four_state_forecast(100000)
    i = 0
    print "For the four_state_forecast method:" 
    while i < 4:
        percentage = float(my_list.count(i))/(len(my_list))*100
        print "percent of " +str(i) + " = " + str(percentage)
        i +=1

    my_list2 = forecast(100000)
    j = 0
    print "For the forecast method:"
    while j <2:
        percentage2 = float(my_list2.count(j))/(len(my_list2))*100
        print "percent of " + str(j) + " = " + str(percentage2)
        j +=1



# Problems 5-6: define and implement the described functions.
def convert_text(filename):
    words = ["$tart"]
    with open(filename, 'r') as myfile:
        contents = myfile.readlines()

    with open(filename + "output.txt", 'w') as myfile:

        for content in contents:
            content = content.replace('\n', '')
            content_list = content.split(" ")
            for item in content_list:
                if item not in words:
                    words.append(item)
                    myfile.write(str(words.index(item))+ " ")
                else:
                    myfile.write(str(words.index(item))+ " ")
            myfile.write("\n")

    words.append('en&')
    return words

def transition_matrix(filename, unique_word_count = None):
    #total_words = convert_text(filename)
    n = unique_word_count+2
    A = np.zeros((n,n))

    with open(filename, 'r') as myfile:
        content = myfile.readlines()

    for content in content:
        content = content.replace('\n', '')
        all_numbers = content.split(" ")
        while "" in all_numbers:
            all_numbers.remove("")
        for i in xrange(len(all_numbers)):
            if i == 0:
                A[int(all_numbers[i])][0] +=1
            else:
                A[int(all_numbers[i])][prev_index] += 1
            prev_index = int(all_numbers[i])
        A[n-1][prev_index] +=1

    collumn_sum = np.sum(A, axis = 0)
    for j in xrange(0,n-1): 
        A[:,j] = (A[:,j])/collumn_sum[j]

    return A 


# Problem 7: implement this function.
def sentences(infile, outfile, num_sentences=1):
    """Generate random sentences using the word list generated in
    Problem 5 and the transition matrix generated in Problem 6.
    Write the results to the specified outfile.

    Parameters:
        infile (str): The path to a filen containing a training set.
        outfile (str): The file to write the random sentences to.
        num_sentences (int): The number of random sentences to write.

    Returns:
        None
    """

    tot_words = convert_text(infile)
    #trans_matrix = transition_matrix(infile + 'output.txt', len(tot_words)-2)
    def find_index(vector):
        return vector.argsort()[-1]
    def numbers_to_words(index_list):
        string = ""
        for i in index_list[:-1]:
            string += tot_words[i]
            string +=" "
        return string 

    with open(outfile, 'w') as myfile:
        for x in xrange(num_sentences):
            word_list = []
            trans_matrix = transition_matrix(infile + 'output.txt', len(tot_words)-2)
            status = 0
            while status != len(tot_words)-1:
                status = find_index(np.random.multinomial(1, trans_matrix[:,status]))
                word_list.append(status)
            myfile.write(numbers_to_words(word_list))
            myfile.write("\n")


        



def test_markov():
    print random_markov(3)
    print np.sum(random_markov(3))

def test_probem_2():
    print forecast(26)

def test_problem_3():
    print four_state_forecast(15)
def analyzing():
    pass

def test():
    sentences('taylor.txt', 'test3.txt', 15)

if __name__ == '__main__':

    test()
    #test_markov()
    #test_probem_2()
    #test_problem_3()
    #analyze_simulation()