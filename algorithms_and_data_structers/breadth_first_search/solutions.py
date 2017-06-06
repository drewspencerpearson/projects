#solutions.py
"""Volume II Lab 7: Breadth-First Search (Kevin Bacon)
Drew Pearson
Math 321
10/20/15
"""
from collections import deque
import networkx as nx 

# Problems 1-4: Implement the following class
class Graph(object):
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a list of the
    corresponding node's neighbors.

    Attributes:
        dictionary: the adjacency list of the graph.
    """

    def __init__(self, adjacency):
        """Store the adjacency dictionary as a class attribute."""
        self.dictionary = adjacency

    # Problem 1
    def __str__(self):
        """String representation: a sorted view of the adjacency dictionary.
        
        Example:
            >>> test = {'A':['B'], 'B':['A', 'C',], 'C':['B']}
            >>> print(Graph(test))
            A: B
            B: A; C
            C: B
        """
        keys = sorted(self.dictionary.keys())
        my_list = []
        for k in keys:
            v = sorted(self.dictionary[k])
            s = '; '
            my_list.append(str(k) + ': ' + s.join(v))
        t = '\n'

        return t.join(my_list)


    # Problem 2
    def traverse(self, start):
        """Begin at 'start' and perform a breadth-first search until all
        nodes in the graph have been visited. Return a list of values,
        in the order that they were visited.

        Inputs:
            start: the node to start the search at.

        Returns:
            the list of visited nodes (in order of visitation).

        Raises:
            ValueError: if 'start' is not in the adjacency dictionary.

        Example:
            >>> test = {'A':['B'], 'B':['A', 'C',], 'C':['B']}
            >>> Graph(test).traverse('B')
            ['B', 'A', 'C']
        """
        if start not in self.dictionary.keys():
            raise ValueError(str(start) + 'is not in the graph')
        current = start
        marked = set()
        visited = list()
        visit_que = deque()
        marked.add(current)

        while current is not None: 
            visited.append(current)
            for neighbor in self.dictionary[current]:
                #print neighbor
                if neighbor not in marked:
                    visit_que.append(neighbor)
                    marked.add(neighbor)
            if len(visit_que) == 0:
                current = None
            else:
                current = visit_que.popleft()
                 
        return visited


    # Problem 3 (Optional)
    def DFS(self, start):
        """Begin at 'start' and perform a depth-first search until all
        nodes in the graph have been visited. Return a list of values,
        in the order that they were visited. If 'start' is not in the
        adjacency dictionary, raise a ValueError.

        Inputs:
            start: the node to start the search at.

        Returns:
            the list of visited nodes (in order of visitation)
        """
        if start not in self.dictionary.keys():
            raise ValueError(str(start) + 'is not in the graph')
        current = start
        marked = set()
        visited = list()
        visit_que = deque()
        marked.add(current)

        while current is not None:
            visited.append(current)
            for neighbor in self.dictionary[current]:
                #print neighbor
                if neighbor not in marked:
                    visit_que.append(neighbor)
                    marked.add(neighbor)
            if len(visit_que) ==0:
                current = None
            else:
                current = visit_que.pop()
                 
        return visited

    # Problem 4
    def shortest_path(self, start, target):
        """Begin at the node containing 'start' and perform a breadth-first
        search until the node containing 'target' is found. Return a list
        containg the shortest path from 'start' to 'target'. If either of
        the inputs are not in the adjacency graph, raise a ValueError.

        Inputs:
            start: the node to start the search at.
            target: the node to search for.

        Returns:
            A list of nodes along the shortest path from start to target,
                including the endpoints.

        Example:
            >>> test = {'A':['B', 'F'], 'B':['A', 'C'], 'C':['B', 'D'],
            ...         'D':['C', 'E'], 'E':['D', 'F'], 'F':['A', 'E', 'G'],
            ...         'G':['A', 'F']}
            >>> Graph(test).shortest_path('A', 'G')
            ['A', 'F', 'G']
        """
        if start not in self.dictionary.keys() or target not in self.dictionary.keys():
            raise ValueError('the node is not in the graph')
        current = start
        marked = set()
        visited = list()
        visit_que = deque()
        marked.add(current)
        path = dict()

        while current is not target:
            if current == start:
                path[current] = None
            visited.append(current)
            for neighbor in self.dictionary[current]:
                if neighbor not in marked:
                    visit_que.append(neighbor)
                    path[neighbor] = current
                    marked.add(neighbor)
            if len(visit_que) == 0:
                current = None
            else:
                current = visit_que.popleft()
                 
        my_list = []
        i = target
        while i is not None:
            my_list.append(i)
            i = path[i]
        return my_list[::-1]


# Problem 5: Write the following function
def convert_to_networkx(dictionary):
    """Convert 'dictionary' to a networkX object and return it."""
    nx_graph = nx.Graph()

    for x in dictionary.keys():
        for y in dictionary[x]:
            nx_graph.add_edge(x,y)
    return nx_graph




# Helper function for problem 6
def parse(filename="movieData.txt"):
    """Generate an adjacency dictionary where each key is
    a movie and each value is a list of actors in the movie.
    """

    # open the file, read it in, and split the text by '\n'
    with open(filename, 'r') as movieFile:
        moviesList = movieFile.read().split('\n')
    graph = dict()

    # for each movie in the file,
    for movie in moviesList:
        # get movie name and list of actors
        names = movie.split('/')
        title = names[0]
        graph[title] = []
        # add the actors to the dictionary
        for actor in names[1:]:
            graph[title].append(actor)
    
    return graph


# Problems 6-8: Implement the following class
class BaconSolver(object):
    """Class for solving the Kevin Bacon problem."""

    # Problem 6
    def __init__(self, filename="movieData.txt"):
        """Initialize the networkX graph and with data from the specified
        file. Store the graph as a class attribute. Also store the collection
        of actors in the file as an attribute.
        """
        movie_dictionary = parse(filename)
        self.movie_dictionary = convert_to_networkx(movie_dictionary)
        actor_list = []
        for key in movie_dictionary.keys():
            for actor in movie_dictionary[key]:
                if actor not in actor_list:
                    actor_list.append(actor)
        self.actor_list = actor_list

    # Problem 6
    def path_to_bacon(self, start, target="Bacon, Kevin"):
        """Find the shortest path from 'start' to 'target'."""
        if start not in self.actor_list or target not in self.actor_list:
            raise ValueError("actor is not in the data set")
        return nx.shortest_path(self.movie_dictionary, start, target)

    # Problem 7
    def bacon_number(self, start, target="Bacon, Kevin"):
        """Return the Bacon number of 'start'."""
        my_list = self.path_to_bacon(start)
        return (len(my_list)-1)/2


    # Problem 7
    def average_bacon(self, target="Bacon, Kevin"):
        """Calculate the average Bacon number in the data set.
        Note that actors are not guaranteed to be connected to the target.

        Inputs:
            target (str): the node to search the graph for
        """
        total_sum = 0.0
        not_connected = 0
        for i in self.actor_list:
            try: 
                total_sum += self.bacon_number(i)
            except nx.NetworkXNoPath:
                not_connected += 1
        return total_sum/(len(self.actor_list)-not_connected), not_connected




def test_print():
    my_dictionary = {'A':['B'], 'B':['A', 'C',], 'C':['B']}
    graph = Graph(my_dictionary)
    print graph

def test_traverse():
    my_dictionary = {'A':['D','B'], 'B':['A', 'D', 'C'], 'C':['B','D'], 'D':['C','B','A']}
    #print my_dictionary['A']
    return Graph(my_dictionary).traverse("A")

    #return graph.traverse('B')
def test_DFS():
    my_dictionary = {'A':['D','B'], 'B':['A', 'D', 'C'], 'C':['B','D'], 'D':['C','B','A']}
    #print my_dictionary['A']
    return Graph(my_dictionary).DFS("A")

def test_shortest():
    my_dictionary = {'A':['B', 'F'], 'B':['A', 'C'], 'C':['B', 'D'], 'D':['C', 'E'], 'E':['D', 'F'], 'F':['A', 'E', 'G'], 'G':['A', 'F']}
    return Graph(my_dictionary).shortest_path('G','T')
    #my_dictionary = {'A':['D','B'], 'B':['A', 'D', 'C'], 'C':['B','D'], 'D':['C','B','A']}

def test_bacon_search():
    b = BaconSolver()
    #print b.path_to_bacon('Jackson, Samuel L.')
    #print b.path_to_bacon('', 'Bacon, Kevin')
    #print b.path_to_bacon('Cruise, Tom')
    print b.path_to_bacon('Cruis, Samuel', 'Bacon, Kevin')
def test_average_bacon():
    b = BaconSolver()
    return b.average_bacon()
if __name__ == '__main__':

    #test_print()
    #print test_traverse()
    #print test_DFS()
    #print test_average_bacon()
    test_bacon_search()
# =========================== END OF FILE =============================== #



