# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import heapq
import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

    stack = []  # Fringe which is a stack in a dfs

    visited = []  # List which keeps visited nodes
    path = []  # Array which keeps the path of a Node
    # adds initial start state node to the stack with the node's empty path
    stack.append((problem.getStartState(), path))

    while stack:
        # print("stack: ", stack)
        # print("visited: ", visited)
        # print("path: ", path)

        currentNode, path = stack.pop()  # Get the node and its path from the fringe
        # C , path = a-c
        # print("current node: ", node)
        # print("current successor: ", successor)

        # Check if the current node is the goal node
        if problem.isGoalState(currentNode):

            return path  # Return goal node's path

        elif currentNode not in visited:  # check if the current node was visited already
            # path.append(successor)
            # if it wasn't, it is then added to the visited array
            visited.append(currentNode)

        # Code was not working, This fixed it.
        elif currentNode in visited:
            # leaf node
            continue

        # get the neighbour node, it's path to a neighbor node and cost
        # We dont use cost in this solution but everything needs to be extracted
        for neighbourNode, successPath, cost in problem.getSuccessors(currentNode):

            # add the current node's path to the neighbour's path
            successor_path = path + [successPath]
            # add the neighbour node to the stack with its new path
            stack.append((neighbourNode,  successor_path))
    return path
    # util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()  # Initailize the Fringe which is a queue in a bfs

    visited = []  # List which keeps visited nodes
    path = []  # List which keeps path of a node

    # Push the starting state to the queue with its path
    # Start with Empty path because A has no path because it is the start
    queue.push((problem.getStartState(), path))

    while queue:  # while the queue is not empty, do

        # print("queue: ", queue)
        # print("visited: ", visited)
        # print("path: ", path)

        node, path = queue.pop()  # Pop the current node and its path from the queue

        # print("current node: ", node)
        # print("current successor: ", successor)
        if problem.isGoalState(node):  # check if the current node is the goal node

            # path.append(successor)

            return path  # return its the current node's path

        elif node not in visited:  # check if current node is not in the visited list
            # path.append(successor)
            visited.append(node)  # add current node to the visited list

        elif node in visited:
            # leaf node
            continue

        for neighbourNode, successPath, cost in problem.getSuccessors(node):

            # add the current node's path to the neighbour's path
            successor_path = path + [successPath]

            # push the neighbour node and its new path to the queue
            queue.push((neighbourNode, successor_path))
    return path
    # util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()  # Initialize the fringe
    # push the start state to the fringe, with a starting cost of 0
    fringe.push((problem.getStartState(), [], 0), 0)

    visited = set()  # Set which keeps visited nodes

    while fringe:
     # Get the current node, its current path and the cost
        node,  path, cost = fringe.pop()

        if problem.isGoalState(node):  # Check if the current node is the goal node
            return path  # Return node path if it is the goal node

        elif node not in visited:  # check if current node is not in the visited list
            # path.append(successor)
            visited.add(node)  # add current node to the visited list

        elif node in visited:
            # leaf node
            continue
        # Get a neighboring node of the current node
        for successor in problem.getSuccessors(node):
            # The states are saved as, for example, ('B', 'B->C', 0)
          # Get each of the successor components and store them in variables = (successor_node, successor_path, successor_cost)
            successor_node = successor[0]
            successor_path = successor[1]
            successor_cost = successor[2]
            # add the current node cost to the neighboring node cost to get the new_cost
            new_cost = cost + successor_cost
            # print(successor_node)
            # Check if the successor(neighbour) state is not in the the visited set
            if successor not in visited:
                # push it to the fringe

                fringe.push(
                    (successor_node, path+[successor_path], new_cost), new_cost)
    return path
    # util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()  # Initialize a priority queue
    # push the start state to the fringe, with a starting cost of 0
    fringe.push((problem.getStartState(), [], 0), 0)
    visited = set()

    while fringe:
        # Get the current node, its current path and the priority
        node,  path, cost = fringe.pop()

        if problem.isGoalState(node):  # Check if the current node is the goal node
            return path  # Return node path if it is the goal node

        elif node not in visited:  # check if current node is not in the visited list
            # path.append(successor)
            visited.add(node)  # add current node to the visited list

        elif node in visited:
            # leaf node
            continue

        # Get a neighboring node of the current node
        for successor in problem.getSuccessors(node):
          # The states are saved as, for example, ('B', 'B->C', 0)
          # Get each of the successor components and store them in variables = (successor_node, successor_path, successor_cost)
            successor_node = successor[0]
            successor_path = successor[1]
            successor_cost = successor[2]
            # add the current node cost to the neighboring node cost to get the new_cost
            new_cost = cost + successor_cost
            # using the heuristic function, add it to the new cost to get the f score
            # f(n) = g(n) + h(n)
            f_score = new_cost + heuristic(successor_node, problem)

            # Check if the successor(neighbour) state is not in the the visited set
            if successor not in visited:
                # push it to the fringe, which accepts in form of (state, priority), state = ('A', [], 0) priority = 0

                fringe.push(
                    (successor_node, path+[successor_path], new_cost), f_score)
    return path


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
