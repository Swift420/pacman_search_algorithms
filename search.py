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

    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    # # print("hello", problem)

    # visited.add(problem.getStartState())

    # for neighbor in problem.getSuccessors(problem.getStartState()):
    #     print("heyyya", neighbor[0])
    # stack = ["A", "B1", "C", "D", "E2", "F", "G"].reverse

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

        # print("current node: ", node)
        # print("current successor: ", successor)

        # Check if the current node is the goal node
        if problem.isGoalState(currentNode):
            # print("is Goal state")
            # path.append(successor)

            return path

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
            # print(neighbour)

            path1 = path + [successPath]
            # add the neighbour node to the stack with its current path
            stack.append((neighbourNode,  path1))
    return path
    # util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()

    visited = []
    path = []
    # use a fifo in bfs, so a queue data structure is used.
    queue.push((problem.getStartState(), path))

    while queue:
        # print("stack: ", stack)
        # print("visited: ", visited)
        # print("path: ", path)
        node, path = queue.pop()
        # print("current node: ", node)
        # print("current successor: ", successor)
        if problem.isGoalState(node):
            # print("is Goal state")
            # path.append(successor)

            return path

        elif node not in visited:
            # path.append(successor)
            visited.append(node)

        elif node in visited:
            # leaf node
            continue

        for neighbourNode, successPath, cost in problem.getSuccessors(node):
            # print(neighbour)
            path1 = path + [successPath]

            queue.push((neighbourNode, path1))
    return path
    # util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    frontier = [(0, problem.getStartState(), [])]
    visited = set()

    while frontier:
        cost,  state, path = heapq.heappop(frontier)
        node = state[0]
        # print(node, cost, " node cost", state)
        # print(path)

        if problem.isGoalState(node):
            return path

        elif node not in visited:
            # path.append(successor)
            visited.add(node)

        elif node in visited:
            # leaf node
            continue

        for successor in problem.getSuccessors(node):
            successor_node = successor[0]
            successor_path = successor[1]
            successor_cost = successor[2]

            new_cost = cost + successor_cost
            if successor not in visited:

                heapq.heappush(
                    frontier, (new_cost, successor, path+[successor_path]))
    return []
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
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
