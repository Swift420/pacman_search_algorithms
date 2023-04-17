# multiAgents.py
# --------------
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

from cmath import inf

from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]
        # print(newPos)
        # print(newFood)
        # print(newScaredTimes)
        # print(newGhostStates)

        "*** YOUR CODE HERE ***"
   # We created a lambda function that takes one argument pos and calls manhattanDistance with newPos and pos as arguments
        def distanceToPacman(pos): return manhattanDistance(newPos, pos)

        def Ghost(ghost):
            # calculate distance between the ghost and pacman
            distance = distanceToPacman(ghost.getPosition())

            if ghost.scaredTimer > distance:
                return float('inf')
            return -float('inf') if distance < +1 else 0

        # Created a list of ghost scores by applying the Ghost function to each ghost state
        ghost_scores = [Ghost(state) for state in newGhostStates]
        ghostScore = min(ghost_scores)
        # Created an object that yields the distance to each food as needed
        distances_to_food = (distanceToPacman(food)
                             for food in newFood.asList())
        # finding the minimum distance to each food
        distanceToClosestFood = min(distances_to_food, default=float('inf'))
        # Calculate the distance to the closest food and create a feature based on it
        closestFoodFeature = 1.0 / (1.0 + distanceToClosestFood)

# Return the sum of the successor game state score, ghost score, and closest food feature
        return successorGameState.getScore() + ghostScore + closestFoodFeature


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        # Get the legal actions for Pacman (agent 0) in the current game state
        actions = gameState.getLegalActions(0)

        # return action with highest gamestate score
        return max(actions, key=lambda x: self.minimax_search(gameState.generateSuccessor(0, x), 1))

    def minimax_search(self, gameState, Direction):
        # Get the number of agents in the game and the index of the current agent
        numAgents = gameState.getNumAgents()
        agentIndex = Direction % numAgents

        # Calculate the depth of the current search
        depth = Direction // numAgents

        # Check if the game is over or if the search has reached the maximum depth
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            # Return the evaluation of the current state
            return self.evaluationFunction(gameState)

        # Get the legal actions for the current agent
        actions = gameState.getLegalActions(agentIndex)

        #  use a recursive funtion to evaluate each possible successor and store scores
        evaluation = [self.minimax_search(gameState.generateSuccessor(
            agentIndex, action), Direction + 1) for action in actions]

        # Return min or max scores for the current agent
        if agentIndex > 0:
            return min(evaluation)
        return max(evaluation)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        # Create a stack with initial state, agent, and depth
        stack = [(gameState, self.index, 0)]
        while stack:
            # Pop from the stack
            state, agent, depth = stack.pop()

            # Check if agent has no legal actions or depth has reached the limit
            if not state.getLegalActions(agent) or depth == self.depth:
                return self.evaluationFunction(state), None

            # If agent is the last one, increment depth and reset agent to the first one
            if agent == state.getNumAgents() - 1:
                depth += 1
                nextAgent = self.index
            else:
                nextAgent = agent + 1

            # Initialize a list to store the values of legal actions
            values = []

            # Loop through the legal actions of the current agent
            for action in state.getLegalActions(agent):
                # Push the next state, next agent, and depth onto the stack
                stack.append((state.generateSuccessor(
                    agent, action), nextAgent, depth))

                # If agent is the maximizing agent, find the best value
                if agent == self.index:
                    value = self.expectiMax(state.generateSuccessor(
                        agent, action), nextAgent, depth)
                    values.append(value)

                # If agent is not the maximizing agent, get the expected value
                else:
                    value = self.expectiMax(state.generateSuccessor(
                        agent, action), nextAgent, depth)
                    values.append(value)

            # If agent is the maximizing agent, return the best action
            if agent == self.index:
                bestValue = max(values)

                bestActions = []
                # Loop through the values and the legal actions
                for i in range(len(values)):
                    if values[i] >= bestValue:
                        # If current value is greater than or equal to the best value, add the corresponding action to the list
                        bestActions.append(state.getLegalActions(agent)[i])
                # Select the first action from the list of best actions
                # There may or may not be multiple best actions, so to not get an error, Select the 1st one.
                bestAction = bestActions[0]
                return bestAction

            # If agent is not the maximizing agent, return the expected value
            else:
                return sum(values) / len(values)

    # Define a separate function for expectimax recursion
    def expectiMax(self, state, agent, depth):
        # If agent has no legal actions or depth has reached the limit, return evaluation value
        if not state.getLegalActions(agent) or depth == self.depth:
            return self.evaluationFunction(state)

        # If agent is the last one, increment depth and reset agent to the first one
        if agent == state.getNumAgents() - 1:
            depth += 1
            nextAgent = self.index
        else:
            nextAgent = agent + 1

        # Initialize a list to store the values of legal actions
        values = []

        # Loop through the legal actions of the current agent
        for action in state.getLegalActions(agent):
            # Push the next state, next agent, and depth onto the stack
            value = self.expectiMax(state.generateSuccessor(
                agent, action), nextAgent, depth)
            values.append(value)

        # If agent is the maximizing agent, return the best value
        if agent == self.index:
            return max(values)

        # If agent is not the maximizing agent, return the expected value
        else:
            return sum(values) / len(values)


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
