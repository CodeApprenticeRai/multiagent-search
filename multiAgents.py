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


from util import manhattanDistance
from game import Directions
import random, util

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
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        pacmanCurrentPosition = currentGameState.getPacmanPosition()
        currentFoodList = currentGameState.getFood().asList()
        currentGhostStates = currentGameState.getGhostStates()
        # currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates ]

        if action == 'Stop':
            return -1

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        pacmanSuccessorPosition = successorGameState.getPacmanPosition()
        successorFoodList = successorGameState.getFood().asList()
        successorGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        currentDistanceToClosestGhost = float('inf')
        for ghost in currentGhostStates:
            currentDistanceToClosestGhost = min( currentDistanceToClosestGhost, util.manhattanDistance( pacmanCurrentPosition, ghost.getPosition()) )

        sucessiveDistanceToClosestGhost = float('inf')
        for ghost in successorGhostStates:
            sucessiveDistanceToClosestGhost = min( sucessiveDistanceToClosestGhost, util.manhattanDistance( pacmanSuccessorPosition, ghost.getPosition()) )

        # reflexively disqaulify states that kill pacman
        if ( ( currentDistanceToClosestGhost == 1 ) and ( sucessiveDistanceToClosestGhost < currentDistanceToClosestGhost ) ):
            return -1

        currentDistanceToClosestPellet = float('inf')
        for pelletPosition in currentFoodList:
            currentDistanceToClosestPellet = min( currentDistanceToClosestPellet, util.manhattanDistance( pacmanCurrentPosition, pelletPosition) )


        sucessiveDistanceToClosestPellet = float('inf')
        for pelletPosition in successorFoodList:
            sucessiveDistanceToClosestPellet = min( sucessiveDistanceToClosestPellet, util.manhattanDistance( pacmanSuccessorPosition, pelletPosition) )


        score = 1 / sucessiveDistanceToClosestPellet # / currentDistanceToClosestPellet # should never be inf as long as pellets exits ( and thus for games that end when there are no more pellets )

        #print score
        return score


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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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

        self.pacmanAgentIndex = 0

        intialActionSpace = gameState.getLegalActions( self.pacmanAgentIndex )
        initialSuccessorStates = [ gameState.generateSuccessor( self.pacmanAgentIndex, action ) for action in intialActionSpace ]

        indexOfBestAction = None
        bestScore = -1 * float('inf')
        for i in range( len(initialSuccessorStates) ):
             expectedScoreFromCurrentSuccessorState = self.minimize( initialSuccessorStates[i], depth=self.depth, ghostIndex=1 )
             if ( expectedScoreFromCurrentSuccessorState > bestScore ):
                 bestScore = expectedScoreFromCurrentSuccessorState
                 indexOfBestAction = i

        return  intialActionSpace[ indexOfBestAction ]


    def maximize(self, currentState, depth  ):
        if ( (depth == 0) or currentState.isLose() or currentState.isWin()):
            return self.evaluationFunction(currentState)

        succesorStates =  [ currentState.generateSuccessor( self.pacmanAgentIndex, action ) for action in currentState.getLegalActions( self.pacmanAgentIndex ) ]
        scores = [ self.minimize(succesorState, depth, 1) for succesorState in succesorStates ]

        return max(scores)

    def minimize( self, currentState, depth, ghostIndex ):
        if ( ( depth == 0) or currentState.isLose() or currentState.isWin()):
            return self.evaluationFunction(currentState)

        succesorStates = [ currentState.generateSuccessor(ghostIndex, action) for action in currentState.getLegalActions(ghostIndex) ]
        scores = []
        for succesorState in succesorStates:
            if ( ghostIndex == ( currentState.getNumAgents() - 1 ) ):
                scores.append(  self.maximize( succesorState, depth - 1 ) )
            else:
                scores.append( self.minimize( succesorState, depth, ghostIndex + 1) )

        return min(scores)



class AlphaBetaAgent(MultiAgentSearchAgent):
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

        self.pacmanAgentIndex = 0

        initialActionSpace = gameState.getLegalActions( self.pacmanAgentIndex )


        indexOfBestAction = None
        bestScore = -1 * float('inf')
        for i in range( len(initialActionSpace) ):
            successorState = gameState.generateSuccessor( self.pacmanAgentIndex, initialActionSpace[i] )
            expectedScoreFromCurrentSuccessorState = self.minimize( successorState, self.depth, 1, bestScore, float("inf") )
            if ( expectedScoreFromCurrentSuccessorState > bestScore ):
                bestScore = expectedScoreFromCurrentSuccessorState
                indexOfBestAction = i

        return initialActionSpace[ indexOfBestAction ]


    def maximize( self, currentState, depth, alpha, beta  ):
        if ( (depth == 0) or currentState.isLose() or currentState.isWin() ):
            return self.evaluationFunction(currentState)

        bestScore = -1 * float("inf")
        for action in currentState.getLegalActions(self.pacmanAgentIndex):
            succesorState = currentState.generateSuccessor( self.pacmanAgentIndex, action )
            bestScore = max( bestScore, self.minimize(succesorState, depth, 1, alpha, beta ) )

            if ( bestScore > beta ): # minimizer wont choose it
                return bestScore

            alpha = max( alpha, bestScore )
        return bestScore

    def minimize( self, currentState, depth, ghostIndex, alpha, beta  ):
        if ( ( depth == 0) or currentState.isLose() or currentState.isWin()):
            return self.evaluationFunction(currentState)

        worstScore = float("inf")
        for action in currentState.getLegalActions(ghostIndex):
            succesorState = currentState.generateSuccessor(ghostIndex, action)

            if ( ghostIndex == ( currentState.getNumAgents() - 1 ) ):
                worstScore = min( worstScore, self.maximize( succesorState, depth - 1, alpha, beta ) )
            else:
                worstScore = min( worstScore, self.minimize( succesorState, depth, ghostIndex + 1, alpha, beta) )

            if ( worstScore < alpha ):
                return worstScore

            beta = min( beta, worstScore )

        return worstScore


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
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

        self.pacmanAgentIndex = 0

        initialActionSpace = gameState.getLegalActions( self.pacmanAgentIndex )


        indexOfBestAction = None
        bestScore = -1 * float('inf')
        for i in range( len(initialActionSpace) ):
            successorState = gameState.generateSuccessor( self.pacmanAgentIndex, initialActionSpace[i] )
            expectedScoreFromCurrentSuccessorState = self.minimize( successorState, self.depth, 1 )
            if ( expectedScoreFromCurrentSuccessorState > bestScore ):
                bestScore = expectedScoreFromCurrentSuccessorState
                indexOfBestAction = i

        return initialActionSpace[ indexOfBestAction ]


    def maximize( self, currentState, depth ):
        if ( (depth == 0) or currentState.isLose() or currentState.isWin() ):
            return self.evaluationFunction(currentState)

        bestScore = -1 * float("inf")
        for action in currentState.getLegalActions(self.pacmanAgentIndex):
            succesorState = currentState.generateSuccessor( self.pacmanAgentIndex, action )
            bestScore = max( bestScore, self.minimize(succesorState, depth, 1 ) )

        return bestScore

    def minimize( self, currentState, depth, ghostIndex ):
        if ( ( depth == 0) or currentState.isLose() or currentState.isWin()):
            return self.evaluationFunction(currentState)

        succesorStates = [ currentState.generateSuccessor(ghostIndex, action) for action in currentState.getLegalActions(ghostIndex) ]
        scores = []
        for succesorState in succesorStates:
            if ( ghostIndex == ( currentState.getNumAgents() - 1 ) ):
                scores.append( self.maximize( succesorState, depth - 1 ) )
            else:
                scores.append( self.minimize( succesorState, depth, ghostIndex + 1 ) )

        return float( sum(scores) ) / len(scores)


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

