
from abc import ABC, abstractmethod

class TicTacToeAgent(ABC):

    @abstractmethod
    def get_action(self, actions, game, epsilon=0.0):
        pass
