from typing import List, Tuple, Any, Optional
import game_alt as game
import time

BoardType = game.BoardType

class ZeroModel:
    def __init__(self):
        pass
    def infer(self, state: BoardType) -> Tuple[List[float], float]:
        # return [0.1, 0.3, 0.2, 0.1], 2.325
        raise NotImplementedError

class ZeroMCTSNode:
    def __init__(self, parent=None, state=None, action=None):
        self.parent = parent
        self.state = state
        self.action = action
        self.children:List[List['ZeroMCTSNode']] = []*4
        self.visits = 0
        self.value = 0

    def expand(self, action: int, state: BoardType):
        child = ZeroMCTSNode(parent=self, state=state, action=action)
        self.children[action].append(child)


class ZeroPlayer:
    def __init__(self, model):
        self.model = model

    def get_best_action_puct(self, state: BoardType, actions: List[int]) -> int:
        # compute Q


    def play(self, state, time_limit: float = 0.5) -> int:
        """
        :param state: the current state of the game
        :param time_limit: max time to think
        :return: the type of action to take (up, down, left, right)
        """

        deadline = time.time() + time_limit
        while time.time() < deadline:

        pass

class ZeroTrainer:
    def __init__(self):
        pass