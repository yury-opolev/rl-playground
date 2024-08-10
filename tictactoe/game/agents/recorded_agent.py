from game.agents.base_agent import TicTacToeAgent

class RecordedAgent(TicTacToeAgent):
    def __init__(self, player_token, game_history):
        self.player_token = player_token
        self.game_history = game_history
        self.name = 'Recorded'

    def get_state_key(self, state):
        result = ''
        for item_1,item_2 in zip(state[0::2], state[1::2]):
            if item_1 == 1:
                result += 'X'
            elif item_2 == 1:
                result += 'O'
            else:
                result += '-'
        return result

    def get_action(self, actions, game, epsilon=0.0, verbose=False):
        state_features = game.extract_features()
        state_key = self.get_state_key(state_features)
        found_items = [item for item in self.game_history if item[0] == state_key]

        result_state, result_action, result_next_state = found_items[0]
        if verbose:
            print(f"[{self.name}] > Taking recorded action: {result_action} for state {result_state}")

        return (result_action, 0.0)
