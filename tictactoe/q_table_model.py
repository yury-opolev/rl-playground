import random
import pickle
from pathlib import Path

from game.game import Game
from game.agents.q_table_agent import QTableAgent
from game.agents.random_agent import RandomAgent

class QTableModel(object):
    def __init__(self, model_path, restore=False, save=False):
        self.model_path = model_path

        self.q_table = {}
        for element in Game.validBoards():
            if Game.isXWin(element):
                self.q_table[str(element)] = 1.0
            elif Game.isOWin(element):
                self.q_table[str(element)] = 0.0
            else:
                self.q_table[str(element)] = 0.5
            # if Game.isXWin(element):
            #     self.q_table[str(element) + '|X'] = 1.0
            #     self.q_table[str(element) + '|O'] = 1.0
            # elif Game.isOWin(element):
            #     self.q_table[str(element) + '|X'] = 0.0
            #     self.q_table[str(element) + '|O'] = 0.0
            # else:
            #     self.q_table[str(element) + '|X'] = 0.5
            #     self.q_table[str(element) + '|O'] = 0.5

        if restore:
            self.restore()

        self.save_on_train = save

    def save(self):
        print(f"Saving to {self.model_path}")
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def restore(self):
        existing_modelfile = Path(self.model_path)
        if not existing_modelfile.exists():
            return
        
        print(f"Loading from {self.model_path}")
        with open(self.model_path, 'rb') as f:
            self.q_table = pickle.load(f)

    def get_output(self, x):
        return self.q_table[x]

    def test(self, episodes=200):
        winners = { Game.EMPTYTOKEN: 0, Game.TOKEN_X: 0, Game.TOKEN_O: 0 }
        for episode in range(episodes):
            game = Game()
            player_agents = [QTableAgent('X', self.q_table), RandomAgent('O')]
            winner_token = game.play(player_agents, draw=False)
            if winner_token is None:
                winner_token = Game.EMPTYTOKEN
            winners[winner_token] = winners[winner_token] + 1

        print(f"Games played: {episodes}, draws: {winners[Game.EMPTYTOKEN]}, 'X' wins: {winners[Game.TOKEN_X]}, 'O' wins: {winners[Game.TOKEN_O]}.")

    def train_q_table(self):
        player_agents = [QTableAgent('X', self.q_table), QTableAgent('O', self.q_table)]

        validation_interval = 10000
        episodes = 1000000
        learning_rate = 0.01
        for episode in range(episodes):
            if episode != 0 and episode % validation_interval == 0:
                print(f"Testing after {episode} episodes")
                self.test(episodes=100)

            game = Game()

            player_num = random.randint(0, 1)
            player_agent = player_agents[player_num]
            game.current_player_token = game.player_tokens[player_num]

            # recorded_states = []
            game_step = 0
            while not game.is_finished():
                observed_state = game.extract_qstate()
                # recorded_states.append(observed_state)
                state_value = player_agent.q_table[observed_state]

                game.make_move(player_agent, greedy=False)

                player_num = (player_num + 1) % 2
                player_agent = player_agents[player_num]
                game.current_player_token = game.player_tokens[player_num]

                next_observed_state = game.extract_qstate()
                next_state_value = self.q_table[next_observed_state]

                adjustment = learning_rate * (next_state_value - state_value)
                self.q_table[observed_state] = self.q_table[observed_state] + adjustment
                game_step += 1

            # recorded_states.append(game.extract_qstate())

            # final_recorded_state = recorded_states[-1]
            # final_value = self.q_table[final_recorded_state]
            # discount_factor = 0.9
            # for index in reversed(range(0, len(recorded_states) - 2)):
            #     recorded_state = recorded_states[index]
            #     adjustment = discount_factor * learning_rate * (final_value - self.q_table[recorded_state])
            #     self.q_table[recorded_state] = self.q_table[recorded_state] + adjustment
            #     discount_factor *= discount_factor

        if self.save_on_train:
            self.save()
