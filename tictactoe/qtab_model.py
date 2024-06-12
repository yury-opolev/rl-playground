import random
import pickle
import numpy as np

from pathlib import Path

from game.env import Game
from game.agents.ai_agent import AIAgent
from game.agents.random_agent import RandomAgent

class QTabModel(object):
    def __init__(self):
        self.q_table = {}

        self.learning_rate = 0.001
        self.gamma = 0.9

    def get_action_index(self, action):
        x, y = action
        action_index = x * 3 + y
        return action_index
    
    def get_state_action_value(self, state, action):
        state_key = self.get_state_key(state)
        if not state_key in self.q_table:
            self.q_table[state_key] = np.zeros(9)
        action_index = self.get_action_index(action)
        return self.q_table[state_key][action_index]
    
    def get_state_key(self, state):
        return ''.join([f"{int(item)}" for item in state])

    def get_output(self, state, action):
        return self.get_state_action_value(state, action)

    def test(self, episodes=100):
        winners = { Game.EMPTYTOKEN: 0, Game.TOKEN_X: 0, Game.TOKEN_O: 0 }
        for episode in range(episodes):
            game = Game()
            player_agents = [AIAgent('X', self), RandomAgent('O')]
            
            game.current_player_token = game.starting_random_player()
            current_player_agent = self.get_player_agent(game, player_agents)

            while not game.is_finished():
                actions = game.get_possible_actions()
                action, value = current_player_agent.get_action(actions, game)
                game.take_action(action, game.current_player_token)

                game.change_player()
                current_player_agent = self.get_player_agent(game, player_agents)

            winner_token = game.winner_token
            if winner_token is None:
                winner_token = Game.EMPTYTOKEN
            winners[winner_token] = winners[winner_token] + 1

        print(f"Games played: {episodes}, draws: {winners[Game.EMPTYTOKEN]}, 'X' wins: {winners[Game.TOKEN_X]}, 'O' wins: {winners[Game.TOKEN_O]}.")

    def train(self, episodes=10000, epsilon=0.5):
        validation_interval = 1000
        for episode in range(episodes):
            if episode % validation_interval == 0:
                print(f"Testing after {episode} episodes:")
                self.test(episodes=100)
                print()

            player_agents = [AIAgent('X', self), AIAgent('O', self)]
            game = Game()

            game.current_player_token = game.starting_random_player()
            current_player_agent = self.get_player_agent(game, player_agents)

            while True:
                # get state S 
                observed_state = game.extract_features()

                # get action A (and Q(S, A)) 
                actions = game.get_possible_actions()
                (action, action_value) = current_player_agent.get_action(actions, game, epsilon)

                # get R
                reward, is_done = game.step(action, game.grid, game.current_player_token)

                game.change_player()
                current_player_agent = self.get_player_agent(game, player_agents)

                # get S'
                next_observed_state = game.extract_features()

                # get action A' (and Q(S',A'))
                actions_next = game.get_possible_actions()
                (action_next, action_next_value) = current_player_agent.get_action(actions_next, game, epsilon)

                # Q(S, A) <- Q(S, A) + alpha * ((R  + gamma * Q(S',A')) - Q(S, A))
                self.update_weights(observed_state, action, (reward + self.gamma * action_next_value) - action_value)

                if is_done:
                    break

        print(f"Final testing:")
        self.test(episodes=100)
        print()

    def update_weights(self, state, action, td_error, discount_rate=1.0):
        state_key = self.get_state_key(state)
        action_index = self.get_action_index(action)
        state_value = self.get_state_action_value(state, action)
        self.q_table[state_key][action_index] = state_value + self.learning_rate * td_error

    def restore_weights(self, path):
        existing_modelfile = Path(path)
        if not existing_modelfile.exists():
            return
        
        print(f"Loading from {path}")
        print()
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)

    def save_weights(self, path):
        print(f"Saving to {path}")
        print()
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def get_player_agent(self, game, player_agents):
        if game.current_player_token == Game.TOKEN_X:
            current_player_agent = player_agents[0]
        else:
            current_player_agent = player_agents[1]
        return current_player_agent
