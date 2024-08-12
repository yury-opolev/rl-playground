import random
import tensorflow as tf
import keras
import numpy as np

from tqdm import tqdm

from pathlib import Path
from keras import layers
from keras import models
from keras import initializers

from game.env import Game
from game.agents.ai_agent_stateval import AIAgentStateVal
from game.agents.random_agent import RandomAgent
from game.agents.recorded_agent import RecordedAgent

class NNModelStateVal(object):
    def __init__(self):
        # self.nn_model = models.Sequential([
        #     layers.Input(shape=(18,)),
        #     layers.Dense(18, activation=keras.activations.leaky_relu,
        #                  kernel_initializer=initializers.RandomNormal(stddev=0.05),
        #                  bias_initializer=initializers.RandomNormal(stddev=0.05)),
        #     layers.Dense(18, activation=keras.activations.leaky_relu,
        #                  kernel_initializer=initializers.RandomNormal(stddev=0.05),
        #                  bias_initializer=initializers.RandomNormal(stddev=0.05)),
        #     layers.Dense(1, activation=keras.activations.linear,
        #                  kernel_initializer=initializers.RandomNormal(stddev=0.05),
        #                  bias_initializer=initializers.RandomNormal(stddev=0.05))
        # ])

        self.nn_model = models.Sequential([
            layers.Input(shape=(18,)),
            layers.Dense(36, activation=keras.activations.leaky_relu,
                         kernel_initializer=initializers.RandomNormal(stddev=0.05),
                         bias_initializer=initializers.RandomNormal(stddev=0.05)),
            layers.Dense(36, activation=keras.activations.leaky_relu,
                         kernel_initializer=initializers.RandomNormal(stddev=0.05),
                         bias_initializer=initializers.RandomNormal(stddev=0.05)),
            layers.Dense(18, activation=keras.activations.leaky_relu,
                         kernel_initializer=initializers.RandomNormal(stddev=0.05),
                         bias_initializer=initializers.RandomNormal(stddev=0.05)),
            layers.Dense(1, activation=keras.activations.linear,
                         kernel_initializer=initializers.RandomNormal(stddev=0.05),
                         bias_initializer=initializers.RandomNormal(stddev=0.05))
        ])

        self.learning_rate = 0.001
        self.gamma = 0.9
        self.lamda = 0.7

        self.optimizer = keras.optimizers.SGD(learning_rate=self.learning_rate)

        self.monitored_states = []

    def init_eligiblity_trace(self):
        self.eligibility_traces = [tf.Variable(tf.zeros(weights.shape), trainable=False) for weights in self.nn_model.trainable_weights]

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

    def get_state_value(self, state):
        input_state = tf.convert_to_tensor([state])
        return self.get_output(input_state)[0].numpy()

    @tf.function(reduce_retracing=True)
    def get_output(self, input_state):
        return self.nn_model(input_state)

    def test(self, episodes=1000, print_failed_games=False):
        ai_wins = { 'WON': 0, 'LOST': 0, 'DRAW': 0 }
        for episode in range(episodes):
            game = Game()

            random_agent_is_second = random.choice([0, 1]) == 0
            if random_agent_is_second:
                player_agents = [AIAgentStateVal('X', self), RandomAgent('O')]
                ai_agent_token = Game.TOKEN_X
                random_agent_token = Game.TOKEN_O
            else:
                player_agents = [RandomAgent('X'), AIAgentStateVal('O', self)]
                ai_agent_token = Game.TOKEN_O
                random_agent_token = Game.TOKEN_X
            
            game.starting_player_token = Game.TOKEN_X
            game.current_player_token = game.starting_player_token

            current_player_agent = self.get_player_agent(game, player_agents)

            game_history = []
            while not game.is_finished():
                observed_state = game.extract_features()
                actions = game.get_possible_actions()
                action, value = current_player_agent.get_action(actions, game)
                game.take_action(action, game.current_player_token)

                game.change_player()
                current_player_agent = self.get_player_agent(game, player_agents)

                next_observed_state = game.extract_features()
                game_history.append((self.get_state_key(observed_state), action, self.get_state_key(next_observed_state)))

            winner_token = game.winner_token
            if winner_token is None or winner_token == Game.EMPTYTOKEN:
                ai_wins['DRAW'] = ai_wins['DRAW'] + 1
            else:
                if ai_agent_token == winner_token:
                    ai_wins['WON'] = ai_wins['WON'] + 1
                else:
                    ai_wins['LOST'] = ai_wins['LOST'] + 1

            if print_failed_games and winner_token == random_agent_token:
                print(Game.get_history_string(game_history))
                # "replay" game with verbose output
                game = Game()
                game.starting_player_token = Game.TOKEN_X
                game.current_player_token = game.starting_player_token
                if random_agent_is_second:
                    player_agents = [AIAgentStateVal('X', self), RecordedAgent('O', game_history)]
                else:
                    player_agents = [RecordedAgent('X', game_history), AIAgentStateVal('O', self)]
                current_player_agent = self.get_player_agent(game, player_agents)
                while not game.is_finished():
                    observed_state = game.extract_features()
                    actions = game.get_possible_actions()
                    action, value = current_player_agent.get_action(actions, game, verbose=True)
                    game.take_action(action, game.current_player_token)
                    game.change_player()
                    current_player_agent = self.get_player_agent(game, player_agents)

        print(f"AI draws: {ai_wins['DRAW']}, wins: {ai_wins['WON']}, losses: {ai_wins['LOST']}.")
        return (ai_wins['DRAW'], ai_wins['WON'], ai_wins['LOST'])

    def train(self, episodes=10000, epsilon=0.5, validate=False):
        validation_interval = 500
        test_episodes_count = 500
        for episode in range(episodes):
            if validate and (episode % validation_interval == 0):
                print(f"Testing after {episode} episodes, ({test_episodes_count} test episodes):")
                self.test(episodes=test_episodes_count)
                print()

            player_agents = [AIAgentStateVal('X', self), AIAgentStateVal('O', self)]
            game = Game()
            self.init_eligiblity_trace()

            current_player_agent = self.get_player_agent(game, player_agents)

            is_done = False
            while not is_done:
                # get state S 
                observed_state = game.extract_features()

                # get action A (and Q(S, A)) 
                actions = game.get_possible_actions()
                (action, action_value) = current_player_agent.get_action(actions, game, epsilon)
                (best_action, best_action_value) = current_player_agent.get_action(actions, game, epsilon=0.0)

                # get R
                reward, is_done = game.step(action, game.grid, game.current_player_token)

                game.change_player()
                current_player_agent = self.get_player_agent(game, player_agents)

                # get S'
                next_observed_state = game.extract_features()

                # get action A' (and Q(S',A'))
                if is_done:
                    next_state_value = 0.0
                else:
                    next_state_value = best_action_value
                    #next_state_value = self.get_state_value(next_observed_state)

                # V(S) <- V(S) + alpha * ((R  + gamma * V(S')) - V(S))
                self.update_weights(observed_state, reward, next_state_value, next_observed_state)

            # update terminal state value
            # V(S) <- V(S) + alpha * ((R  + gamma * V(S')) - V(S))
            self.update_weights(next_observed_state, reward, next_state_value, next_observed_state)

        print(f"Final testing ({test_episodes_count} test episodes):")
        (draw, win_x, win_o) = self.test(episodes=test_episodes_count)
        print()

        return (draw, win_x, win_o)

    def update_weights(self, state, reward, next_state_value, next_state):
        input_state = tf.stop_gradient(tf.convert_to_tensor([state]))
        expected_value = tf.stop_gradient(tf.convert_to_tensor([reward + self.gamma * next_state_value]))
        self.train_step(input_state, expected_value)

        state_key = self.get_state_key(state)
        if state_key in self.monitored_states:
            print(f"CHANGING '{state_key}' value ? -> {self.nn_model(input_state)[0].numpy()}, next state {self.get_state_key(next_state)} value {next_state_value}, reward {reward}.")

    @tf.function(reduce_retracing=True)
    def train_step(self, input_state, expected_value):
        # V(S) <- V(S) + alpha * ((R  + gamma * V(S')) - V(S))
        with tf.GradientTape(persistent=True) as tape:
            predicted_value = self.nn_model(input_state)
            # loss = tf.reduce_mean(tf.square(expected_value - predicted_value))

        # gradients = tape.gradient(loss, self.nn_model.trainable_weights)
        gradients = tape.gradient(predicted_value, self.nn_model.trainable_weights)
        for i, gradient in enumerate(gradients):
            self.eligibility_traces[i].assign(self.lamda * self.eligibility_traces[i] + gradient)
            weight = self.nn_model.trainable_weights[i]
            weight.assign_add(self.learning_rate * tf.reshape(expected_value - predicted_value, shape=(1,)) * self.eligibility_traces[i])

        # self.optimizer.apply_gradients(zip(gradients, self.nn_model.trainable_weights))

    def restore_weights(self, path):
        weights_filepath = Path(path)
        if weights_filepath.exists():
            print(f'Restoring weights: {path}')
            self.nn_model.load_weights(path)

    def save_weights(self, path):
        print(f'Saving weights: {path}')
        self.nn_model.save_weights(path)

    def get_player_agent(self, game, player_agents):
        if game.current_player_token == Game.TOKEN_X:
            current_player_agent = player_agents[0]
        else:
            current_player_agent = player_agents[1]
        return current_player_agent
