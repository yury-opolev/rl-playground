import random
import tensorflow as tf
import keras
import numpy as np
import pandas as pd

from pathlib import Path
from keras import layers
from keras import models
from keras import initializers

from game.env import Game
from game.agents.ai_agent import AIAgent
from game.agents.random_agent import RandomAgent
from replay_memory import Transition, ReplayMemory

class NNModel(object):
    def __init__(self):
        self.create_model()
        self.learning_rate = 0.001
        self.gamma = 0.9
        self.lamda = 0.7

        self.optimizer = keras.optimizers.SGD(learning_rate=self.learning_rate)
        self.use_eligibility_traces = True

    def create_model(self):
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
            layers.Dense(9, activation=keras.activations.linear,
                         kernel_initializer=initializers.RandomNormal(stddev=0.05),
                         bias_initializer=initializers.RandomNormal(stddev=0.05))
        ])

    def init_eligiblity_trace(self):
        if not self.use_eligibility_traces:
            return
        self.eligibility_traces = [tf.Variable(tf.zeros(weights.shape), trainable=False) for weights in self.nn_model.trainable_weights]

    def get_actions_output(self, state_features):
        input_state = tf.convert_to_tensor([state_features])
        return self.get_output(input_state)[0].numpy()

    @tf.function
    def get_output(self, input_state):
        return self.nn_model(input_state)

    def get_action_index(self, action):
        x, y = action
        action_index = x * 3 + y
        return action_index

    def test(self, episodes=1000):
        ai_wins = { 'WON': 0, 'LOST': 0, 'DRAW': 0 }
        for episode in range(episodes):
            game = Game()

            if random.choice([0, 1]) == 0:
                player_agents = [AIAgent('X', self), RandomAgent('O')]
                ai_agent_token = Game.TOKEN_X
                random_agent_token = Game.TOKEN_O
            else:
                player_agents = [RandomAgent('X'), AIAgent('O', self)]
                ai_agent_token = Game.TOKEN_O
                random_agent_token = Game.TOKEN_X

            current_player_agent = self.get_player_agent(game, player_agents)

            while not game.is_finished():
                actions = game.get_possible_actions()
                action_value = current_player_agent.get_action(actions, game)
                action, value = action_value
                game.take_action(action, game.current_player_token)

                game.change_player()
                current_player_agent = self.get_player_agent(game, player_agents)

            winner_token = game.winner_token
            if winner_token is None or winner_token == Game.EMPTYTOKEN:
                ai_wins['DRAW'] = ai_wins['DRAW'] + 1
            else:
                if ai_agent_token == winner_token:
                    ai_wins['WON'] = ai_wins['WON'] + 1
                else:
                    ai_wins['LOST'] = ai_wins['LOST'] + 1

        print(f"AI draws: {ai_wins['DRAW']}, wins: {ai_wins['WON']}, losses: {ai_wins['LOST']}.")
        return (ai_wins['DRAW'], ai_wins['WON'], ai_wins['LOST'])

    def train(self, episodes=10000, epsilon=0.5, validate=False):
        global_steps = 0
        validation_interval = 1000
        test_episodes_count = 1000
        for episode in range(episodes):
            if validate and (episode > 0) and (episode % validation_interval == 0):
                print(f"Testing after {episode} episodes ({test_episodes_count} test episodes):")
                self.test(episodes=test_episodes_count)
                print()

            player_agents = [AIAgent('X', self), AIAgent('O', self)]
            game = Game()

            current_player_agent = self.get_player_agent(game, player_agents)

            self.init_eligiblity_trace()

            is_done = False
            while not is_done:
                global_steps += 1

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
                if is_done:
                    actions_next = []
                    best_next_state_action_value = 0.0
                else:
                    actions_next = game.get_possible_actions()
                    (best_next_action, best_next_state_action_value) = current_player_agent.get_action(actions_next, game, 0.0)

                # Q(S, A) <- Q(S, A) + alpha * ((R  + gamma * Q(S',A')) - Q(S, A))
                self.update_weights(observed_state, self.get_action_index(action), reward, best_next_state_action_value)

        print(f"Final testing ({test_episodes_count} test episodes):")
        (draw, won_ai, lost_ai) = self.test(episodes=test_episodes_count)
        print()

        return (draw, won_ai, lost_ai)

    def get_action_index(self, action):
        x, y = action
        action_index = x * 3 + y
        return action_index

    def update_weights(self, state, action_index, reward, next_best_state_action_value):
        input_state = tf.stop_gradient(tf.convert_to_tensor([state]))
        with tf.GradientTape() as tape:
            predicted_values = self.get_output(input_state)
            indices = tf.constant([[ 0, action_index ]])
            updates = tf.constant([ reward + self.gamma * next_best_state_action_value ], dtype=tf.float32)
            expected_values = tf.stop_gradient(tf.tensor_scatter_nd_update(predicted_values, indices, updates))
            expected_action_value = expected_values[0][action_index]
            predicted_action_value = predicted_values[0][action_index]
            loss = tf.abs(expected_action_value - predicted_action_value)

        if self.use_eligibility_traces:
            gradients = tape.gradient(predicted_action_value, self.nn_model.trainable_weights)
            for i, gradient in enumerate(gradients):
                self.eligibility_traces[i].assign(self.lamda * self.eligibility_traces[i] + gradient)
                weight = self.nn_model.trainable_weights[i]
                weight.assign_add(self.learning_rate * tf.reshape(expected_action_value - predicted_action_value, shape=(1,)) * self.eligibility_traces[i])
        else:
            gradients = tape.gradient(loss, self.nn_model.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.nn_model.trainable_weights))

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
