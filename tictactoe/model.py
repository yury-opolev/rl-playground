import random
import tensorflow as tf
import keras
import copy
from pathlib import Path
from keras import layers
from keras import models
from keras import initializers

from game.game import Game
from game.agents.ai_agent import AIAgent
from game.agents.random_agent import RandomAgent

class Model(object):
    def __init__(self, model_path, summary_path, checkpoint_dir, restore=False, save=False):
        self.model_path = model_path
        self.summary_path = summary_path
        self.checkpoint_dir = checkpoint_dir

        self.nn_model = models.Sequential([
            layers.Input(shape=(20,)),
            layers.Dense(20, activation='sigmoid',
                         kernel_initializer=initializers.RandomNormal(stddev=0.05),
                         bias_initializer=initializers.RandomNormal(stddev=0.05)),
            # result is a prediction of probability of winning: 1 - 'X' wins, 0 - 'O' wins, 0.5 - draw
            layers.Dense(1, activation='sigmoid',
                         kernel_initializer=initializers.RandomNormal(stddev=0.05),
                         bias_initializer=initializers.RandomNormal(stddev=0.05))
        ])

        self.learning_rate = 0.001
        self.optimizer = keras.optimizers.SGD(learning_rate=self.learning_rate)

        self.save = save
        if restore:
            self.restore_weights()

    def get_output(self, state):
        return self.nn_model(state)

    def update_weights(self, state, expected_value, discount_rate=1.0):
        with tf.GradientTape() as tape:
            predicted_value = self.get_output(state)
            loss = tf.abs(tf.multiply(tf.subtract(expected_value, predicted_value), discount_rate))
            gradients = tape.gradient(loss, self.nn_model.trainable_variables)
        self.optimizer.apply(gradients, self.nn_model.trainable_variables)

    def restore_weights(self):
        weights_filename = f"{self.model_path}tf-model.weights.h5"
        weights_filepath = Path(weights_filename)
        if weights_filepath.exists():
            print(f'Restoring weights: {weights_filename}')
            self.nn_model.load_weights(weights_filename)

    def save_weights(self):
        weights_filename = f"{self.model_path}tf-model.weights.h5"
        print(f'Saving weights: {weights_filename}')
        self.nn_model.save_weights(weights_filename)

    def test(self, episodes=100):
        winners = { Game.EMPTYTOKEN: 0, Game.TOKEN_X: 0, Game.TOKEN_O: 0 }
        for episode in range(episodes):
            game = Game()
            player_agents = [AIAgent('X', self), RandomAgent('O')]
            winner_token = game.play(player_agents, draw=False)
            if winner_token is None:
                winner_token = Game.EMPTYTOKEN
            winners[winner_token] = winners[winner_token] + 1

        print(f"Games played: {episodes}, draws: {winners[Game.EMPTYTOKEN]}, 'X' wins: {winners[Game.TOKEN_X]}, 'O' wins: {winners[Game.TOKEN_O]}.")

    def train(self):
        validation_interval = 1000
        episodes = 10000
        for episode in range(episodes):
            if episode % validation_interval == 0:
                print(f"Testing after {episode} episodes:")
                self.test(episodes=100)

            player_agents = [AIAgent('X', self), AIAgent('O', self)]
            game = Game()

            player_num = random.randint(0, 1)
            player_agent = player_agents[player_num]
            game.current_player_token = game.player_tokens[player_num]
            game.first_player_token = game.player_tokens[player_num]

            game_step = 0
            recorded_states = []
            while not game.is_finished():
                observed_state = game.extract_features()
                observed_state_description = game.extract_qstate()
                state_value = self.get_output(observed_state)
                recorded_states.append((copy.deepcopy(observed_state), observed_state_description, state_value))

                game.make_move(player_agent, greedy=False)

                player_num = (player_num + 1) % 2
                player_agent = player_agents[player_num]
                game.current_player_token = game.player_tokens[player_num]

                next_observed_state = game.extract_features()
                next_observed_state_description = game.extract_qstate()

                if (game.is_finished()):
                    next_state_value = tf.Variable([[0.5]], trainable=False)
                    if game.winner_token == Game.TOKEN_X:
                        next_state_value = tf.Variable([[1.0]], trainable=False)
                    elif game.winner_token == Game.TOKEN_O:
                        next_state_value = tf.Variable([[0.0]], trainable=False)
                else:
                    next_state_value = self.get_output(next_observed_state)

                self.update_weights(observed_state, next_state_value, discount_rate=1.0)

                game_step += 1

            recorded_states.append((copy.deepcopy(next_observed_state), next_observed_state_description, next_state_value))

            line_to_print = ""
            for state, description, value in recorded_states:
                value_single = value[0,0].numpy()
                line_to_print += f"{description}:{value_single:.8f};"
            print(f"BEFORE: {line_to_print}")

            discount_rate = 1.0
            recorded_states.reverse()
            for state, description, value in recorded_states:
                self.update_weights(state, next_state_value, discount_rate)
                discount_rate *= 0.9

            line_to_print = ""
            for state, description, value in recorded_states:
                value_single = self.get_output(state)[0,0].numpy()
                line_to_print += f"{description}:{value_single:.8f};"
            print(f"AFTER: {line_to_print}")

        if self.save:
            self.save_weights()

