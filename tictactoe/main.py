from absl import app
from absl import flags
import os
import tensorflow as tf

print(">>> Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from nn_model import NNModel
from qtab_model import QTabModel
from game.env import Game
from game.agents.human_agent import HumanAgent
from game.agents.random_agent import RandomAgent
from game.agents.ai_agent import AIAgent

FLAGS = flags.FLAGS

flags.DEFINE_string('mode', 'play', 'List of modes: play, test, train, train_q, test_q.')
flags.DEFINE_boolean('restore', False, 'If true, restore the model from latest checkpoint.')
flags.DEFINE_boolean('save', False, 'If true, save the trained model (works only for q model and train mode).')

model_path = os.environ.get('MODEL_PATH', 'models/')

if not os.path.exists(model_path):
    os.makedirs(model_path)

def main(argv):
    if FLAGS.mode == 'play':
        ai_model = NNModel()
        if FLAGS.restore:
            ai_model.restore_weights('models/current.weights.h5')

        game = Game()
        player_agents = [AIAgent('X', ai_model), HumanAgent('O')]

        game.current_player_token = game.starting_random_player()
        if game.current_player_token == Game.TOKEN_X:
            current_player_agent = player_agents[0]
        else:
            current_player_agent = player_agents[1]

        while not game.is_finished():
            game.clear_screen()
            game.draw()

            actions = game.get_possible_actions()
            action_value = current_player_agent.get_action(actions, game)
            action, value = action_value
            game.take_action(action, game.current_player_token)

            game.change_player()
            if game.current_player_token == Game.TOKEN_X:
                current_player_agent = player_agents[0]
            else:
                current_player_agent = player_agents[1]

        game.clear_screen()
        game.draw()

        if game.winner_token == None:
            print("DRAW.")
        elif game.winner_token == Game.TOKEN_X:
            print("X wins!")
        else:
            print("O wins!")

    if FLAGS.mode == 'train':
        ai_model = NNModel()
        if FLAGS.restore:
            ai_model.restore_weights('models/current.weights.h5')

        for batch in range(1000):
            print(f"training batch: {batch}")
            ai_model.train(episodes=1000)
            if FLAGS.save:
                ai_model.save_weights('models/current.weights.h5')

    if FLAGS.mode == 'test':
        ai_model = NNModel()
        if FLAGS.restore:
            ai_model.restore_weights('models/current.weights.h5')

        ai_model.test()

    if FLAGS.mode == 'q_train':
        qtab_model = QTabModel()
        if FLAGS.restore:
            qtab_model.restore_weights('models/current.weights.qtab')

        for batch in range(100):
            print(f"training batch: {batch}")
            qtab_model.train(episodes=10000, epsilon=0.8)
            if FLAGS.save:
                qtab_model.save_weights('models/current.weights.qtab')

    if FLAGS.mode == 'q_test':
        qtab_model = QTabModel()
        if FLAGS.restore:
            qtab_model.restore_weights('models/current.weights.qtab')

        qtab_model.test()


if __name__ == '__main__':
    app.run(main)
