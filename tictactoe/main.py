from absl import app
from absl import flags
import os
import tensorflow as tf

from model import Model
from q_table_model import QTableModel

from game.game import Game
from game.agents.human_agent import HumanAgent
from game.agents.random_agent import RandomAgent
from game.agents.ai_agent import AIAgent
from game.agents.q_table_agent import QTableAgent

FLAGS = flags.FLAGS

flags.DEFINE_string('mode', 'play', 'List of modes: play, test, train, train_q, test_q.')
flags.DEFINE_boolean('restore', False, 'If true, restore the model from latest checkpoint.')
flags.DEFINE_boolean('save', False, 'If true, save the trained model (works only for q model and train mode).')

model_path = os.environ.get('MODEL_PATH', 'models/')
summary_path = os.environ.get('SUMMARY_PATH', 'summaries/')
checkpoint_path = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')

if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

if not os.path.exists(summary_path):
    os.makedirs(summary_path)

def main(argv):
    ai_model = Model(model_path, summary_path, checkpoint_path, restore=FLAGS.restore)
    if FLAGS.mode == 'test':
        # TODO: add testing
        pass

    if FLAGS.mode == 'play':
        game = Game()
        #player_agents = [HumanAgent('X'), AIAgent('O', ai_model)]
        q_model = QTableModel(model_path + 'q_model_18052024', restore=True)
        player_agents = [HumanAgent('X'), QTableAgent('O', q_model.q_table)]
        game.play(player_agents, draw=True)
        pass

    if FLAGS.mode == 'train':
        # TODO: add training
        pass

    if FLAGS.mode == 'train_q':
        q_model = QTableModel(model_path + 'q_model', restore=FLAGS.restore, save=FLAGS.save)
        q_model.train_q_table()

    if FLAGS.mode == 'test_q':
        q_model = QTableModel(model_path + 'q_model', restore=FLAGS.restore, save=FLAGS.save)
        q_model.test()

if __name__ == '__main__':
    app.run(main)
