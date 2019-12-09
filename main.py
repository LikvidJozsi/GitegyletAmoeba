import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.AmoebaAgent import RandomAgent
from AmoebaPlayGround.AmoebaTrainer import AmoebaTrainer
from AmoebaPlayGround.HandWrittenAgent import HandWrittenAgent
from AmoebaPlayGround.NeuralAgent import NeuralAgent, ResNetLike
# graphicalView = GraphicalView((10, 10))
# game_board = np.array([[0,0,0,0],[0,1,1,1],[-1,-1,-1,0],[0,1,-1,0]])
# graphicalView.display_game_state(game_board)
# demo
# consoleAgent = ConsoleAgent()
# view = ConsoleView()
from AmoebaPlayGround.RewardCalculator import PolicyGradients
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--log-file-name', action="store",
                    dest="log_file_name", default="")
args = parser.parse_args()

Amoeba.map_size = (8, 8)
Amoeba.win_sequence_length = 5

learning_agent = NeuralAgent(model_type=ResNetLike())
random_agent = RandomAgent()
hand_written_agent = HandWrittenAgent()
trainer = AmoebaTrainer(learning_agent, teaching_agents=[random_agent, hand_written_agent], self_play=True,
                        reward_calculator=PolicyGradients(teach_with_losses=False))

trainer.train(batch_size=500, num_episodes=5, log_file_name=args.log_file_name)
