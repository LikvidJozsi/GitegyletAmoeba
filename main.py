
import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.AmoebaAgent import RandomAgent
from AmoebaPlayGround.AmoebaTrainer import AmoebaTrainer
from AmoebaPlayGround.HandWrittenAgent import HandWrittenAgent
from AmoebaPlayGround.NeuralAgent import NeuralAgent, ShallowNetwork

from AmoebaPlayGround.RewardCalculator import PolicyGradients, PolicyGradientsWithNegativeTeaching

from AmoebaPlayGround.Logger import Logger, FileLogger, AmoebaTrainingFileLogger
from AmoebaPlayGround.Input import get_model_filename

file_name = get_model_filename()

Amoeba.map_size = (8, 8)
Amoeba.win_sequence_length = 5

learning_agent = NeuralAgent(load_latest_model=True)
learning_agent.print_model_saummary()
random_agent = RandomAgent()
hand_written_agent = HandWrittenAgent()

#trainer = AmoebaTrainer(learning_agent, teaching_agents=[random_agent, hand_written_agent], self_play=False,
#                        reward_calculator=PolicyGradientsWithNegativeTeaching())

trainer = AmoebaTrainer(learning_agent, teaching_agents=[hand_written_agent], self_play=False,
                        reward_calculator=PolicyGradientsWithNegativeTeaching())

trainer.train(batch_size=1000, num_episodes=50, model_save_file=file_name, logger=AmoebaTrainingFileLogger(file_name))
