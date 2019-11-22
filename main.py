import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.AmoebaAgent import RandomAgent
from AmoebaPlayGround.AmoebaTrainer import AmoebaTrainer
from AmoebaPlayGround.NeuralAgent import NeuralNetwork
# graphicalView = GraphicalView((10, 10))
# game_board = np.array([[0,0,0,0],[0,1,1,1],[-1,-1,-1,0],[0,1,-1,0]])
# graphicalView.display_game_state(game_board)
# demo
# consoleAgent = ConsoleAgent()
# view = ConsoleView()
from AmoebaPlayGround.RewardCalculator import PolicyGradients

map_size = (8, 8)
Amoeba.win_sequence_length = 5

learning_agent = NeuralNetwork(map_size)
random_agent = RandomAgent()
trainer = AmoebaTrainer(learning_agent, teaching_agents=[random_agent], self_play=True,
                        reward_calculator=PolicyGradients(teach_with_losses=False))

trainer.train(batch_size=500, map_size=map_size, num_episodes=5)
