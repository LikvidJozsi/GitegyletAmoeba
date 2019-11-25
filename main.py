import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.AmoebaAgent import RandomAgent
from AmoebaPlayGround.AmoebaTrainer import AmoebaTrainer
from AmoebaPlayGround.NeuralAgent import NeuralAgent, ShallowNetwork
# graphicalView = GraphicalView((10, 10))
# game_board = np.array([[0,0,0,0],[0,1,1,1],[-1,-1,-1,0],[0,1,-1,0]])
# graphicalView.display_game_state(game_board)
# demo
# consoleAgent = ConsoleAgent()
# view = ConsoleView()
from AmoebaPlayGround.RewardCalculator import PolicyGradients

Amoeba.map_size = (8, 8)
Amoeba.win_sequence_length = 5

learning_agent = NeuralAgent(model_type=ShallowNetwork())
random_agent = RandomAgent()
trainer = AmoebaTrainer(learning_agent, teaching_agents=[random_agent], self_play=True,
                        reward_calculator=PolicyGradients(teach_with_losses=False))

trainer.train(batch_size=500, num_episodes=5)
