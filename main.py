from AmoebaPlayGround.AmoebaTrainer import AmoebaTrainer
from AmoebaPlayGround.NeuralAgent import NeuralNetwork

# graphicalView = GraphicalView((10, 10))
# game_board = np.array([[0,0,0,0],[0,1,1,1],[-1,-1,-1,0],[0,1,-1,0]])
# graphicalView.display_game_state(game_board)
# demo
#consoleAgent = ConsoleAgent()
# view = ConsoleView()
size = (10, 10)
agent = NeuralNetwork(size)
trainer = AmoebaTrainer(agent)
trainer.train(batch_size=1000, map_size=size, view=None, num_episodes=5)
