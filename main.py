from AmoebaPlayGround.AmoebaTrainer import AmoebaTrainer
from AmoebaPlayGround.NeuralAgent import NeuralNetwork

# graphicalView = GraphicalView((10, 10))
# game_board = np.array([[0,0,0,0],[0,1,1,1],[-1,-1,-1,0],[0,1,-1,0]])
# graphicalView.display_game_state(game_board)
# demo
#consoleAgent = ConsoleAgent()
# view = ConsoleView()
agent = NeuralNetwork((10, 10))
trainer = AmoebaTrainer(agent)
trainer.train(batch_size=1000, map_size=(10, 10), view=None, num_episodes=5)
