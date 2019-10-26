from AmoebaPlayGround.AmoebaAgent import ConsoleAgent, RandomAgent
from AmoebaPlayGround.AmoebaTrainer import AmoebaTrainer
from AmoebaPlayGround.AmoebaView import GraphicalView

graphicalView = GraphicalView((10, 10))
# game_board = np.array([[0,0,0,0],[0,1,1,1],[-1,-1,-1,0],[0,1,-1,0]])
# graphicalView.display_game_state(game_board)
# demo
consoleAgent = ConsoleAgent()
# view = ConsoleView()

trainer = AmoebaTrainer(graphicalView, RandomAgent())
trainer.train(1, map_size=(10, 10), view=graphicalView)
