from AmoebaPlayGround.AmoebaTrainer import AmoebaTrainer
from AmoebaPlayGround.AmoebaView import GraphicalView
from AmoebaPlayGround.NeuralAgent import NeuralNetwork
from AmoebaPlayGround.AmoebaAgent import RandomAgent


map_size = (8, 8)
win_sequence_length = 5

graphicalView = GraphicalView(map_size)
agent = NeuralNetwork(map_size)

trainer = AmoebaTrainer(agent, graphicalView)
trainer.train(batch_size=1, map_size=map_size, win_sequence_length=win_sequence_length, view=graphicalView, num_episodes=1)

