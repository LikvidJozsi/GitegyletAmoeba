from AmoebaPlayGround.AmoebaTrainer import AmoebaTrainer
from AmoebaPlayGround.NeuralAgent import NeuralNetwork
from AmoebaPlayGround.AmoebaAgent import RandomAgent

# graphicalView = GraphicalView((10, 10))
# game_board = np.array([[0,0,0,0],[0,1,1,1],[-1,-1,-1,0],[0,1,-1,0]])
# graphicalView.display_game_state(game_board)
# demo
# consoleAgent = ConsoleAgent()
# view = ConsoleView()

map_size = (8, 8)
win_sequence_length = 5

agent = NeuralNetwork(map_size)
random_agent = RandomAgent()

trainer_ai_random = AmoebaTrainer(agent, random_agent)
trainer_ai_ai = AmoebaTrainer(agent)

for i in range(100):
    print('\nCycle {}'.format(i))
    print('\nTraining against random agent.')
    trainer_ai_random.train(batch_size=1000, map_size=map_size, win_sequence_length=win_sequence_length, view=None,
                            num_episodes=1)
    print('\nTraining against itself.\nCycle {}'.format(i))
    trainer_ai_ai.train(batch_size=2000, map_size=map_size, win_sequence_length=win_sequence_length, view=None,
                        num_episodes=1)

    if i % 10 == 0:
        agent.save('model{}'.format(i))
