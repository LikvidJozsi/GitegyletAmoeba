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
win_sequence_length = 5

agent = NeuralNetwork(map_size)
counter_agent = NeuralNetwork(map_size)
random_agent = RandomAgent()

trainer_ai_random = AmoebaTrainer(agent, random_agent)
trainer_aic_random = AmoebaTrainer(counter_agent, random_agent)
trainer_ai_aic = AmoebaTrainer(agent, counter_agent,
                               reward_calculator=PolicyGradients(teach_with_losses=False))
trainer_aic_ai = AmoebaTrainer(counter_agent, agent,
                               reward_calculator=PolicyGradients(teach_with_losses=False))


batch_againt_random = 500
batch_against_ai = 1000
save_step = 1


def non_episodic_train(trainer, batch_size):
    trainer.train(batch_size, map_size=map_size, view=None,
                  num_episodes=1)


for i in range(100):
    print('\nCycle {}'.format(i))

    # These two steps could be parallelised.
    print('\nTraining agent against random agent.')
    non_episodic_train(trainer_ai_random, batch_againt_random)

    print('\nTraining counter agent against random.\nCycle {}'.format(i))
    non_episodic_train(trainer_aic_random, batch_againt_random)

    print('\nTraining agent against counter agent.\nCycle {}'.format(i))
    non_episodic_train(trainer_ai_aic, batch_against_ai)

    print('\nTraining counter agent against agent.\nCycle {}'.format(i))
    non_episodic_train(trainer_aic_ai, batch_against_ai)

    if i % save_step == 0:
        agent.save('model{}'.format(i))
        counter_agent.save('model_c{}'.format(i))
