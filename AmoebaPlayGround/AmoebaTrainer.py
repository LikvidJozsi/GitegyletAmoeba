from AmoebaPlayGround.Evaluator import EloEvaluator
from AmoebaPlayGround.GameGroup import GameGroup
from AmoebaPlayGround.RewardCalculator import PolicyGradients


class AmoebaTrainer:
    def __init__(self, learning_agent, teaching_agent=None, reward_calculator=PolicyGradients()):
        self.learning_agent = learning_agent
        if teaching_agent is None:
            self.teaching_agent = learning_agent
        else:
            self.teaching_agent = teaching_agent
        self.reward_calculator = reward_calculator

    def train(self, batch_size=1, map_size=(8, 8), view=None, num_episodes=1):
        evaluator = EloEvaluator(map_size)
        evaluator.set_reference_agent(self.teaching_agent)
        for episode_index in range(num_episodes):
            print('\nEpisode %d:' % episode_index)
            game_group = GameGroup(batch_size, map_size, self.learning_agent, self.teaching_agent,
                                   view, log_progress=True)  # TODO swap x and o agents
            print('Playing games:')
            games = game_group.play_all_games()
            training_samples = self.reward_calculator.get_training_data(games)
            print('Training agent:')
            self.learning_agent.train(training_samples)
            print('Evaluating agent:')
            agent_rating = evaluator.evaluate_agent(self.learning_agent)
            print('Learning agent rating: %f' % agent_rating)
            evaluator.set_reference_agent(self.learning_agent, agent_rating)

            # 1. There is a basic neural network implementation that is conv -> dense. Further ideas:
        #    - network could be convolution -> dense -> deconvolution or convolution -> locally connected, or simply convolution -> dense or no
        #      convolution at all though i am skeptical about that
        #    - alphago zero used the resnet architecture
        # 2. DONE, could still be extended with different reward calculation methods in the future, like heuristics
        # 3. There is a basic neural network implementation. Remaining questions:
        #    - should only the latest batch of games be fed, or earlier ones too?
            # 4. evalutating the new network is be done by having it play multiple games against the previous version, the
            #    performance of the agent is quantified according to the Elo rating system which calcualtes a rating from
            #    the winrate of the agent and the rating of the previous version
