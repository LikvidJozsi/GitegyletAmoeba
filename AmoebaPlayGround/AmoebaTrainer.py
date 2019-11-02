from AmoebaPlayGround.Evaluator import EloEvaluator
from AmoebaPlayGround.RewardCalculator import PolicyGradients
from AmoebaPlayGround.GameGroup import GameGroup


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
            print('Episode %d' % episode_index)
            game_group = GameGroup(batch_size, map_size, self.learning_agent, self.teaching_agent,
                                   view)  # TODO swap x and o agents
            games = game_group.play_all_games()
            training_samples = self.reward_calculator.get_training_data(games)
            self.learning_agent.train(training_samples)
            agent_rating = evaluator.evaluate_agent(self.learning_agent)
            print('Learning agent rating: %f' % agent_rating)
            evaluator.set_reference_agent(self.learning_agent, agent_rating)

            # 1. There is a basic neural network implementation that is conv -> dense. Further ideas:
        #    - network could be convolution -> dense -> deconvolution or convolution -> locally connected, or simply convolution -> dense or no
        #      convolution at all though i am skeptical about that
        #    - alphago zero used the resnet architecture
        # 2. DONE, could still be extended with different reward calculation methods in the future, like heuristics
        # 3. There is a basic neural network implementation. Remaining questions:
        #    - does it need more than one epoch?
        #    - should only the latest batch of games be fed, or earlier ones too?
        # TODO evaluation
        # 4. evalutating the new network should be done by having it play multiple games against previous versions, but could there be a way to provide
        #    an absolute value of performance instead of a relative one (someting like how chess player ratings work maybe)?
        # These 4 steps are needed to get the full workflow of learning going which would be the second phase of the homework
