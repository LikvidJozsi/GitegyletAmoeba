from AmoebaPlayGround.RewardCalculator import PolicyGradients
from AmoebaPlayGround.GameGroup import GameGroup

class AmoebaTrainer:
    def __init__(self, x_agent, o_agent, reward_calculator=PolicyGradients()):
        self.x_agent = x_agent
        self.o_agent = o_agent
        self.reward_calculator = reward_calculator

    def train(self, batch_size=1, map_size=(8, 8), view=None, num_episodes=1):
        for episode_index in range(num_episodes):
            print('Episode %d' % episode_index)
            game_group = GameGroup(batch_size, map_size, self.x_agent, self.o_agent, view)  # TODO swap x and o agents
            games = game_group.play_all_games()
            training_samples = self.reward_calculator.get_training_data(games)
            agents_to_train = self.get_agents_to_train()
            for agent in agents_to_train:
                agent.train(training_samples)

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

    def get_agents_to_train(self):
        if self.x_agent == self.o_agent:
            return [self.x_agent, ]
        else:
            return [self.x_agent, self.o_agent]

