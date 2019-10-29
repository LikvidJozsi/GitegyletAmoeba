from AmoebaPlayGround.Amoeba import AmoebaGame, Player
from AmoebaPlayGround.RewardCalculator import PolicyGradients


class AmoebaTrainer:
    def __init__(self, x_agent, o_agent, reward_calculator=PolicyGradients()):
        self.x_agent = x_agent
        self.o_agent = o_agent
        self.reward_calculator = reward_calculator

    def init_games(self, batch_size, map_size, view):
        games = []
        for index in range(batch_size):
            games.append(AmoebaGame(map_size, view))
        return games

    def train(self, batch_size=1, map_size=(8, 8), view=None, num_episodes=1):
        for episode_index in range(num_episodes):
            print('Episode %d' % episode_index)
            games = self.init_games(batch_size, map_size, view)
            games = self.play_all_games(games)
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


    def play_all_games(self, active_games):
        finished_games = []
        while len(active_games) != 0:
            next_agent = self.get_next_agent(active_games[0])  # the same agent has its turn in every active game at the
            # same time, therfore getting the agent of any of them is enough
            maps = self.get_maps_of_games(active_games)
            actions = next_agent.get_step(maps)
            for index, game in enumerate(active_games):
                game.step(actions[index])
                if game.has_game_ended():
                    finished_games.append(game)
            active_games = [game for game in active_games if not game in finished_games]
        return finished_games

    def get_maps_of_games(self, games):
        maps = []
        for game in games:
            maps.append(game.get_map_for_next_player())
        return maps

    def get_next_agent(self, game):
        if game.next_player == Player.X:
            return self.x_agent
        else:
            return self.o_agent
