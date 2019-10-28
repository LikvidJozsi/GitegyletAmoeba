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

    def train(self, batch_size=1, map_size=(8, 8), view=None):
        games = self.init_games(batch_size, map_size, view)
        games = self.play_all_games(games)
        training_samples = self.reward_calculator.get_training_data(games)
        for training_sample in training_samples:
            print(training_sample)
        # TODO training
        # 1. play_all_games is already written but there is no neural network agent implementation yet. Ideas for it:
        #    - outputs size same as input size each output feature is the probability of making a move in that cell(non empty cells are ignored)
        #    - network could be convolution -> dense -> deconvolution or convolution -> locally connected, or simply convolution -> dense or no
        #      convolution at all though i am skeptical about that
        #    - the move is not the most probable one but a sample from the probability distribution of every move, this makes the agent discover
        #      new tactics
        #    - this algorithm can be referred to as monte carlo tree search because putting monte carlo in an algorithm has the same effect as
        #      painting flames on a car
            # 2. DONE, could still be extended with different reward calculation methods in the future, like heuristics
        # 3. these combos, lets call them training points can be fed into the network for supervised training. Questions:
        #    - does it need more than one epoch?
        #    - should only the latest batch of games be fed, or earlier ones too?
        # 4. evalutating the new network should be done by having it play multiple games against previous versions, but could there be a way to provide
        #    an absolute value of performance instead of a relative one (someting like how chess player ratings work maybe)?
        # These 4 steps are needed to get the full workflow of learning going which would be the second phase of the homework

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
