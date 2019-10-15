from AmoebaPlayGround.Amoeba import AmoebaGame,Player

class AmoebaTrainer:
    def __init__(self,x_agent,o_agent):
        self.x_agent = x_agent
        self.o_agent = o_agent

    def init_games(self,batch_size,map_size,view):
        games = []
        for index in range(batch_size):
            games.append(AmoebaGame(map_size,view))
        return games


    def train(self,batch_size=1,map_size=(8,8),view=None):
        games = self.init_games(batch_size,map_size,view)
        games = self.play_all_games(games)
        # TODO training
        # 1. calculate rewards
        # 2. train agent on each game
        # 3. there should be some sotrt of memory dunno gotta read

    def play_all_games(self,active_games):
        next_agent = self.get_next_agent(active_games[0])
        finished_games = []
        while len(active_games) != 0:
            maps = []
            for game in active_games:
                maps.append(game.get_map_for_next_player())
            actions = next_agent.get_step(maps)
            for index,game in enumerate(active_games):
                game.step(actions[index])
                if game.has_game_ended():
                    finished_games.append(game)
            active_games = [game for game in active_games if not game in finished_games]
        return finished_games


    def get_next_agent(self,game):
        if game.next_player == Player.X:
            return self.x_agent
        else:
            return self.o_agent