import math

from AmoebaPlayGround.GameGroup import GameGroup
from AmoebaPlayGround.AmoebaAgent import AmoebaAgent
from AmoebaPlayGround.GameBoard import Player


class Evaluator:
    def evaluate_agent(self, agent: AmoebaAgent):
        pass

    def set_reference_agent(self, agent: AmoebaAgent, rating):
        pass


class EloEvaluator(Evaluator):
    def __init__(self, map_size, evaluation_match_number=100):
        self.reference_agent = None
        self.reference_agent_rating = None
        self.evaluation_match_number = evaluation_match_number
        self.map_size = map_size

    def evaluate_agent(self, agent: AmoebaAgent):
        game_group_size = int(self.evaluation_match_number / 2)
        game_group_reference_starts = GameGroup(game_group_size, self.map_size, self.reference_agent, agent)
        game_group_agent_started = GameGroup(game_group_size, self.map_size, agent, self.reference_agent)
        finished_games_reference_started = game_group_reference_starts.play_all_games()
        finished_games_agent_started = game_group_agent_started.play_all_games()

        games_agent_won, games_reference_won, draw_games = self.get_win_statistics(finished_games_agent_started)
        won_by_reference, lost_by_reference, draw = self.get_win_statistics(finished_games_reference_started)
        games_agent_won += lost_by_reference + 1
        games_reference_won += won_by_reference + 1
        draw_games += draw + 1
        all_games_num = games_agent_won + games_reference_won + draw_games
        agent_expected_score = games_agent_won / all_games_num + 0.5 * draw_games / all_games_num
        agent_rating = self.reference_agent_rating - 400 * math.log10(1 / agent_expected_score - 1)
        return agent_rating

    def get_win_statistics(self, games):
        games_x_won = 0
        games_o_won = 0
        games_draw = 0
        for game in games:
            winner = game.winner
            if winner == Player.X:
                games_x_won += 1
            elif winner == Player.O:
                games_o_won += 1
            else:
                games_draw += 1
        return games_x_won, games_o_won, games_draw

    def set_reference_agent(self, agent: AmoebaAgent, rating):
        self.reference_agent = agent
        self.reference_agent_rating = rating

# 1. evaluator recieves an agent
# 2. takes this agent and runs a 1000 games between it and the agent evaluated against ( half one staring half the other)
# 3. calculates elo rating from the elo rating of the reference agent and the win ratio
# 4. returns the win ratio and elo rating
# 5. initially elo of the first agent is 0, what we evaluate against is the previous episode agent
# future ideas:
# what if elo rating is not consistent when evaluating against multiple agents?
