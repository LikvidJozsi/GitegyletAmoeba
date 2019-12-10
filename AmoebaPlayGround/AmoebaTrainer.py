import os
from datetime import datetime

from AmoebaPlayGround import AmoebaAgent
from AmoebaPlayGround.Evaluator import EloEvaluator
from AmoebaPlayGround.Evaluator import fix_reference_agents
from AmoebaPlayGround.GameGroup import GameGroup
from AmoebaPlayGround.NeuralAgent import NeuralAgent
from AmoebaPlayGround.RewardCalculator import PolicyGradients

logs_folder = 'Logs/'

class AmoebaTrainer:
    def __init__(self, learning_agent, teaching_agents, reward_calculator=PolicyGradients(), self_play=True):
        self.learning_agent: AmoebaAgent = learning_agent
        self.reward_calculator = reward_calculator
        self.teaching_agents = teaching_agents
        self.self_play = self_play
        if self.self_play:
            # TODO have a factory method so neuralagent doesn't have to be hardcoded
            self.learning_agent_with_old_state = NeuralAgent(model_type=self.learning_agent.model_type)
            self.teaching_agents.append(self.learning_agent_with_old_state)

    def train(self, batch_size=1, view=None, num_episodes=1, log_file_name=""):
        if log_file_name == "":
            date_time = datetime.now()
            log_file_name = date_time.strftime("%Y-%m-%d_%H-%M-%S")
        log_file_name = log_file_name + ".log"
        log_file_name = os.path.join(logs_folder, log_file_name)
        log_file = open(log_file_name, mode="a+", newline='')
        log_file.write("episode\taverage_game_length\tloss\trating\t")
        for reference_agent in fix_reference_agents:
            log_file.write("%s\t" % reference_agent.name)
        log_file.write("\n")
        self.batch_size = batch_size
        self.view = view
        evaluator = EloEvaluator()
        if self.self_play:
            evaluator.set_reference_agent(self.learning_agent_with_old_state)
        for episode_index in range(num_episodes):
            log_file.write("%d\t" % episode_index)
            print('\nEpisode %d:' % episode_index)
            played_games = []
            aggregate_average_game_length = 0
            for teacher_index, teaching_agent in enumerate(self.teaching_agents):
                print('Playing games against ' + teaching_agent.get_name())
                games, average_game_length = self.play_games_between_agents(self.learning_agent, teaching_agent)
                aggregate_average_game_length += average_game_length
                played_games.extend(games)
            aggregate_average_game_length /= len(self.teaching_agents)
            log_file.write("%f\t" % aggregate_average_game_length)
            training_samples = self.reward_calculator.get_training_data(played_games)
            if self.self_play:
                self.learning_agent.copy_weights_into(self.learning_agent_with_old_state)
            print('Training agent:')
            train_history = self.learning_agent.train(training_samples)
            last_loss = train_history.history['loss'][-1]
            log_file.write("%f\t" % last_loss)
            print('Evaluating agent:')
            scores_against_fixed, agent_rating = evaluator.evaluate_agent(self.learning_agent)
            log_file.write("%f\t" % agent_rating)
            for reference_agent in fix_reference_agents:
                log_file.write("%f\t" % scores_against_fixed[reference_agent.name])
            print('Learning agent rating: %f' % agent_rating)
            if self.self_play:
                evaluator.set_reference_agent(self.learning_agent_with_old_state, agent_rating)
            log_file.write("\n")
        log_file.close()

    def play_games_between_agents(self, agent_one, agent_two):
        game_group_1 = GameGroup(int(self.batch_size / 2), agent_one, agent_two,
                                 self.view, log_progress=True)
        game_group_2 = GameGroup(int(self.batch_size / 2), agent_two, agent_one,
                                 self.view, log_progress=True)

        played_games_1, average_game_length_1 = game_group_1.play_all_games()
        played_games_2, average_game_length_2 = game_group_2.play_all_games()
        played_games_1.extend(played_games_2)
        return played_games_1, (average_game_length_1 + average_game_length_2) / 2
