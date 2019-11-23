from AmoebaPlayGround import AmoebaAgent
from AmoebaPlayGround.Evaluator import EloEvaluator
from AmoebaPlayGround.GameGroup import GameGroup
from AmoebaPlayGround.NeuralAgent import NeuralAgent
from AmoebaPlayGround.RewardCalculator import PolicyGradients


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

    def train(self, batch_size=1, view=None, num_episodes=1):
        self.batch_size = batch_size
        self.view = view
        evaluator = EloEvaluator()
        evaluator.set_reference_agent(self.learning_agent_with_old_state)
        for episode_index in range(num_episodes):
            print('\nEpisode %d:' % episode_index)
            played_games = []
            for teacher_index, teaching_agent in enumerate(self.teaching_agents):
                print('Playing games against ' + teaching_agent.get_name())
                played_games.extend(self.play_games_between_agents(self.learning_agent, teaching_agent))
            training_samples = self.reward_calculator.get_training_data(played_games)
            if self.self_play:
                self.learning_agent.copy_weights_into(self.learning_agent_with_old_state)
            print('Training agent:')
            self.learning_agent.train(training_samples)
            print('Evaluating agent:')
            agent_rating = evaluator.evaluate_agent(self.learning_agent)
            print('Learning agent rating: %f' % agent_rating)
            if self.self_play:
                evaluator.set_reference_agent(self.learning_agent_with_old_state, agent_rating)


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

    def play_games_between_agents(self, agent_one, agent_two):
        game_group_1 = GameGroup(int(self.batch_size / 2), agent_one, agent_two,
                                 self.view, log_progress=True)
        game_group_2 = GameGroup(int(self.batch_size / 2), agent_two, agent_one,
                                 self.view, log_progress=True)

        played_games = game_group_1.play_all_games()
        played_games.extend(game_group_2.play_all_games())
        return played_games
