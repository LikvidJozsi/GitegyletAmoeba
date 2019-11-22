from AmoebaPlayGround import AmoebaAgent
from AmoebaPlayGround.Evaluator import EloEvaluator
from AmoebaPlayGround.GameGroup import GameGroup
from AmoebaPlayGround.NeuralAgent import NeuralNetwork
from AmoebaPlayGround.RewardCalculator import PolicyGradients


class AmoebaTrainer:
    def __init__(self, learning_agent, teaching_agents, reward_calculator=PolicyGradients(), self_play=True):
        self.learning_agent: AmoebaAgent = learning_agent
        self.reward_calculator = reward_calculator
        self.teaching_agents = teaching_agents
        self.self_play = self_play
        if self.self_play:
            # TODO have a factory method so neuralagent doesn't have to be hardcoded
            self.learning_agent_with_old_state = NeuralNetwork(self.learning_agent.board_size)
            self.teaching_agents.append(self.learning_agent_with_old_state)

    def train(self, batch_size=1, map_size=(8, 8), view=None, num_episodes=1):
        evaluator = EloEvaluator(map_size)
        evaluator.set_reference_agent(self.learning_agent_with_old_state)
        for episode_index in range(num_episodes):
            print('\nEpisode %d:' % episode_index)
            played_games = []
            for teacher_index, teaching_agent in enumerate(self.teaching_agents):
                game_group = GameGroup(batch_size, map_size, self.learning_agent, teaching_agent,
                                       view, log_progress=True)  # TODO swap x and o agents
                print('Playing games against agent ' + str(teacher_index))
                played_games.extend(game_group.play_all_games())
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
