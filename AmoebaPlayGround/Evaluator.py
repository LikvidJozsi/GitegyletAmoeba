from AmoebaPlayGround import AmoebaAgent


class Evaluator:
    def evaluate_agent(self, agent: AmoebaAgent):
        pass

    def set_reference_agent(self, agent: AmoebaAgent, rating):
        pass


class EloEvaluator(Evaluator):

# 1. evaluator recieves an agent
# 2. takes this agent and runs a 1000 games between it and the agent evaluated against ( half one staring half the other)
# 3. calculates elo rating from the elo rating of the reference agent and the win ratio
# 4. returns the win ratio and elo rating
# 5. initially elo of the first agent is 0, what we evaluate against is the previous episode agent
# future ideas:
# what if elo rating is not consistent when evaluating against multiple agents?
