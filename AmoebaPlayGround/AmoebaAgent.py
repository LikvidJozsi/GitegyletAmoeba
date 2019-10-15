import numpy as np

class AmoebaAgent:
    def get_step(self,game_board):
        pass

    def train(self,game_board,played_action,reward):
        pass

class ConsoleAgent(AmoebaAgent):
    def get_step(self, game_boards):
        answers = np.zeros((len(game_boards),2),dtype='int')
        print('Give position in row column format (zero indexing):')
        for index,game_board in enumerate(game_boards):
            answer = input().split(' ')
            answers[index] = int(answer[0]), int(answer[1])
        return answers
