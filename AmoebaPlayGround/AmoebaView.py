
class AmoebaView:
    def display_game_state(self,game_board):
        pass


class ConsoleView(AmoebaView):
    def display_game_state(self,game_board):
        for line in game_board:
            for cell in line:
                print(self.get_cell_representation(cell),end='')
            print()

    def get_cell_representation(self,cell):
        if cell == 0:
            return '.'
        if cell == 1:
            return 'x'
        if cell == -1:
            return 'o'
        raise Exception('Unknown cell value')