import copy
from typing import List

from AmoebaPlayGround.GameBoard import AmoebaBoard, Symbol, Player


class Move:
    def __init__(self, board_state, step, player: Player):
        self.board_state = board_state
        self.step = step
        self.player = player

class AmoebaGame:
    def __init__(self, map_size, view=None):
        self.view = view
        if len(map_size) != 2:
            raise Exception('Map must be two dimensional but found shape %s' % (map_size))
        self.map = AmoebaBoard(map_size, perspective=Player.X)
        self.reset()

    def init_map(self):
        self.map.reset()
        self.place_initial_symbol()

    def place_initial_symbol(self):
        self.map.set(self.map.get_middle_of_map_index(), Symbol.X)

    def reset(self):
        self.init_map()
        self.next_player = Player.O
        self.history = []
        self.winner = None
        self.num_steps = 1
        if self.view is not None:
            self.view.display_game_state(self.map)

    def step(self, action):
        player_symbol = self.next_player.get_symbol()
        if not self.map.is_cell_empty(action):
            raise Exception('Trying to place symbol in position already occupied')
        self.map.set(action, player_symbol)
        self.history.append(action)
        self.num_steps += 1
        self.next_player = self.next_player.get_other_player()
        if self.view is not None:
            self.view.display_game_state(self.map)

    def get_last_moves(self, number_of_steps: int) -> List[Move]:
        moves = []
        map = copy.deepcopy(self.map)
        if len(self.history) < number_of_steps:
            number_of_steps = len(self.history)

        player = self.next_player.get_other_player()
        for index in range(number_of_steps):
            step = self.history[len(self.history) - index - 1]
            map.set(step, Symbol.EMPTY)
            map.perspective = player
            move = Move(map.get_numeric_representation(), step, player)
            moves.append(move)
            player = player.get_other_player()
        return moves

    def has_game_ended(self):
        last_action = self.history[-1]
        y = last_action[0]
        x = last_action[1]
        player_won = (self.is_there_winning_line_in_direction(y_start=y - 4, x_start=x + 0,
                                                              y_direction=1, x_direction=0) or  # vertical
                      self.is_there_winning_line_in_direction(y_start=y - 4, x_start=x - 4,
                                                              y_direction=1, x_direction=1) or  # diagonal1
                      self.is_there_winning_line_in_direction(y_start=y, x_start=x - 4,
                                                              y_direction=0, x_direction=1) or  # horizontal
                      self.is_there_winning_line_in_direction(y_start=y + 4, x_start=x - 4,
                                                              y_direction=-1, x_direction=1))  # diagonal2
        if player_won:
            self.winner = self.next_player.get_other_player()
            if self.view != None:
                self.view.game_ended(self.winner)
            return True
        is_draw = self.is_map_full()
        if is_draw and self.view != None:
            self.view.game_ended(Player.NOBODY)
        return is_draw

    def is_map_full(self):
        return self.num_steps == self.map.get_size()

    def is_there_winning_line_in_direction(self, y_start, x_start, y_direction, x_direction):
        # ....x....
        # only 4 places in each direction count in determining if the new move created a winning condition of
        # a five figure long line
        search_length = 9
        player_symbol = self.next_player.get_other_player().get_symbol()  # this is ugly, get_other_player is needed
        # because this function is evaluated AFTER the move
        line_length = 0
        for line_index in range(0, search_length):
            # depending on the direction of the line being searched direction may be 0 meaning the coordinate does
            # not change on any iterations,
            x_offset = line_index * x_direction
            y_offset = line_index * y_direction
            y = y_start + y_offset
            x = x_start + x_offset
            if self.map.is_within_bounds((y, x)) and self.map.get((y, x)) == player_symbol:
                line_length += 1
            else:
                line_length = 0
            if line_length == 5:
                return True
        return False

    def get_map_for_next_player(self):
        map_to_return = copy.copy(self.map)
        map_to_return.perspective = self.next_player
        return map_to_return

