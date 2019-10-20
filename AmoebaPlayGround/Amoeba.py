from enum import Enum

import numpy as np


class Player(Enum):
    X = 1
    O = -1
    NOBODY = 0

    def get_other_player(self):
        if self == Player.X:
            return Player.O
        else:
            return Player.X


class Symbol(Enum):
    X = 1
    O = -1
    EMPTY = 0

class AmoebaGame:
    def __init__(self, map_size, view=None):
        self.view = view
        if len(map_size) != 2:
            raise Exception('Map must be two dimensional but found shape %s' % (map_size))
        self.map = np.zeros(map_size, dtype=int)
        self.reset()

    def init_map(self, map_size):
        self.map = np.zeros(map_size, dtype=int)
        self.place_initial_symbol(map_size)

    def place_initial_symbol(self, map_size):
        middle_of_map_y = round(map_size[0] / 2)
        middle_of_map_x = round(map_size[0] / 2)
        self.map[middle_of_map_x][middle_of_map_y] = 1

    def reset(self):
        self.init_map(self.map.shape)
        self.next_player = Player.O
        self.history = []
        self.winner = None
        self.num_steps = 1
        if self.view != None:
            self.view.display_game_state(self.map)

    def step(self, action):
        player_symbol = self.get_symbol_value_of_next_player()
        if self.map[action[0], action[1]] != Symbol.EMPTY.value:
            raise Exception('Trying to place symbol in position already occupied')
        self.map[action[0], action[1]] = player_symbol
        self.history.append(action)
        self.num_steps += 1
        self.next_player = self.next_player.get_other_player()
        if self.view != None:
            self.view.display_game_state(self.map)



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
        return self.num_steps == self.map.size

    def is_there_winning_line_in_direction(self, y_start, x_start, y_direction, x_direction):
        # ....x....
        # only 4 places in each direction count in determining if the new move created a winning condition of
        # a five figure long line
        search_length = 9
        player_symbol = -1 * self.get_symbol_value_of_next_player()
        line_length = 0
        for line_index in range(0, search_length):
            # depending on the direction of the line being searched direction may be 0 meaning the coordinate does
            # not change on any iterations,
            x_offset = line_index * x_direction
            y_offset = line_index * y_direction
            y = y_start + y_offset
            x = x_start + x_offset
            if self.is_within_bounds(y, x) and self.map[y, x] == player_symbol:
                line_length += 1
            else:
                line_length = 0
            if line_length == 5:
                return True
        return False

    def is_within_bounds(self, y, x):
        return 0 <= y and y < self.map.shape[0] and 0 <= x and x < self.map.shape[1]

    def get_symbol_value_of_next_player(self):
        return Symbol.X.value if self.next_player == Player.X else Symbol.O.value

    def get_map_for_next_player(self):
        if self.next_player == Player.X:
            return self.map
        else:
            return self.map * (-1)
