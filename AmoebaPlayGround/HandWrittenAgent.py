import collections
import math

import numpy as np

import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.AmoebaAgent import AmoebaAgent
from AmoebaPlayGround.GameBoard import AmoebaBoard, Symbol

Importance = collections.namedtuple('Importance', 'level value')


class HandWrittenAgent(AmoebaAgent):
    def get_step(self, game_boards):
        steps = []
        for board in game_boards:
            self.board: AmoebaBoard = board
            steps.append(self.get_step_for_game())
        return steps

    def get_step_for_game(self):
        offensive_importances = self.get_importances(self.board.perspective.get_symbol())
        defensive_importances = self.get_importances(self.board.perspective.get_other_player().get_symbol())
        highest_offensive_level = self.get_highest_importance_level(offensive_importances)
        highest_defensive_level = self.get_highest_importance_level(defensive_importances)
        if highest_offensive_level >= highest_defensive_level:
            importances_to_use = offensive_importances
        else:
            importances_to_use = defensive_importances
        return self.get_step_from_importances(importances_to_use)

    def get_step_from_importances(self, importances):
        best_move = (0, 0)
        highest_importance_level = importances[0, 0].level
        highest_importance_value = importances[0, 0].value
        for row_index, row in enumerate(importances):
            for column_index, importance in enumerate(row):
                if importance.level > highest_importance_level:
                    highest_importance_level = importance.level
                    highest_importance_value = importance.value
                    best_move = (row_index, column_index)
                elif importance.level == highest_importance_level and \
                                highest_importance_value < importance.value:
                    highest_importance_value = importance.value
                    best_move = (row_index, column_index)
        return best_move

    def get_highest_importance_level(self, importances):
        max_level = Amoeba.win_sequence_length - 1
        highest_level = 0
        for importance in importances.flat:
            if importance.level > highest_level:
                highest_level = importance.level
                if highest_level == max_level:
                    return highest_level
        return highest_level

    def get_importances(self, player_symbol):
        importances = np.empty(self.board.shape, dtype=Importance)
        for row_index, row in enumerate(self.board):
            for column_index, cell in enumerate(row):
                if cell == Symbol.EMPTY:
                    importance = self.calculate_importance_for_cell(player_symbol
                                                                    , row_index, column_index)
                else:
                    importance = Importance(level=-1, value=0)
                importances[row_index, column_index] = importance
        return importances

    def calculate_importance_for_cell(self, player_symbol, row_index, column_index) -> Importance:
        distance_to_check = Amoeba.win_sequence_length - 1
        vertical_importance = self.get_importance_in_direction(player_symbol=player_symbol,
                                                               row_start=row_index - distance_to_check,
                                                               column_start=column_index,
                                                               row_direction=1, column_direction=0)
        diagonal_importance_1 = self.get_importance_in_direction(player_symbol=player_symbol,
                                                                 row_start=row_index - distance_to_check,
                                                                 column_start=column_index - distance_to_check,
                                                                 row_direction=1, column_direction=1)
        horizontal_importance = self.get_importance_in_direction(player_symbol=player_symbol,
                                                                 row_start=row_index,
                                                                 column_start=column_index - distance_to_check,
                                                                 row_direction=0, column_direction=1)
        diagonal_importance_2 = self.get_importance_in_direction(player_symbol=player_symbol,
                                                                 row_start=row_index + distance_to_check,
                                                                 column_start=column_index - distance_to_check,
                                                                 row_direction=-1, column_direction=1)
        importances = [vertical_importance, diagonal_importance_1, horizontal_importance, diagonal_importance_2]
        return self.combine_importances(importances)

    def combine_importances(self, importances):
        max_level = 0
        combined_value = 0
        for importance in importances:
            if importance.level > max_level:
                max_level = importance.level
                combined_value = importance.value
            if importance.level == max_level:
                combined_value += importance.level
        return Importance(level=max_level, value=combined_value)

    def get_importance_in_direction(self, player_symbol, row_start, column_start,
                                    row_direction, column_direction) -> Importance:
        window_start = (row_start, column_start)
        window_length = Amoeba.win_sequence_length
        window_count = Amoeba.win_sequence_length
        max_window_importance_level = 0
        max_window_importance_value = 0
        for window_index in range(window_count):
            if self.window_is_within_bounds(window_start, row_direction, column_direction, window_length):
                window_importance = \
                    self.get_window_importance(player_symbol, window_index, window_start, row_direction
                                               , column_direction, window_length)
                if window_importance.level > max_window_importance_level:
                    max_window_importance_level = window_importance.level
                    max_window_importance_value = window_importance.value
                if window_importance.level == max_window_importance_level:
                    max_window_importance_value += window_importance.value
            window_start = (window_start[0] + row_direction, window_start[1] + column_direction)
        return Importance(level=max_window_importance_level, value=max_window_importance_value)

    def get_window_importance_value(self, window_index, window_length):
        return window_length / 2 - math.fabs(window_index - (window_length / 2))

    def window_is_within_bounds(self, start_cell_index, row_direcion, column_direction, window_length):
        last_cell_index = (start_cell_index[0] + row_direcion * (window_length - 1)
                           , start_cell_index[1] + column_direction * (window_length - 1))
        return self.board.is_within_bounds(start_cell_index) \
               and self.board.is_within_bounds(last_cell_index)

    def get_window_importance(self, player_symbol, window_index, start_index,
                              row_direcion, column_direction, window_length):
        own_symbol_count = 0
        cell_index = start_index
        sum_own_symbol_distance = 0
        for index in range(window_length):
            cell = self.board.get(cell_index)
            if cell == player_symbol:
                own_symbol_count += 1
                sum_own_symbol_distance += math.fabs(Amoeba.win_sequence_length - 1 - index - window_index)
            elif cell != Symbol.EMPTY:
                return Importance(level=0, value=0)
            cell_index = (cell_index[0] + row_direcion, cell_index[1] + column_direction)
        if own_symbol_count == 0:
            average_own_symbol_closeness = 0
        else:
            average_own_symbol_closeness = own_symbol_count / sum_own_symbol_distance
        return Importance(level=own_symbol_count, value=average_own_symbol_closeness)

    def get_name(self):
        return 'HandWrittenAgent'
