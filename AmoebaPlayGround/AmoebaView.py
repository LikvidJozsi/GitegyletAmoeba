import queue
import threading
from tkinter import *

from AmoebaPlayGround.Amoeba import Symbol, Player
from AmoebaPlayGround.AmoebaAgent import AmoebaAgent


# The display function of a view is called by the AmoebaGame at init time and after every move
class AmoebaView:
    def display_game_state(self, game_board):
        pass

    def game_ended(self, winner: Player):
        pass


class ConsoleView(AmoebaView):
    def display_game_state(self, game_board):
        for line in game_board:
            for cell in line:
                print(self.get_cell_representation(cell), end='')
            print()

    def get_cell_representation(cell: int):
        if cell == 0:
            return '.'
        if cell == 1:
            return 'x'
        if cell == -1:
            return 'o'
        raise Exception('Unknown cell value')

    def game_ended(self, winner: Player):
        if winner != Player.NOBODY:
            print('Game ended! Winner is %s' % (winner.name))
        else:
            print('Game ended! It is a draw')



class BoardCell:
    def __init__(self, window, row, column, symbol_size):
        self.symbol = Symbol.EMPTY
        self.symbol_size = symbol_size
        self.row = row
        self.column = column
        self.canvas = Canvas(window, width=symbol_size, height=symbol_size)
        self.canvas['bg'] = 'white'
        self.canvas.grid(column=column, row=row, padx=2, pady=2)

    def set_click_event_handler(self, click_event_handler):
        self.canvas.bind("<Button-1>", click_event_handler)

    def is_empty(self):
        return self.symbol == Symbol.EMPTY

    def update(self, new_symbol):
        if self.symbol != new_symbol:
            self.canvas.delete("all")
            if new_symbol == Symbol.X:
                self.drawX()
            elif new_symbol == Symbol.O:
                self.drawO()

    def drawX(self):
        self.canvas.create_line(2, 2, self.symbol_size, self.symbol_size)
        self.canvas.create_line(2, self.symbol_size, self.symbol_size, 2)

    def drawO(self):
        # drawing a circle is done by giving its enclosing rectangle
        self.canvas.create_oval(2, 2, self.symbol_size, self.symbol_size)


class GraphicalView(AmoebaView, AmoebaAgent):
    def __init__(self, board_size, symbol_size=30):
        self.symbol_size = symbol_size
        self.board_size = board_size
        self.board_update_queue = queue.Queue()
        self.message_queue = queue.Queue()
        self.clicked_cell = None
        self.move_entered_event = threading.Event()
        gui_thread = threading.Thread(target=self.create_window)
        gui_thread.start()

    def create_window(self):
        self.window = Tk()
        self.window.title("Amoeba width:%d height:%d" % (self.board_size[1], self.board_size[0]))
        self.window.geometry('500x500')
        self.window.configure(background='gray')
        call_delay_in_milliseconds = 100
        self.window.after(call_delay_in_milliseconds, self.check_for_board_update)
        self.game_board = self.create_game_board()
        self.window.mainloop()

    def create_game_board(self):
        game_board = []
        for row in range(self.board_size[0]):
            board_row = []
            for column in range(self.board_size[1]):
                board_cell = BoardCell(self.window, row, column, self.symbol_size)
                board_cell.set_click_event_handler(self.create_click_event_handler(board_cell))
                board_row.append(board_cell)
            game_board.append(board_row)
        return game_board

    def create_click_event_handler(self, board_cell):
        def click_event_handler(event):
            if board_cell.is_empty():
                self.clicked_cell = board_cell.row, board_cell.column
                self.move_entered_event.set()
                self.move_entered_event.clear()

        return click_event_handler

    def check_for_board_update(self):
        if not self.board_update_queue.empty():
            game_board = self.board_update_queue.get()
            self.update_board(game_board)
        if not self.message_queue.empty():
            message = self.message_queue.get()
            self.display_message(message)
        call_delay_in_milliseconds = 100
        self.window.after(call_delay_in_milliseconds, lambda: self.check_for_board_update())

    def display_message(self, message):
        label = Label(self.window, text=message)
        label.grid(column=0, row=0, columnspan=6)

    def update_board(self, game_board):
        self.validate_game_board_update(game_board)
        for row_index, row in enumerate(self.game_board):
            for column_index, cell in enumerate(row):
                new_symbol = Symbol(game_board[row_index, column_index])
                cell.update(new_symbol)

    def validate_game_board_update(self, game_board):
        if self.board_size != game_board.shape:
            raise Exception("Size of gameboard (%d) does not match size of size of graphical view(%d)" % (
                game_board.shape, self.board_size))

    def get_step(self, game_boards):
        if len(game_boards) != 1:
            raise Exception('GraphicalView does not support multiple parallel matches')
        self.move_entered_event.wait()
        return [self.clicked_cell, ]

    def train(self, game_board, played_action, reward):
        pass

    def display_game_state(self, game_board):
        self.board_update_queue.put(game_board)

    def game_ended(self, winner: Player):
        if winner != Player.NOBODY:
            text = 'Game ended! Winner is %s' % (winner.name)
        else:
            text = 'Game ended! It is a draw'
        self.message_queue.put(text)
