from tkinter import *
import threading
import time
import queue
import enum
import copy
from AmoebaPlayGround.AmoebaAgent import AmoebaAgent
# The display function of a view is called by the AmoebaGame at init time and after every move
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

class Symbol(enum.Enum):
    X = 1
    O = -1
    EMPTY = 0

class GraphicalView(AmoebaView,AmoebaAgent):

    def __init__(self,map_size,symbolsize = 30):
        self.symbolsize = symbolsize
        self.map_size = map_size
        self.queue = queue.Queue()
        thread = threading.Thread(target=self.create_window)
        self.clicked_cell = None
        self.event = threading.Event()
        thread.start()

    def create_window(self):
        self.window = Tk()
        self.window.title("Amoeba width:%d height:%d" % (self.map_size[1], self.map_size[0]))
        self.window.geometry('500x500')
        self.window.configure(background='gray')
        self.window.after(100, self.updateGUI)
        self.createGameBoard()
        self.window.mainloop()

    def createGameBoard(self):
        self.game_board = []
        for row in range(self.map_size[0]):
            board_row = []
            for column in range(self.map_size[1]):
                canvas = Canvas(self.window, width=self.symbolsize, height=self.symbolsize)
                canvas['bg'] = 'white'
                canvas.grid(column=column, row=row,padx=2,pady=2)
                canvas.bind("<Button-1>",self.create_click_event_handler(row,column))
                board_row.append((canvas,Symbol.EMPTY))
            self.game_board.append(board_row)

    # externalizing defining the click_event_handler into a seperate function
    #  is required because calling this function will pass row and column index by value ensuring that different cells
    # will not share the same variables, which would mean creating other cells would overwrite them
    def create_click_event_handler(self,row_index, column_index):
        def click_event_handler(event):
            if self.game_board[row_index][column_index][1] == Symbol.EMPTY:
                self.clicked_cell = row_index, column_index
                self.event.set()
                self.event.clear()

        return click_event_handler

    def drawX(self,canvas):
        canvas.create_line(2, 2, self.symbolsize,self.symbolsize)
        canvas.create_line(2, self.symbolsize, self.symbolsize, 2)

    def drawO(self,canvas):
        canvas.create_oval(2, 2, self.symbolsize, self.symbolsize)

    def updateGUI(self):
        if not self.queue.empty():
            game_board = self.queue.get()
            self.update_board(game_board)
        self.window.after(100, lambda: self.updateGUI())


    def update_board(self,game_board):
        if(self.map_size[0] != game_board.shape[0] or self.map_size[1] != game_board.shape[1]):
            raise Exception("Size of gameboard (%d) does not match size of size of graphical view(%d)" % (game_board.shape,self.map_size))
        for row_index,row in enumerate(self.game_board):
            for column_index,element in enumerate(row):
                new_figure = Symbol(game_board[row_index,column_index])
                canvas,current_symbol = element[0],element[1]
                if current_symbol != new_figure:
                    canvas.delete("all")
                    self.game_board[row_index][column_index] = (canvas,new_figure)
                    if new_figure == Symbol.X:
                        self.drawX(canvas)
                    elif new_figure == Symbol.O:
                        self.drawO(canvas)

    def get_step(self, game_boards):
        if len(game_boards) != 1:
            raise Exception('GraphicalView does not support multiple paralell matches')
        self.event.wait()
        return [self.clicked_cell,]

    def train(self, game_board, played_action, reward):
        pass

    def display_game_state(self,game_board):
        self.queue.put(game_board)