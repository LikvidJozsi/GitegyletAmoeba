import glob
import os
from typing import List

import keras
import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Model
from keras.optimizers import SGD, Adam

from AmoebaPlayGround.AmoebaAgent import AmoebaAgent
from AmoebaPlayGround.GameBoard import AmoebaBoard, Symbol, Player
from AmoebaPlayGround.RewardCalculator import TrainingSample

models_folder = 'Models/'


class NeuralNetwork(AmoebaAgent):
    def __init__(self, board_size=None, model_name=None, load_latest_model=False):
        if board_size is None and model_name is None and not load_latest_model:
            raise Exception('board size, file path and load latest model cannot both be None/False')
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session()
            with self.session.as_default():
                if load_latest_model:
                    self.get_latest_model()
                else:
                    if model_name is None:
                        self.board_size = board_size
                        self.model: Model = self.create_model()
                    else:
                        self.load_model(self.get_model_file_path(model_name))

    def get_latest_model(self):
        list_of_files = glob.glob(os.path.join(models_folder, '*.h5'))
        latest_file = max(list_of_files, key=os.path.getctime)
        self.load_model(latest_file)

    def load_model(self, file_path):
        self.model: Model = keras.models.load_model(file_path)
        self.board_size = self.model.get_layer(index=0).output_shape[1:3]

    def get_model_file_path(self, model_name):
        return os.path.join(models_folder, model_name + '.h5')

    def save(self, model_name):
        with self.graph.as_default():
            with self.session.as_default():
                self.model.save(self.get_model_file_path(model_name))

    def create_model(self):
        input = Input(shape=self.board_size + (2,))
        conv_1 = Conv2D(32, kernel_size=(7, 7), activation='relu', padding='same')(input)
        conv_2 = Conv2D(64, kernel_size=(5, 5), strides=(2, 2), activation='relu', padding='same')(conv_1)
        pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_2)
        conv_3 = Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same')(pooling)
        conv_4 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(conv_3)
        pooling_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_4)
        flatten = Flatten()(pooling_2)
        dense_1 = Dense(128, activation= 'relu')(flatten)
        dropout_1 = Dropout(0.3)(dense_1)
        dense_2 = Dense(256, activation='relu')(dropout_1)
        dropout = Dropout(0.3)(dense_2)
        output = Dense(np.prod(self.board_size), activation='softmax')(dropout)
        model = Model(inputs=input, outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.15))
        #model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.15))

        return model

    def get_step(self, game_boards: List[AmoebaBoard]):
        output = self.get_model_output(game_boards)
        return self.get_steps_from_output(output, game_boards)

    def get_steps_from_output(self, output, game_boards: List[AmoebaBoard]) -> Model:
        steps = []
        for probabilities, game_board in zip(output, game_boards):
            valid_steps, valid_probabilities = self.get_valid_steps(probabilities, game_board)

            if np.random.randint(low=0, high=2) >= 0.4:
                step_in_1d = valid_steps[np.argmax(valid_probabilities)]
            else:
                step_in_1d = np.random.choice(valid_steps, p=valid_probabilities)

            #step_in_1d = np.random.choice(valid_steps, p=valid_probabilities)

            step = self.to_2d(step_in_1d)
            steps.append(step)
        return steps

    def get_valid_steps(self, probabilites, game_board):
        valid_steps = []
        index = 0
        for row in game_board:
            for cell in row:
                if cell == Symbol.EMPTY:
                    valid_steps.append(index)
                index += 1
        valid_probabilities = probabilites[valid_steps]
        return valid_steps, valid_probabilities / np.sum(valid_probabilities)

    def to_2d(self, index_1d):
        return (int(np.floor(index_1d / self.board_size[0])), index_1d % self.board_size[0])

    def to_1d(self, index_2d):
        return int(index_2d[0] * self.board_size[0] + index_2d[1])

    def get_model_output(self, game_boards: List[AmoebaBoard]):
        formatted_input = self.format_input(game_boards)
        with self.graph.as_default():
            with self.session.as_default():
                output = self.model.predict(formatted_input, batch_size=32)
        # disclaimer: output has only one spatial dimension, the map is flattened
        return output

    def format_input(self, game_boards: List[AmoebaBoard]):
        numeric_input = []
        for game_board in game_boards:
            numeric_input.append(self.one_hot_encode_input(game_board))
        return np.array(numeric_input)

    def one_hot_encode_input(self, input):
        numeric_representation = np.zeros(input.shape + (2,))
        for row_index, row in enumerate(input.cells):
            for column_index, cell in enumerate(row):
                if input.perspective == Player.X:
                    dimension_for_x = 0
                else:
                    dimension_for_x = 1
                if cell == Symbol.X:
                    numeric_representation[row_index, column_index, dimension_for_x] = 1
                elif cell == Symbol.O:
                    numeric_representation[row_index, column_index, 1 - dimension_for_x] = 1
        return numeric_representation

    def one_hot_encode_output(self, output):
        expanded_dims = np.zeros(np.prod(self.board_size))
        output_1d = self.to_1d(output)
        expanded_dims[output_1d] = 1
        return expanded_dims

    def one_hot_encode_outputs(self, outputs):
        one_hot_outputs = list(map(self.one_hot_encode_output, outputs))
        return np.array(one_hot_outputs)

    def train(self, training_samples: List[TrainingSample]):
        input, output, weights = TrainingSample.unpack(training_samples)
        output = self.one_hot_encode_outputs(output)
        print('number of training samples: ' + str(len(training_samples)))
        input = np.array([self.one_hot_encode_input(x) for x in input])
        with self.graph.as_default():
            with self.session.as_default():
                self.model.fit(x=input, y=np.array(output), sample_weight=np.array(weights), epochs=8, shuffle=True,
                               verbose=2, batch_size=32)

