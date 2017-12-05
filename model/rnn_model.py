# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

def rnn_model(model, input_data, output_data, vocab_size, rnn_size=128, num_layers=2, batch_size=64, learning_rate=0.01):
    end_points = {}
    if model == 'rnn':
        cell_fun = tf.contrib.rnn.BasicRNNCell
    elif model == 'gru':
        cell_fun = tf.contrib.rnn.GRUCell
    elif model == 'lstm':
        cell_fun = tf.contrib.rnn.BasicLSTMCell

    cell = cell_fun(rnn_size, state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell(
        [cell] * num_layers, state_is_tuple=True)
