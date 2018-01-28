'''
    gui_multilabel.py
    Copyright @ 2018 Jiaoyan<jchen11@wpi.edu>, Ziqi<zlin3@wpi.edu>, Han <hjiang@wpi.edu>
    License: MIT

'''

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/27/18 6:39 PM
# @Author  : Han Jiang

from sklearn.model_selection import train_test_split
# import numpy as np
import tensorflow as tf
from parameters import *


class Model():
    def __init__(self, data): ### npy file
        para = parameters()
        self.output_label_size, self.learning_rate, self.epoch, self.display_step, self.first_layer, self.second_layer, self.batch_size = para.return_parameters()
        self.data = data
        self.input_data_size = self.data.shape[1] - 10
        self.weights = {
            'middle1': tf.Variable(tf.truncated_normal([self.input_data_size, self.first_layer])),
            'middle2': tf.Variable(tf.truncated_normal([self.first_layer, self.second_layer])),
            'out': tf.Variable(tf.truncated_normal([self.second_layer, self.output_label_size])),
        }
        self.biases = {
            'middle1': tf.Variable(tf.random_normal([self.first_layer])),
            'middle2': tf.Variable(tf.random_normal([self.second_layer])),
            'out': tf.Variable(tf.random_normal([self.output_label_size])),
        }
        self.x = tf.placeholder(tf.float32, [None, self.input_data_size])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])

    def deal_input(self):
        input = self.data[:,10:]
        label = self.data[:,:10]
        X_train, X_test, y_train, y_test = train_test_split(
        input, label, test_size = 0.30, random_state = 42)
        return X_train, X_test, y_train, y_test

    def FC(self,x):
        h1 = tf.nn.relu(tf.matmul(x,self.weights['middle1']) + self.biases['middle1'])
        h2 = tf.nn.relu(tf.matmul(h1,self.weights['middle2']) + self.biases['middle2'])
        output = tf.nn.sigmoid(tf.matmul(h2,self.weights['out']) + self.biases['out'])
        return output