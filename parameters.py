#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/27/18 7:41 PM
# @Author  : Han Jiang

class parameters():
    def __init__(self):
        self.output_label_size = 10
        self.learning_rate = 0.0001
        self.epoch = 30
        self.display_step = 5
        self.first_layer = 30
        self.second_layer = 15
        self.batch_size = 100

    def return_parameters(self):
        return self.output_label_size, self.learning_rate, self.epoch, self.display_step, self.first_layer, self.second_layer, self.batch_size