#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/27/18 8:06 PM
# @Author  : Han Jiang
from Model import *
import tensorflow as tf
from parameters import *
import numpy as np


class control():
    def __init__(self, data):
        self.model = Model(data)
        self.X_train, self.X_test, self.y_train, self.y_test = self.model.deal_input()

    # whitening:
    def whiten(self, faces):
        covarianceMatrix = faces.T.dot(faces) + 1e-3 * np.eye(self.X_train.shape[1])
        eigenvalues, eigenvectors = np.linalg.eigh(covarianceMatrix)
        eigenvaluesHalf = np.power(eigenvalues, -1 / 2)
        eigenvalues_update = np.diag(eigenvaluesHalf)
        L = eigenvectors.dot(eigenvalues_update)
        return L

    def train(self):
        x = tf.placeholder(tf.float32, [None, self.X_train.shape[1]], name="data")
        y = tf.placeholder(tf.float32, shape=[None, self.model.output_label_size], name="label")

        pred = self.model.FC(x)
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.model.learning_rate).minimize(cost)
        index = np.arange(self.X_train.shape[0])
        np.random.shuffle(index)
        input_train = self.X_train.dot(self.whiten(self.X_train))
        # input_train = self.X_train[index]
        shu_x_trian = input_train[index]
        print(shu_x_trian.shape)
        shu_y_trian = self.y_train[index]
        minibatch_number = int(np.ceil(self.X_train.shape[0] / self.model.batch_size))

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            # Keep training until reach max iterations
            batch_loss = []
            saver = tf.train.Saver()
            tf.add_to_collection("pred", pred)

            for i in range(self.model.epoch):
                for j in range(minibatch_number - 1):
                    mini_train_x = shu_x_trian[i * self.model.batch_size:(i + 1) * self.model.batch_size, :]
                    mini_train_y = shu_y_trian[i * self.model.batch_size:(i + 1) * self.model.batch_size, :]
                    sess.run(optimizer, feed_dict={x: mini_train_x, y: mini_train_y})
                    if j % self.model.display_step == 0:
                        loss = sess.run(cost, feed_dict={x: mini_train_x, y: mini_train_y})
                        batch_loss.append(loss)
                        print('epoch ' + str(i) + ', batch ' + str(j) + ', minibatch training_loss ' + str(loss))

                if (i + 1) % self.model.display_step == 0:
                    saver.save(sess, './model/epoch%s.ckpt' % (i))
            print("Optimization Finished!")
            print("Save finished!")

    def test(self, test_data, label, epoch):
        checkpoint_path = './model/epoch%s.ckpt' % (epoch)
        tf.reset_default_graph()
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(
                './model/epoch%s.ckpt.meta' % (epoch))
            saver.restore(sess, checkpoint_path)
            print("Restoring finished!!!")
            pred = tf.get_collection("pred")[0]
            actualPredictions = sess.run(pred, feed_dict={"data:0": test_data})
            soft_actual = sess.run(tf.nn.sigmoid(actualPredictions))
            prediction = np.zeros(label.shape)
            prediction[np.where(soft_actual > 0.5)] = 1
            acc = np.mean(prediction == label)
            print(prediction)
            # print(label)
            print(acc)

    def prediction(self, test_data):
        checkpoint_path = './model/epoch%s.ckpt' % (29)
        tf.reset_default_graph()
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(
                './model/epoch%s.ckpt.meta' % (29))
            saver.restore(sess, checkpoint_path)
            # print("Restoring finished!!!")
            pred = tf.get_collection("pred")[0]
            actualPredictions = sess.run(pred, feed_dict={"data:0": test_data})
            soft_actual = sess.run(tf.nn.sigmoid(actualPredictions))
            prediction = np.zeros(soft_actual.shape)
            prediction[np.where(soft_actual > 0.5)] = 1
            prediction = prediction[0]
            # print(prediction.shape)
            # print(prediction)
            disease = ['ICD9_DGNS_CD_1', 'ICD9_DGNS_CD_2', 'ICD9_DGNS_CD_3',
                       'ICD9_DGNS_CD_4', 'ICD9_DGNS_CD_5', 'ICD9_DGNS_CD_6', 'ICD9_DGNS_CD_7',
                       'ICD9_DGNS_CD_8', 'ICD9_DGNS_CD_9', 'ICD9_DGNS_CD_10']
            pred_disease = []
            for i in range(len(prediction)):
                # print(i)
                if prediction[i] == 1:
                    pred_disease.append(disease[i])
            output_result = "You have high risk of getting the following disease(s): {}".format(pred_disease)
            # print(output_result)
            # print(label)
            # print(acc)
            return output_result