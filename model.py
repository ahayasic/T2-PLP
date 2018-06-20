from keras.layers import Dense
from keras.models import Sequential
from keras import backend as K
import tensorflow as tf
import threading

class SmallMLP(object):
    graph = None
    model = None

    def __init__(self):
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            self.model = Sequential()
            self.model.add(Dense(196, activation='relu', input_dim=784))
            self.model.add(Dense(196, activation='relu'))
            self.model.add(Dense(196, activation='relu'))
            self.model.add(Dense(10,  activation='softmax'))
            self.model.compile(optimizer = "adagrad", 
                               loss      = "categorical_crossentropy",
                               metrics   = ["accuracy"])

    def train_and_test_operations(self):
        def train(x_train, y_train):
            with tf.Session(graph=self.graph) as sess:
                sess.run(tf.global_variables_initializer())
                self.model.fit(x_train, y_train, batch_size=256, epochs=5)

        def test(x_test, y_test):
            with tf.Session(graph=self.graph) as sess:
                sess.run(tf.global_variables_initializer())
                return self.model.evaluate(x_test, y_test)

        return train, test


class MediumMLP(object):
    graph = None
    model = None

    def __init__(self):
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            self.model = Sequential()
            self.model.add(Dense(392, activation='relu', input_dim=784))
            self.model.add(Dense(392, activation='relu'))
            self.model.add(Dense(392, activation='relu'))
            self.model.add(Dense(10,  activation='softmax'))
            self.model.compile(optimizer = "adagrad", 
                               loss      = "categorical_crossentropy",
                               metrics   = ["accuracy"])

    def train_and_test_operations(self):
        def train(x_train, y_train):
            with tf.Session(graph=self.graph) as sess:
                sess.run(tf.global_variables_initializer())
                self.model.fit(x_train, y_train, batch_size=256, epochs=5)

        def test(x_test, y_test):
            with tf.Session(graph=self.graph) as sess:
                sess.run(tf.global_variables_initializer())
                return self.model.evaluate(x_test, y_test)

        return train, test


class BigMLP(object):
    graph = None
    model = None

    def __init__(self):
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            self.model = Sequential()
            self.model.add(Dense(784, activation='relu', input_dim=784))
            self.model.add(Dense(784, activation='relu'))
            self.model.add(Dense(784, activation='relu'))
            self.model.add(Dense(10,  activation='softmax'))
            self.model.compile(optimizer = "adagrad", 
                               loss      = "categorical_crossentropy",
                               metrics   = ["accuracy"])

    def train_and_test_operations(self):
        def train(x_train, y_train):
            with tf.Session(graph=self.graph) as sess:
                sess.run(tf.global_variables_initializer())
                self.model.fit(x_train, y_train, batch_size=256, epochs=5)

        def test(x_test, y_test):
            with tf.Session(graph=self.graph) as sess:
                sess.run(tf.global_variables_initializer())
                return self.model.evaluate(x_test, y_test)

        return train, test