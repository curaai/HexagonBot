import tensorflow as tf
import numpy as np
import os
from random import sample
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DQN:
    def __init__(self, sess, learning_rate, batch_size, width, height, n_action, channel=3):
        self.GAMMA = 0.99
        self.BUFFER_SIZE = 3000
        self.ACTION_SIZE = n_action
        self.STATE_SIZE = 4
        self.state = None

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.sess = sess

        self.width = width
        self.height = height
        self.channel = channel

        self.X = tf.placeholder(tf.float32, [None, self.STATE_SIZE, height, width, channel])
        self.Y = tf.placeholder(tf.float32, [None])
        self.A = tf.placeholder(tf.int32, [None])

        self.main_Q = self._build_network('main')
        self.target_Q = self._build_network('target')
        self.buffer = deque()

        self.cost, self.train_op = self._build_op()

    '''
    conv2d 32 [5, 5]
    pooling [2, 2]
    conv2d 64 [3, 3]
    pooling [2, 2]
    flatten

    fc 1
    fc 2
    output
    '''
    def _build_network(self, name):
        with tf.variable_scope(name):
            model = tf.layers.conv2d(self.X, 32, [5, 5], padding='same', activation=tf.nn.relu)
            model = tf.layers.max_pooling2d(model, pool_size=[2, 2], strides=2)
            model = tf.layers.conv2d(model, 64, [3, 3], padding='same', activation=tf.nn.relu)
            model = tf.layers.max_pooling2d(model, pool_size=[2, 2], strides=1)
            model = tf.contrib.layers.flatten(model)

            model = tf.layers.dense(model, 256, activation=tf.nn.relu)
            model = tf.layers.dense(model, 256, activation=tf.nn.relu)

            Q = tf.layers.dense(model, self.ACTION_SIZE, activation=None)
            return Q

    def _build_op(self):
        one_hot = tf.one_hot(self.A, self.ACTION_SIZE, 1.0, 0.0)
        Q_value = tf.reduce_sum(tf.multiply(self.main_Q, one_hot), axis=1)
        cost = tf.reduce_mean(tf.square(self.Y - Q_value))
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

        return cost, train_op

    # copy trained main network to target network
    def copy2target(self):
        copy_op = []

        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'main')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target')

        for main_var, target_var in zip(main_vars, target_vars):
            copy_op.append(target_var.assign(main_var.value()))

        self.sess.run(copy_op)

    # get action from Q network
    def get_action(self):
        value = self.sess.run(self.main_Q, feed_dict={self.X: self.state})
        action = np.argmax(value[0])
        return action

    # save current situation
    def save_memory(self, action, reward, done, new_state):
        new_state = np.reshape(new_state, (1, self.height, self.width, self.channel))
        new_state = np.append(self.state[1:, :, :, :], new_state, axis=0)

        # save  s, a, r, d, s`
        self.buffer.append((self.state, action, reward, done, new_state))

        if len(self.buffer) > self.BUFFER_SIZE:
            self.buffer.popleft()

        self.state = new_state

    def init_state(self, state):
        self.state = np.array([state for _ in range(self.STATE_SIZE)])

    # get random samples from buffer
    def _get_samples(self):
        memories = sample(self.buffer, self.batch_size)

        s, a, r, d, new_s = [[memory[i] for memory in memories] for i in range(5)]
        return s, a, r, d, new_s

    # train
    def train(self):
        state, action, reward, done, new_state = self._get_samples()

        target_Q_value = self.sess.run(self.target_Q, feed_dict={self.X: new_state})

        Y = []
        for i in range(self.batch_size):
            if done[i]:
                Y.append(reward[i])
            else:
                Y.append(reward + self.GAMMA * np.max(target_Q_value[i]))

        return self.sess.run(self.train_op, self.cost,
                             feed_dict={
                                 self.X: state,
                                 self.A: action,
                                 self.Y: Y
                             })
