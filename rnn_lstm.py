import tensorflow as tf
from tensorflow.contrib import rnn

n_input = 28
n_output = 10
n_steps = 28
n_hidden = 128
batch_size = 50

def RNN(x):
    # [入力データ数、シーケンス数、特徴数]
    x = tf.placeholder(tf.float32, [None, n_steps, n_input])

    ## 時系列データをTensorFlowのRNNで利用できる形式に変換
    # 1. [シーケンス数、入力データ数、特徴数]に転置
    # x = tf.transpose(self.x, [1, 0, 2])

    # 2. [入力データ数 x シーケンス数、特徴数]にreshape。シーケンスを縦につなげたイメージ
    # x = tf.reshape(x, [-1, n_input])

    # 3. [入力データ、特徴数]のtensorをシーケンス個に分割する。
    # x = tf.split(x, n_steps, 0)

    # 1, 2, 3をすべてやってくれるAPI
    x = tf.unstack(x, n_steps, 1)

    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    weight = tf.Variable(tf.random_normal([n_hidden, n_output]))
    bias = tf.Variable(tf.random_normal([n_output]))

    return tf.matmul(outputs[-1], weight) + bias