import tensorflow as tf
from tensorflow.contrib import rnn
import input_data
import time

# 開始時刻
start_time = time.time() # unixタイム
print("開始")

# MNISTのデータをダウンロードしてローカルへ
print("--- MNISTデータの読み込み開始 ---")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("--- MNISTデータの読み込み完了 ---")

n_input = 28
n_output = 10
n_steps = 28
n_hidden = 128

# [入力データ数、シーケンス数、特徴数]
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_output])

def RNN(x):

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

prediction = RNN(x)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
step = tf.train.AdagradOptimizer(0.001).minimize(cost)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# セッションを作成する。
session = tf.Session()

init_op = tf.global_variables_initializer()
session.run(init_op)

n_epoch = 10000
batch_size = 128

for epoch in range(n_epoch):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)

    # next_batchで返されるbatch_xは[batch_size, 784]のテンソルなので、batch_size×28×28に変換する。
    batch_xs = batch_xs.reshape((batch_size, 28, 28))
    session.run(step, feed_dict={x: batch_xs, y: batch_ys})

    if epoch % 100 == 0:
        acc = session.run(accuracy, feed_dict={x: batch_xs, y:batch_ys})
        loss = session.run(cost, feed_dict={x: batch_xs, y:batch_ys})
        print('epoch: {} / loss: {:.6f} / acc: {:.5f}'.format(epoch, loss, acc))

test_len = 128
test_data = mnist.test.images[:test_len].reshape((-1, 28, 28))
test_label = mnist.test.labels[:test_len]
test_acc = session.run(accuracy, feed_dict={x: test_data, y: test_label})
print("Test Accuracy: {}".format(test_acc))

# 終了時刻
end_time = time.time()
print("終了")
print("かかった時間: " + str(end_time - start_time))