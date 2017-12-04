import tensorflow as tf
import input_data
import time

# このコードは重みの初期値が0.01なので、勾配消失の問題がある

# 開始時刻
start_time = time.time() # unixタイム
print("開始時刻: " + str(start_time))

# MNISTのデータをダウンロードしてローカルへ
print("--- MNISTデータの読み込み開始 ---")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("--- MNISTデータの読み込み完了 ---")

sess = tf.InteractiveSession()


def weight(shape):
    initial = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias(shape):
    initial = tf.zeros(shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

x = tf.placeholder(tf.float32, shape=[None, 784])
label = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])

# 畳み込み層1
# [縦、横、チャンネル数、フィルター数]
w_conv1 = weight([5, 5, 1, 32])

# フィルター数と対応するバイアス
b_conv1 = bias([32])

# 畳み込み層
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)

# プーリング層
h_pool1 = max_pool_2x2(h_conv1)

# 畳み込み層2
w_conv2 = weight([5, 5, 32, 64])
b_conv2 = bias([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 全結合層
w_fc1 = weight([7 * 7 * 64, 1024])
b_fc1 = bias([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# ドロップアウト
# keep_probはドロップアウトさせない率
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop =tf.nn.dropout(h_fc1, keep_prob)

# 出力層
w_fc2 = weight([1024, 10])
b_fc2 = bias([10])
y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
y = tf.nn.softmax(y_conv)
cross_entropy = -tf.reduce_sum(label*tf.log(y))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(label,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

for i in range(1000):
  batch = mnist.train.next_batch(50)
  if i % 100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], label: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], label: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, label: mnist.test.labels, keep_prob: 1.0}))

# 終了時刻
end_time = time.time()
print("終了時刻： " + str(end_time))
print("かかった時間: " + str(end_time - start_time))
