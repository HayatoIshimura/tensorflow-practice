import tensorflow as tf
import input_data
import time
from datetime import datetime

# tensorflowではノードとエッジによって構成された計算グラフによって計算を進める。

# 開始時刻
start_time = time.time() # unixタイム
print("開始時刻: " + str(start_time))

# MNISTのデータをダウンロードしてローカルへ
print("--- MNISTデータの読み込み開始 ---")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("--- MNISTデータの読み込み完了 ---")


def affine(layer_data, next_layer_size, layer_number):
    w = tf.Variable(tf.random_normal([layer_data.shape.as_list()[1], next_layer_size], mean=0.0, stddev=0.01))
    b = tf.Variable(tf.zeros(next_layer_size))
    z = tf.matmul(layer_data, w) + b

    tf.summary.histogram("weight:" + str(layer_number), w)
    tf.summary.histogram("bias:" + str(layer_number), b)
    return z

# InputLayer
y_width = 10
x = tf.placeholder(tf.float32, [None, 784])

layer_input = x
hidden_layer_size = 5

for layer in range(1, hidden_layer_size):
    next_layer_width = round((layer_input.shape.as_list()[1] + y_width) * 2 / 3)

    with tf.name_scope("Wx_b") as scope:
        z = affine(layer_input, next_layer_width, layer)

    with tf.name_scope("activation") as scope:
        a = tf.nn.relu(z)
        a_hist = tf.summary.histogram("activation:" + str(layer_input), a)
    layer_input = a

with tf.name_scope("Wx_b") as scope:
    z = affine(layer_input, y_width, hidden_layer_size)

# name scope を使ってグラフ・ビジュアライザにおけるノードをまとめる。
with tf.name_scope("output_layer") as scope:
    # 出力層でソフトマックス回帰を実行
    y = tf.nn.softmax(z)


# 正解ラベル
label = tf.placeholder(tf.float32, [None, 10], name="label-input")

with tf.name_scope('xent') as scope:
    # 交差エントロピー
    cross_entropy = -tf.reduce_sum(label*tf.log(y))
    tf.summary.scalar("cross entropy", cross_entropy)

with tf.name_scope('train') as scope:
    # 最急降下法を使う
    train_gradient_decent = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

with tf.name_scope("test") as scope:
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(label,1))
    # 精度の計算
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar("accuracy", accuracy)

# セッションを作成する。
session = tf.Session()

# 全ての要約をマージしてそれらを /tmp/mnist_logs に書き出します。
time_string = datetime.now().strftime('%Y%m%d/%H%M%S')

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("mnist_logs/" + time_string, session.graph_def)

# 変数の初期化
init_op = tf.global_variables_initializer()

# 変数の初期化セッションを実行する
session.run(init_op)

# ランダムミニバッチ学習を行う
print("--- 訓練開始 ---")
for epoch in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)

    if epoch % 10 == 0:
        result = session.run([merged, accuracy], feed_dict={x: mnist.test.images, label: mnist.test.labels})
        summary_str = result[0]
        acc = result[1]
        writer.add_summary(summary_str, epoch)
        print("Accuracy at epoch %s: %s" % (epoch, acc))
    else:
        result = session.run(train_gradient_decent, feed_dict={x: batch_xs, label: batch_ys})
print("--- 訓練完了 ---")

# 終了時刻
end_time = time.time()
print("終了時刻： " + str(end_time))
print("かかった時間: " + str(end_time - start_time))
