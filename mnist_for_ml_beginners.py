import tensorflow as tf
import input_data
import time

# tensorflowではノードとエッジによって構成された計算グラフによって計算を進める。

# 開始時刻
start_time = time.time() # unixタイム
print("開始時刻: " + str(start_time))

# MNISTのデータをダウンロードしてローカルへ
print("--- MNISTデータの読み込み開始 ---")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("--- MNISTデータの読み込み完了 ---")

# セッションを作成する。
session = tf.Session()

# 訓練データ
x = tf.placeholder(tf.float32, [None, 784], name="x-input")

# 重み
w = tf.Variable(tf.zeros([784, 10]), name="weights")

# バイアス
b = tf.Variable(tf.zeros([10]), name="bias")

# name scope を使ってグラフ・ビジュアライザにおけるノードをまとめる。
with tf.name_scope("Wx_b") as scope:
    # 出力層でソフトマックス回帰を実行
    y = tf.nn.softmax(tf.matmul(x, w) + b)

# データ収集のための要約 OP を追加します。
w_hist = tf.summary.histogram("weights", w)
b_hist = tf.summary.histogram("biases", b)
y_hist = tf.summary.histogram("y", y)

# 正解ラベル
label = tf.placeholder(tf.float32, [None, 10], name="label-input")

with tf.name_scope('xent') as scope:
    # 交差エントロピー
    cross_entropy = -tf.reduce_sum(label*tf.log(y))
    ce_summ = tf.summary.scalar("cross entropy", cross_entropy)

with tf.name_scope('train') as scope:
    # 最急降下法を使う
    train_gradient_decent = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 全ての要約をマージしてそれらを /tmp/mnist_logs に書き出します。
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("/tmp/mnist_logs", session.graph_def)

# 変数の初期化
init_op = tf.global_variables_initializer()

# 変数の初期化セッションを実行する
session.run(init_op)

with tf.name_scope("test") as scope:
    # 予測
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(label,1))
    # 精度の計算
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy_summary = tf.summary.scalar("accuracy", accuracy)

# ランダムミニバッチ学習を行う
print("--- 訓練開始 ---")
for epoch in range(1000):
    if epoch % 10 == 0:  # 要約データと正解率を記録します。
        result = session.run([merged, accuracy], feed_dict={x: mnist.test.images, label: mnist.test.labels})
        summary_str = result[0]
        acc = result[1]
        writer.add_summary(summary_str, epoch)
        print("Accuracy at step %s: %s" % (epoch, acc))
    else:
        batch_xs, batch_ys = mnist.train.next_batch(100)
        session.run(train_gradient_decent, feed_dict={x: batch_xs, label: batch_ys})
print("--- 訓練完了 ---")

# 終了時刻
end_time = time.time()
print("終了時刻： " + str(end_time))
print("かかった時間: " + str(end_time - start_time))