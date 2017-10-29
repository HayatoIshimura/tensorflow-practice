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

# 訓練データ
x = tf.placeholder(tf.float32, [None, 784])

# 重み
w = tf.Variable(tf.zeros([784, 10]))

# バイアス
b = tf.Variable(tf.zeros([10]))

# 出力層でソフトマックス回帰を実行
y = tf.nn.softmax(tf.matmul(x, w) + b)

# 正解ラベル
label = tf.placeholder(tf.float32, [None, 10])

# 交差エントロピー
cross_entropy = -tf.reduce_sum(label*tf.log(y))

# 最急降下法を使う
train_gradient_decent = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 変数の初期化
init_op = tf.global_variables_initializer()

# セッションを作成する
session = tf.Session()

# 変数の初期化セッションを実行する
session.run(init_op)

# ランダムミニバッチ学習を行う
# 1000エポック
print("--- 訓練開始 ---")
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    session.run(train_gradient_decent, feed_dict={x: batch_xs, label: batch_ys})
print("--- 訓練完了 ---")

# 予測
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(label,1))

# 精度の計算
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 精度の実行と表示
print("精度")
print(session.run(accuracy, feed_dict={x: mnist.test.images, label: mnist.test.labels}))

#　終了時刻
end_time = time.time()
print("終了時刻： " + str(end_time))
print("かかった時間: " + str(end_time - start_time))