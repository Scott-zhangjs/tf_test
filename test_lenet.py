# -*- coding: utf-8 -*-

"""
Desc:

Authors:  zhangjingshuai
Contact:  zhangjingshuai@baidu.com)
Date:     2018/8/3
Version:  0.5
License : Copyright(C), Baidu TIC
"""
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.contrib.layers import flatten

EPOCHS = 10
BATCH_SIZE = 128

def LeNet(x):
    mu = 0
    sigma = 0.1

    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_w = tf.Variable(tf.truncated_normal(shape=[5,5,1,6], mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1,1,1,1], padding='VALID') + conv1_b
    # activation
    conv1 = tf.nn.relu(conv1)
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    pool_1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides=[1,1,1,1], padding='VALID') + conv2_b
    # activation
    conv2 = tf.nn.relu(conv2)
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    pool_2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    fc1 = flatten(pool_2)
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape=[400, 120], mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1, fc1_w)+fc1_b
    # activation
    fc1 = tf.nn.relu(fc1)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape=[120, 84], mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_w) + fc2_b
    # activation
    fc2 = tf.nn.relu(fc2)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_w = tf.Variable(tf.truncated_normal(shape=[84, 10], mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    # activation
    return logits


def loss(logits, one_hot_y, rate=0.001):
    # 对输出先进性softmax 再 求交叉熵
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=rate)
    training_operation = optimizer.minimize(loss_operation)
    return training_operation


def evaluate(X_data, y_data, logits, one_hot_y):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    # cast 类型转换函数
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x:batch_x, y:batch_y})
        total_accuracy += accuracy*len(batch_x)
    return total_accuracy/num_examples


def train(X_train, y_train, X_validation, y_validation):
    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, 10)
    logits = LeNet(x)
    train_operation = loss(logits, one_hot_y)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)
        print("Training...")
        print()
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset+BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(train_operation, feed_dict={x: batch_x, y:batch_y})
            validation_accuracy = evaluate(X_validation, y_validation)
            print("EPOCH {} ...".format(i + 1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()
        saver.save(sess, 'lenet')
        print("Model saved")


def main():
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
    X_train, y_train = mnist.train.images, mnist.train.labels
    X_validation, y_validation = mnist.validation.images, mnist.validation.labels
    X_test, y_test = mnist.test.images, mnist.test.labels

    assert (len(X_train) == len(y_train))
    assert (len(X_validation) == len(y_validation))
    assert (len(X_test) == len(y_test))

    print()
    print("Image Shape: {}".format(X_train[0].shape))
    print()
    print("Training Set:   {} samples".format(len(X_train)))
    print("Validation Set: {} samples".format(len(X_validation)))
    print("Test Set:       {} samples".format(len(X_test)))
    train(X_train, y_train, X_validation, y_validation)


def test():
    input = tf.Variable(tf.random_normal([1, 3, 3, 1]))
    filter = tf.Variable(tf.random_normal([1, 1, 1, 2]))
    # input的shape: [batch, in_height, in_width, in_channels]
    # filter的shape: [filter_height, filter_width, in_channels, out_channels]

    op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
    init = tf.global_variables_initializer()
    a = tf.constant([-1, -2, -3])
    with tf.Session() as sess:
        init.run()
        # print input.eval()
        # print filter.eval()
        input_value, filter_value = sess.run([input, filter])
        # print input_value, filter_value
        # eval 一步只计算一个tensor中的值，run可以计算多个
        # tmp = op.eval()
        b = tf.nn.relu(a)
        print sess.run(b)

if __name__ == '__main__':

    main()
    # test()