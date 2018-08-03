# -*- coding: utf-8 -*-

"""
Desc:

Authors:  zhangjingshuai
Contact:  zhangjingshuai@baidu.com)
Date:     2018/8/2
Version:  0.5
License : Copyright(C), Baidu TIC
"""

import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def normal_eauation(X,Y):
    return tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(X, X, transpose_a=True)), X, transpose_b=True), Y)


if __name__ == '__main__':
    reset_graph()

    housing = fetch_california_housing()
    m, n = housing.data.shape
    print m, n
    housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
    X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

    theta = normal_eauation(X, y)
    with tf.Session() as sess:
        theta_value = theta.eval()
    print theta_value