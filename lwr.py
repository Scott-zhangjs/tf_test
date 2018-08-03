# -*- coding: utf-8 -*-

# python 3.5.3  @蔡军生
# http://edu.csdn.net/course/detail/2592
#  计算加权回归
# locally weighted linear regression

import numpy as np
import random
import matplotlib.pyplot as plt


def gaussian_kernel(x, x0, c, a=1.0):
    """
    Gaussian kernel.
    :Parameters:
      - `x`: nearby datapoint we are looking at.
      - `x0`: data point we are trying to estimate.
      - `c`, `a`: kernel parameters.
    """
    # Euclidian distance
    diff = x - x0
    dot_product = diff * diff.T
    return a * np.exp(dot_product / (-2.0 * c ** 2))


def get_weights(training_inputs, datapoint, c=1.0):
    """
    Function that calculates weight matrix for a given data point and training
    data.
    :Parameters:
      - `training_inputs`: training data set the weights should be assigned to.
      - `datapoint`: data point we are trying to predict.
      - `c`: kernel function parameter
    :Returns:
      NxN weight matrix, there N is the size of the `training_inputs`.
    """
    x = np.mat(training_inputs)
    n_rows = x.shape[0]
    # Create diagonal weight matrix from identity matrix
    weights = np.mat(np.eye(n_rows))
    for i in range(n_rows):
        weights[i, i] = gaussian_kernel(datapoint, x[i], c)

    return weights


def lwr_predict(training_inputs, training_outputs, datapoint, c=1.0):
    """
    Predict a data point by fitting local regression.
    :Parameters:
      - `training_inputs`: training input data.
      - `training_outputs`: training outputs.
      - `datapoint`: data point we want to predict.
      - `c`: kernel parameter.
    :Returns:
      Estimated value at `datapoint`.
    """
    weights = get_weights(training_inputs, datapoint, c=c)

    x = np.mat(training_inputs)
    y = np.mat(training_outputs).T

    xt = x.T * (weights * x)
    betas = xt.I * (x.T * (weights * y))

    return datapoint * betas


def genData(numPoints, bias, variance):
    x = np.zeros(shape=(numPoints, 2))
    y = np.zeros(shape=numPoints)
    # 构造一条直线左右的点
    for i in range(0, numPoints):
        # 偏移
        x[i][0] = 1
        x[i][1] = i
        # 目标值
        y[i] = bias + i * variance + random.uniform(0, 1) * 20
    return x, y


if __name__ == '__main__':

    # 生成数据
    a1, a2 = genData(100, 10, 0.6)

    a3 = []
    # 计算每一点
    for i in a1:
        pdf = lwr_predict(a1, a2, i, 1)
        a3.append(pdf.tolist()[0])

    plt.plot(a1[:, 1], a2, "x")
    plt.plot(a1[:, 1], a3, "r-")
    plt.show()
