""" Neural Network.

A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).

This example is using TensorFlow layers, see 'neural_network_raw' example for
a raw implementation with variables.

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/

步骤：
1、自定义neural_net神经网络
2、定义模型函数model_fn：
   a、获取输出层结果
   b、求 pred_classes = tf.argmax(logits, axis=1)
         pred_probas = tf.nn.softmax(logits)
   c、返回估算器对象实例（包括损失函数、优化方式、准确率函数）
      tf.estimator.EstimatorSpec(mode=mode,
                                 predictions=pred_classes,
                                 loss=loss_op,
                                 train_op=train_op,
                                 eval_metric_ops={'accuracy': acc_op})
3、构建估算器 model = tf.estimator.Estimator(model_fn)
4、定义训练用的输入函数  input_fn = tf.estimator.inputs.numpy_input_fn
5、训练  model.train(input_fn, steps=num_steps)
6、评估  model.evaluate(input_fn)
"""

from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

import tensorflow as tf

# Parameters
learning_rate = 0.1
num_steps = 1000
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)


# Define the neural network
def neural_net(x_dict):
    # TF Estimator input is a dict, in case of multiple inputs
    x = x_dict['images']
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.layers.dense(x, n_hidden_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.layers.dense(layer_2, num_classes)
    return out_layer


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    logits = neural_net(features)

    # Predictions
    pred_classes = tf.argmax(logits, axis=1) # 根据axis取值的不同返回每行或者每列最大值的索引
    pred_probas = tf.nn.softmax(logits)      # 输出logits中最大概率的值

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Define loss and optimizer
    # tf.nn.sparse_softmax_cross_entropy_with_logits函数是将softmax和cross_entropy放在一起计算，
    # 对于分类问题而言，最后一般都是一个单层全连接神经网络，
    # 比如softmax分类器居多，对这个函数而言，tensorflow神经网络中是没有softmax层，而是在这个函数中
    # 进行softmax函数的计算。
    # 这里的logits通常是最后的全连接层的输出结果，labels是具体哪一类的标签，
    # 这个函数是直接使用标签数据的，而不是采用one-hot编码形式。
    # tf.cast 转换数据类型
    # 计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes) # 计算模型输出的准确率

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs

# Build the Estimator
model = tf.estimator.Estimator(model_fn)

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images}, y=mnist.train.labels,
    batch_size=batch_size, num_epochs=None, shuffle=True)
# Train the Model
model.train(input_fn, steps=num_steps)

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images}, y=mnist.test.labels,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])
