import pandas as pd
import tensorflow as tf
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
import matplotlib.pyplot as plt
from PIL.ImagePalette import sepia

#tf.compat.v1.disable_eager_execution()

echos = []
status = []

#Layer
input_layer_nodes = 128
hidden_layer_1_nodes = 512
hidden_layer_2_nodes = 512
output_layer_nodes = 128

# Neuronen in den Layern
classes = 4
batch_size = 32
class_names = ['normal', 'high peak', 'zu niedrig', 'HW Defekt']

col_list = ['mac', 'slocTime', 'status', 'echo']

def file_input():
    Tk().withdraw()
    file = askopenfilename()
    df = pd.read_csv(file, names=col_list, sep=';')

    help2 = []
    for stat in df['status']:
        if(stat != 'status'):
            status.append(float(stat))

    for echo in df['echo']:
        if(echo != 'echo'):
            help = echo.split(',')
            for h in help:
                help2.append(float(h))
            echos.append(help2)
            help2 = []

def neuronales_netz_model(X):
    input_layer = {'weights': tf.Variable(tf.compat.v1.random_normal([input_layer_nodes, hidden_layer_1_nodes])),
                   'biases': tf.Variable(tf.compat.v1.random_normal([hidden_layer_1_nodes]))}

    hidden_layer_1 = {'weights': tf.Variable(tf.compat.v1.random_normal([hidden_layer_1_nodes, hidden_layer_2_nodes])),
                   'biases': tf.Variable(tf.compat.v1.random_normal([hidden_layer_2_nodes]))}

    hidden_layer_2 = {'weights': tf.Variable(tf.compat.v1.random_normal([hidden_layer_2_nodes, output_layer_nodes])),
                   'biases': tf.Variable(tf.compat.v1.random_normal([output_layer_nodes]))}

    output_layer = {'weights': tf.Variable(tf.compat.v1.random_normal([output_layer_nodes, classes])),
                   'biases': tf.Variable(tf.compat.v1.random_normal([classes]))}

    input_layer_sum = tf.add(tf.matmul(X, input_layer['weights']),input_layer['biases'])
    input_layer_sum = tf.nn.relu(input_layer_sum)

    hidden_layer_1_sum = tf.add(tf.matmul(input_layer_sum, hidden_layer_1['weights']),hidden_layer_1['biases'])
    hidden_layer_1_sum = tf.nn.relu(hidden_layer_1_sum)

    hidden_layer_2_sum = tf.add(tf.matmul(hidden_layer_1_sum, hidden_layer_2['weights']),hidden_layer_2['biases'])
    hidden_layer_2_sum = tf.nn.relu(hidden_layer_2_sum)

    outpu_layer_sum = tf.add(tf.matmul(hidden_layer_2_sum, output_layer['weights']), output_layer['biases'])

    return outpu_layer_sum

def loss(model_output, wirklicher_wert):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return loss_object(y_true=wirklicher_wert, y_pred=model_output)

def grad(auswertung, status):
  with tf.GradientTape() as tape:
    loss_value = loss(auswertung,status)
  return loss_value, tape.gradient(loss_value, auswertung)

def neuronales_netz_train(X):
    file_input()
    auswertung = neuronales_netz_model(X)
    print(auswertung)
    l = loss(auswertung,status)
    print('Loss test ', l)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    loss_value, grads = grad(auswertung, status)

    print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                              loss_value.numpy()))

    optimizer.apply_gradients(zip(grads, auswertung))

    print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
                                              loss(auswertung,status).numpy()))

neuronales_netz_train(echos)
