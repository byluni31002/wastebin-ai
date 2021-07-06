import tensorflow as tf
import pandas as pd
import csv
tf.compat.v1.disable_eager_execution()

values = []
results = []
lines = []
help = []

with open("input.txt", newline='') as file_in:
    for line in file_in:
        help = line.split(',')
        for h in help:
            if h.__contains__('\r\n'):
                h = h.replace('\r\n','')
        for h in help:
            h = float(h)
        values.append(help)
    print('val ',values)


#Layer
input_layer_nodes = 128
hidden_layer_1_nodes = 128
hidden_layer_2_nodes = 128
output_layer_nodes = 4

# Neuronen in den Layern
classes = 4
batch_size = 150

X = tf.compat.v1.placeholder('float', [None, input_layer_nodes])
Y = tf.compat.v1.placeholder('float')

def nn_model(X):
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

    hidden_layer_1_sum = tf.add(tf.matmul(input_layer, hidden_layer_1['weights']),hidden_layer_1['biases'])
    hidden_layer_1_sum = tf.nn.relu(hidden_layer_1_sum)

    hidden_layer_2_sum = tf.add(tf.matmul(hidden_layer_1, hidden_layer_2['weights']),hidden_layer_2['biases'])
    hidden_layer_2_sum = tf.nn.relu(hidden_layer_2_sum)

    outpu_layer_sum = tf.add(tf.matmul(hidden_layer_2), output_layer['weights'] + output_layer['biases'])

    return outpu_layer_sum

def nn_train(X):
    prediction = nn_model(X)
    cost = tf.compat.v1.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, Y))
    optimizer = tf.compat.v1.train.AdamOptimizer().minimize(cost)
    epochs = 10
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer)

#nn_train()


print(nn_model(values))