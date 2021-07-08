import keras
import pandas as pd
import tensorflow as tf
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import PySimpleGUI as sg
import numpy as np

echos = []
status = []

# Layer
input_layer_nodes = 128
hidden_layer_1_nodes = 512
hidden_layer_2_nodes = 512
output_layer_nodes = 4
classes = 4
batch_size = 128

class_names = ['normal', 'zu niedrig', 'high peak', 'HW Defekt']
col_list = ['mac', 'slocTime', 'status', 'echo']

def file_input():
    Tk().withdraw()
    file = askopenfilename()
    df = pd.read_csv(file, names=col_list, sep=';')

    for stat in df['status']:
        if (stat != 'status'):
            status.append(float(stat))

    for echo in df['echo']:
        if (echo != 'echo'):
            echos.append(string2float(echo))

def neuronales_netz_testdaten_implementierne():
    file_input()
    model.fit(echos, status, epochs=25)
    test_loss, test_acc = model.evaluate(echos, status, verbose=2)
    print('Test accuracy:', test_acc)

@tf.function(input_signature=[tf.TensorSpec(shape=(1, input_layer_nodes))])
def predict(x):
    return model(x)

def string2float(str):
    floatarray = []
    strarray = str.split(',')
    float_array_2d = []
    for s in strarray:
        floatarray.append(float(s))
    float_array_2d.append(floatarray)
    return float_array_2d

def demo_gui():
    layout = [[sg.InputText(), sg.Text("Echo")],
              [sg.Button("OK")],
              [sg.Button("EXIT")]]
    window = sg.Window("Demo", layout)

    while True:
        event, values = window.read()
        if event == "OK":
            p = predict(string2float(values[0]))
            sg.popup('Ist: ', class_names[np.argmax(p)])
        if event == "EXIT" or event == sg.WIN_CLOSED:
            break
    window.close()

if __name__ == "__main__":
    model = keras.Sequential([tf.compat.v1.keras.layers.Flatten(input_shape=(0, input_layer_nodes)),
                              tf.compat.v1.keras.layers.Dense(hidden_layer_1_nodes, activation=tf.nn.relu),
                              tf.compat.v1.keras.layers.Dense(hidden_layer_2_nodes, activation=tf.nn.relu),
                              tf.compat.v1.keras.layers.Dense(output_layer_nodes, activation=tf.nn.softmax),
                              ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    neuronales_netz_testdaten_implementierne()
    demo_gui()
