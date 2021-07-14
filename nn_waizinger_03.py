import keras
import pandas as pd
import tensorflow as tf
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import PySimpleGUI as sg
import numpy as np

echos = []
fuellstand = []

# Layer
input_layer_nodes = 128
hidden_layer_1_nodes = 128
hidden_layer_2_nodes = 128
output_layer_nodes = 11
classes = 11
batch_size = 128

class_names = ['0%', '10%', '20%', '30% ', '40%', '50%', '60%', '70%', '80%', '90%', '100%']
col_list = ['mac', 'fill level', 'echo']

def file_input():
    Tk().withdraw()
    file = askopenfilename()
    df = pd.read_csv(file, names=col_list, sep=';')

    for lvl in df['fill level']:
        if (lvl != 'fill level'):
            fuellstand.append(int(lvl))

    for echo in df['echo']:
        if (echo != 'echo'):
            echos.append(string2int(echo))

def neuronales_netz_testdaten_implementierne():
    file_input()
    model.fit(echos, fuellstand, epochs=25)
    test_loss, test_acc = model.evaluate(echos, fuellstand, verbose=2)
    print('Test accuracy:', test_acc)

@tf.function(input_signature=[tf.TensorSpec(shape=(1, input_layer_nodes))])
def predict(x):
    return model(x)

def string2int(str):
    intarray = []
    strarray = str.split(',')
    int_array_2d = []
    for s in strarray:
        intarray.append(int(s))
    int_array_2d.append(intarray)
    return int_array_2d

def demo_gui():
    layout = [[sg.InputText(), sg.Text("Echo")],
              [sg.Button("OK")],
              [sg.Button("EXIT")]]
    window = sg.Window("Demo", layout)

    while True:
        event, values = window.read()
        if event == "OK":
            p = predict(string2int(values[0]))
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