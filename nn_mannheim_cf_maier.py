import keras
import pandas as pd
import tensorflow as tf
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
import get_csv_data

echos = []
fuellstand = []

# Layer
input_layer_nodes = 128
hidden_layer_1_nodes = 256
hidden_layer_2_nodes = 256
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
    model.fit(echos, fuellstand, epochs=75)
    test_loss, test_acc = model.evaluate(echos, fuellstand, verbose=2)
    print('Test accuracy:', test_acc)

@tf.function(input_signature=[tf.TensorSpec(shape=(1, input_layer_nodes))])
def predict(x):
    return model(x)

def string2int(str):
    intarray = []
    strarray = str.split(',')
    float_array_2d = []
    for s in strarray:
        intarray.append(int(s))
    float_array_2d.append(intarray)
    return float_array_2d

def get_dump(dump1,dump2):
    d1 = dump1.split(',')
    d2 = dump2.split(',')
    dump = []
    for d in d1:
        if(d != ''):
            dump.append(d)
    for d in d2:
        if (d != ''):
            dump.append(d)
    output = ''
    i = 0
    while(i<len(dump)):
        if(i==0):
            output = output + dump[i]
        else:
            output = output + str(', ' + dump[i])
        i += 1
    return output

if __name__ == "__main__":
    print('input database csv')
    print(get_csv_data.data_file_input())
    model = keras.Sequential([tf.compat.v1.keras.layers.Flatten(input_shape=(0, input_layer_nodes)),
                              tf.compat.v1.keras.layers.Dense(hidden_layer_1_nodes, activation=tf.nn.relu),
                              tf.compat.v1.keras.layers.Dense(hidden_layer_2_nodes, activation=tf.nn.relu),
                              tf.compat.v1.keras.layers.Dense(output_layer_nodes, activation=tf.nn.softmax),
                              ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print('input train data for ai')
    neuronales_netz_testdaten_implementierne()
    print('1: input mac address - 2: input ptofilename')
    inp = input()
    if(inp == '1'):
        print('enter mac:')
        inp = input()
        dump1 = ''
        dump2 = ''
        for data in get_csv_data.get_by_mac(inp):
            if (data[4] == 'Fill Level'):
                print('Given fill level: ', data[5])
                dump2 = ''
            elif (data[4] == 'Echo Dump Part 1'):
                dump1 = data[5]
            elif (data[4] == 'Echo Dump Part 2'):
                dump2 = data[5]
            else:
                dump1 = ''
                dump2 = ''

            if (dump1 != '' and dump2 != ''):
                dump = get_dump(dump1, dump2)
                p = predict(string2int(dump))
                print('Echo dump: ', dump)
                print('predicted fill level: ', class_names[np.argmax(p)], '\n')
        print('\n')
        pass
    elif(inp == '2'):
        print('enter profilename:')
        inp = input()
        macs = get_csv_data.get_macs_from_profilename(inp)
        for mac in macs:
            print('mac: ', mac)
            dump1 = ''
            dump2 = ''
            for data in get_csv_data.get_by_mac(mac):
                if (data[4] == 'Fill Level'):
                    print('Given fill level: ', data[5])
                    dump2 = ''
                elif (data[4] == 'Echo Dump Part 1'):
                    dump1 = data[5]
                elif (data[4] == 'Echo Dump Part 2'):
                    dump2 = data[5]
                else:
                    dump1 = ''
                    dump2 = ''

                if (dump1 != '' and dump2 != ''):
                    dump = get_dump(dump1, dump2)
                    p = predict(string2int(dump))
                    print('Echo dump: ', dump)
                    print('predicted fill level: ', class_names[np.argmax(p)], '\n')
        print('\n')