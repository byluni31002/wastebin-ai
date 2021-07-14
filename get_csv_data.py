import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import operator

datas = []
data_col_list = ['MAC', 'ProfileId', 'ProfileName', 'SLOCTime', 'DataType', 'Value']

def data_file_input():
    Tk().withdraw()
    file = askopenfilename()
    df = pd.read_csv(file, names=data_col_list, sep='\t')

    for l in df['MAC']:
        line = l.replace('"', '').split(';')
        datas.append(line)
    datas.sort(key=operator.itemgetter(0))
    return 'Finished loading data'

def get_by_mac(mac):
    output = []
    for data in datas:
        if(data[0]==mac):
            output.append(data)
    output.sort(key=operator.itemgetter(3))
    return output

def get_macs_from_profilename(profilename):
    macs = []
    for data in datas:
        if(profilename in data[2]):
            macs.append(data[0])
    return macs

if __name__ == "__main__":
    data_file_input()
    mac = input()
    get_by_mac(mac)
