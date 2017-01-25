


import pandas as pd
from .base import *
import matplotlib.pyplot as plt
import numpy as np
DATA_PATH='Datasets/Data/'
FILENAME='STABLE_GJOA.csv'

SENSORS=['PWH','PBH','PDC','QGAS','CHK']
GJOA_QGAS_COL='GJOA_SEP_1_QGAS_'
data_type = 'mea'

X_COLS=['CHK','PWH']
Y_Q_COLS=['QGAS']

X_COLS_MULTI=[('CHK','QGAS')]



MULTI_INPUT=True





def fetch_gjoa_data():
    data=pd.read_csv(DATA_PATH+FILENAME)


    WELL_F1=data[generate_well_headers('F1',data_type)]
    WELL_B2=data[generate_well_headers('B2',data_type)]
    WELL_D3=data[generate_well_headers('D3',data_type)]
    WELL_E1=data[generate_well_headers('E1',data_type)]

    GJOA_SEP_1=data[GJOA_QGAS_COL+data_type].to_frame('GJOA_QGAS_mea') #Rename col name

    data={'WELL_F1':WELL_F1,'WELL_B2':WELL_B2,'WELL_D3':WELL_D3,'WELL_E1':WELL_E1,'GJOA_SEP_1':GJOA_SEP_1}

    X,Y,Y_Q=data_to_X_Y(data)

    #print_rank(X,'GJOA')




    GjoaData=DataContainer(X,Y,Y_Q,name='GJOA')

    #plt.plot(GjoaData.X['D3_PWH'])
    #plt.show()
    #exit()
    #print(data['WELL_F1'])
    #plot_scatter(data['WELL_E1'],'E1_CHK_mea','GJOA_QGAS',GJOA_SEP_1)
    #visualizeData(GjoaData.X_transformed)
    return GjoaData

def generate_well_headers(name,type='mea'):

    headers=[]
    for key in SENSORS:
        headers.append(name+'_'+key+'_'+type)

    return headers


def plot_scatter(data,x_tag,y_tag,data_tot=None):
    plt.figure()

    if len(data_tot)>1:
        plt.scatter(data[x_tag],data_tot)
    else:
        plt.scatter(data[x_tag],data[y_tag])
    plt.xlabel(x_tag)
    plt.ylabel(y_tag)
    plt.show()
def visualizeData(X):
    COL='CHK'
    i=1

    for col in X.columns:
        if col.split('_')[1]==COL:
            plt.subplot(2,3,i)
            i+=1
            plt.plot(X[col])
            plt.title(col)
    plt.show()

def data_to_X_Y(data):
    Y_Q=pd.DataFrame()
    X=pd.DataFrame()
    Y=pd.DataFrame()
    for key in data:
        for sensor in data[key].columns:
            sensor_splitted = sensor.split('_')
            for col in X_COLS:
                if sensor_splitted[1]==col and sensor_splitted[0]!='GJOA':
                    X[sensor_splitted[0]+'_'+sensor_splitted[1]]=data[key][sensor]
                    if col=='CHK':
                        X[X<0]=0
            for col in Y_Q_COLS:
                if sensor_splitted[1]==col and sensor_splitted[0]!='GJOA':
                    Y_Q[sensor_splitted[0]+'_'+sensor_splitted[1]]=data[key][sensor]
    Y['GJOA_QGAS']=data['GJOA_SEP_1']
    return X,Y,Y_Q


