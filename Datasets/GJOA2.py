


import pandas as pd
from .base import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
DATA_PATH='Datasets/Data/'
FILENAME='STABLE_GJOA_OIL.csv'

X_tags=['CHK','PWH','PBH','PDC']
X_GJOA_tags=['RISER_OIL_B_PDC','RISER_OIL_B_CHK']
Y_GJOA_tags=['TOTAL_QOIL','TOTAL_QWAT']
Y_tags=['QOIL','QWAT','QGAS']
GJOA_QGAS_COL='GJOA_SEP_1_QGAS_'
data_type = 'mea'

X_COLS=['CHK','PWH','PBH','PDC','QGAS']
#X_COLS=['CHK']
#Y_Q_COLS=['QGAS']
#Y_COLS=['QGAS']
Y_COLS=['PBH','PWH','PDC','QGAS']
#Y_COLS=['PWH']

X_COLS_MULTI=[('CHK','QGAS')]



MULTI_INPUT=True




well_names=['C1','C2','C3','C4','D1','B3','B1']
def fetch_gjoa_data():
    data=pd.read_csv(DATA_PATH+FILENAME)

    X,Y=data_to_X_Y(data,'mea')





    GjoaData=DataContainer(X,Y,Y,name='GJOA2',Y_SCALE=100)

    print(X.columns)
    print(Y.columns)
    return GjoaData


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

def data_to_X_Y(data,type):
    X=pd.DataFrame()
    Y=pd.DataFrame()

    for name in well_names:
        for tag in X_tags:
            col=name+'_'+tag
            X[col]=data[col+'_'+type]
        for tag in Y_tags:
            col=name+'_'+tag
            Y[col]=data[col+'_'+type]
    for tag in X_GJOA_tags:
        col = 'GJOA' + '_' + tag
        X[col] = data[col + '_' + type]
    for tag in Y_GJOA_tags:
        col = 'GJOA' + '_' + tag
        Y[col] = data[col + '_' + type]


    return X,Y


