


import pandas as pd
from .base import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
DATA_PATH='Datasets/Data/'
FILENAME='STABLE_GJOA_OIL_NEW.csv'

X_tags=['CHK','PWH','PBH','PDC']
X_GJOA_tags=['RISER_OIL_B_PDC','RISER_OIL_B_CHK','RISER_OIL_A_PDC']
Y_GJOA_tags=['TOTAL_QOIL','TOTAL_QGAS_DEPRECATED','TOTAL_QWAT','SEP_1_QOIL','SEP_1_QWAT']
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


def plot_values(X,Y):
    sp_y = 2
    sp_x = 4
    i = 1
    for well in well_names:
        name = well + '_CHK'
        plt.subplot(sp_y, sp_x, i)
        i += 1
        plt.plot(X[name])  # ,Y['GJOA_TOTAL_QOIL'])
        plt.title(name)
    # plt.scatter(X['GJOA_RISER_OIL_A_CHK'], Y['GJOA_TOTAL_QOIL'])
    plt.figure()
    plt.plot(Y['GJOA_SEP_1_QWAT'])
    plt.show()

well_names=['C1','C2','C3','C4','D1','B3','B1']
def fetch_gjoa_data():
    data=pd.read_csv(DATA_PATH+FILENAME)

    X,Y=data_to_X_Y(data,'mea')
    sum_oil = np.zeros((len(Y), 1))  # Y['C1_QOIL']

    tags=[]
    for key in well_names:
        name = key + '_' + 'QOIL'
        tags.append(name)
        # if key!='C1':
        #sum_oil += Y[name]
    sum_oil=Y[tags].sum(axis=1)
    Y['GJOA_TOTAL_QOIL_SUM']=sum_oil

    X['time'] = np.arange(0, len(X.index))

    #

    tags=['GJOA_TOTAL_QOIL','GJOA_TOTAL_QWAT']

    #plt.plot(Y['GJOA_TOTAL_QOIL_SUM'],color='red')
    #plt.plot(Y['GJOA_TOTAL_QOIL'],color='blue')
    #plt.show()
    #plot_together(X,Y,tags)

    #plot_values(X,Y)
    '''''
    print(X.columns)
    sum_oil=np.zeros((len(Y),1))#Y['C1_QOIL']
    #print(sum_oil.shape)
    #print(Y['C1_QOIL'].shape)
    i=1
    for key in well_names:
        name=key+'_'+'QOIL'
        plt.subplot(2,4,i)
        plt.plot(Y[name])
        plt.title(name)
        i+=1
        #if key!='C1':
        sum_oil+=Y[name].reshape((len(Y),1))
    plt.subplot(2,4,i)
    print(sum_oil.shape)
    YDIFF=Y['GJOA_TOTAL_QOIL'].reshape((len(Y),1))-sum_oil
    plt.plot(sum_oil,color='red')
    plt.plot(Y['GJOA_TOTAL_QOIL'].reshape((len(Y),1))-YDIFF,color='blue')
    #plt.plot(Y['GJOA_SEP_1_QOIL'],color='green')
    plt.show()
    '''''





    GjoaData=DataContainer(X,Y,Y,name='GJOA2',Y_SCALE=100)

    #plot_test(GjoaData.X_transformed, GjoaData.Y_transformed)

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
            print(col)
            X[col]=data[col+'_'+type]
            if tag=='CHK':
                X[col][X[col]<0]=0
        for tag in Y_tags:
            col=name+'_'+tag
            Y[col]=data[col+'_'+type]
    for tag in X_GJOA_tags:
        col = 'GJOA' + '_' + tag
        X[col] = data[col + '_' + type]
    for tag in Y_GJOA_tags:
        col = 'GJOA' + '_' + tag
        Y[col] = data[col + '_' + type]
    #Y['GJOA_SEP_1_QOIL'][Y['GJOA_SEP_1_QOIL']>0]=0


    return X,Y


def plot_together(X,Y,tags):
    i=1
    for tag in tags:
        plt.subplot(2, 1, i)
        i+=1
        plt.scatter(np.arange(0, len(Y)), Y[tag],color='black')
        plt.xlabel('Time')
        plt.ylabel(tag)
        plt.title(tag)
    plt.show()
def plot_test(X,Y):
    # _,(ax1,ax2,ax3,ax4,ax5)=plt.subplots(8,1,sharex=True)
    _, axes = plt.subplots(8, 1, sharex=True)

    # cols=['CHK']
    #plt.figure()
    for ax, tag in zip(axes, well_names):
        name = tag + '_' + 'PBH'
        #plt.plot(X[name])
        ax.plot(X[name])
        ax.set_title(name)
    axes[-1].plot(Y['GJOA_TOTAL_QOIL'])
    axes[-1].set_title('GJOA_TOTAL_QOIL')
    plt.figure()
    plt.plot(Y['GJOA_TOTAL_QGAS_DEPRECATED'])
    plt.show()
