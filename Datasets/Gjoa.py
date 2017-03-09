


import pandas as pd
from .base import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
DATA_PATH='Datasets/Data/'
FILENAME='STABLE_GJOA.csv'

SENSORS=['CHK','PWH','PBH','PDC','QGAS']
GJOA_QGAS_COL='GJOA_SEP_1_QGAS_'
DATA_TYPE = 'mea'

X_tags=['CHK','PWH','PBH','PDC','QGAS']
Y_tags=['PBH','PWH','PDC','QGAS']



X_COLS_MULTI=[('CHK','QGAS')]



well_names=['F1','B2','D3','E1']
MULTI_INPUT=True


def test_bed(X,Y):
    tags = []
    for key in well_names:
        name = key + '_' + 'QGAS'
        tags.append(name)

    sum_gas = Y[tags].sum(axis=1)
    plt.scatter(X['time'],sum_gas-Y['GJOA_QGAS'], color='blue')
    #plt.plot(sum_gas,color='red')
    plt.show()


def fetch_gjoa_data():
    data=pd.read_csv(DATA_PATH+FILENAME)

    X,Y=data_to_X_Y(data)
    Y=add_diff(Y)

    #Y=addModify(X,Y,'E1_PWH')
    #X = addModify(Y, X, 'E1_PDC')
    X['time']=np.arange(0,len(X))

    #test_bed(X,Y)

    print('MAX: {}, MEAN: {}'.format(np.max(Y['GJOA_QGAS']),np.mean(Y['GJOA_QGAS'])))
    print('Data size: {}'.format(len(Y)))

    #test_bed(X, Y)
    for key in well_names:
        ind_zero = X[key + '_CHK'] < CHK_THRESHOLD

        X = set_index_values_to_zero(X, ind_zero, key + '_PWH')
        X = set_index_values_to_zero(X, ind_zero, key + '_PBH')
        X = set_index_values_to_zero(X, ind_zero, key + '_PDC')

        Y = set_index_values_to_zero(Y, ind_zero, key + '_PWH')
        Y = set_index_values_to_zero(Y, ind_zero, key + '_PBH')
        Y = set_index_values_to_zero(Y, ind_zero, key + '_PDC')

        X[key + '_CHK_zero'] = np.array([0 if x < CHK_THRESHOLD else 1 for x in X[key + '_CHK']])

    GjoaData=DataContainer(X,Y,name='GJOA')

    return GjoaData


def data_to_X_Y(data):
    X=pd.DataFrame()
    Y=pd.DataFrame()

    X['X'] = np.zeros((len(data),))
    Y['Y'] = np.zeros((len(data),))

    for name in well_names:

        for tag in X_tags:
            tag_name=name+'_'+tag
            X[tag_name]=data[tag_name+'_'+DATA_TYPE]

            if tag=='CHK':
                X=negative_values_to_zero(X, tag_name)

        for tag in Y_tags:
            col=name+'_'+tag
            Y[col]=data[col+'_'+DATA_TYPE]

    Y['GJOA_QGAS'] = data['GJOA_SEP_1_QGAS'+'_'+DATA_TYPE]

    return X,Y

def add_diff(X):
    for key in well_names:
        PWH_tag=key+'_'+'PWH'
        PDC_tag=key+'_'+'PDC'

        X[key+'_deltap']=X[PWH_tag]-X[PDC_tag]
    return X

def addModify(X,Y,type):

    t = np.arange(0, len(X.index))
    t = t.reshape((len(t), 1))
    YPWH = Y[type]
    YPWH = YPWH.reshape((len(YPWH), 1))


    a = 5
    b = 2

    model = Ridge()
    model.fit(t, YPWH)

    a=model.coef_
    b=model.intercept_

    Y_new=YPWH-model.predict(t)

    Y[type+'_tweaked']=Y_new

    #plt.plot(t, model.predict(t), color='red')
    #plt.plot(t, Y_new, color='blue')
    #plt.show()
    return Y
