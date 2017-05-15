


import pandas as pd
from .base import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
DATA_PATH='Datasets/Data/'
FILENAME='STABLE_GJOA.csv'

MEAN_PATH='Models/NeuralNetworks/ConfigFilesUseful/GJOA_GAS_MEAN.csv'
FILENAME='STABLE_GJOA.csv'
SENSORS=['CHK','PWH','PBH','PDC','QGAS']
GJOA_QGAS_COL='GJOA_SEP_1_QGAS_'
DATA_TYPE = 'mea'

X_tags=['CHK','PWH','PBH','PDC']
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

    X['time_sample_days'] = (data['T2'] - data['T1']) / (1000 * 60 * 60*24)

    mean_time = np.max(X['time_sample_days'])
    print(mean_time)

    DROP = [900]#,443,605,442]
    X.drop(DROP, inplace=True)
    Y.drop(DROP, inplace=True)


    X['time']=np.arange(0,len(X))


    print('MAX: {}, MEAN: {}'.format(np.max(Y['GJOA_QGAS']),np.mean(Y['GJOA_QGAS'])))
    print('Data size: {}'.format(len(Y)))

    ind_zero_all=None
    for key in well_names:
        ind_gas_zero = Y[key + '_QGAS'] == 0
        ind_zero = X[key + '_CHK'] < CHK_THRESHOLD
        if ind_zero_all is None:
            ind_zero_all=ind_zero
        else:
            ind_zero_all=ind_zero_all&ind_zero


        X = set_index_values_to_zero(X, ind_zero, key + '_CHK')
        Y = set_index_values_to_zero(Y, ind_zero, key + '_QGAS')

        X[key + '_shifted_CHK'] = X[key + '_CHK'].shift(1) * -1
        X[key + '_shifted_PWH'] = X[key + '_PWH'].shift(1)
        X[key + '_shifted_PDC'] = X[key + '_PDC'].shift(1)
        X[key + '_shifted_PBH'] = X[key + '_PBH'].shift(1)

        delta_CHK = X[key + '_CHK'] - X[key + '_CHK'].shift(1)

        Y[key + '_delta_PWH'] = Y[key + '_PWH'] - Y[key + '_PWH'].shift(1)
        Y[key + '_delta_PDC'] = Y[key + '_PDC'] - Y[key + '_PDC'].shift(1)
        Y[key + '_delta_PBH'] = Y[key + '_PBH'] - Y[key + '_PBH'].shift(1)
        X[key + '_delta_CHK'] = delta_CHK
        Y[key + '_delta_CHK'] = delta_CHK

    Y = set_index_values_to_zero(Y, ind_zero_all, 'GJOA_QGAS')

    GjoaData=DataContainer(X,Y,name='GJOA',csv_path=MEAN_PATH,well_names=well_names)
    if False:
        CTHRESH=10
        ind_zero=None
        chk_zero=None
        for key in well_names:

            if ind_zero is None:
                ind_zero = abs(X[key + '_delta_CHK']) > CTHRESH
                chk_zero = abs(X[key + '_CHK']) ==0
                ind_zero=ind_zero#|ind_zero2
            else:
                ind_temp = abs(X[key + '_delta_CHK']) > CTHRESH
                chk_zero_temp = abs(X[key + '_CHK']) == 0
                ind_zero=ind_zero|ind_temp#|ind_zero2
                chk_zero=chk_zero&chk_zero_temp
        ind_zero=ind_zero|chk_zero
        GjoaData.X =GjoaData.X[~ind_zero]
        GjoaData.Y = GjoaData.Y[~ind_zero]
        GjoaData.X_transformed = GjoaData.X_transformed[~ind_zero]
        GjoaData.Y_transformed = GjoaData.Y_transformed[~ind_zero]

    if False:

        #cols = ['GJOA_QGAS','E1_QGAS']
        cols=[]
        #ols=['F1_PBH','F1_PWH','F1_PDC']
        cols=['F1_CHK']
        MAP_cols={'D3_CHK':'G3 choke opening','E1_CHK':'G4 choke opening'}
        #for key in well_names:#['QGAS','PBH','PDC','PWH','CHK']:
        #    cols.append(key+'_'+'QGAS')
        fig, axes = plt.subplots(len(cols), 1, sharex=True)
        axes=[axes]

        for i, key in zip(range(0, len(cols)), cols):
            #i=0
            try:
                axes[i].scatter(X['time'], GjoaData.X[key], color='black')
            except(KeyError):
                axes[i].scatter(X['time'], GjoaData.Y_transformed[key], color='black')
           # axes[i].set_title('G2_QGAS',fontsize=30)
            axes[i].set_xlabel('Sample number',fontsize=25)
            axes[i].tick_params(labelsize=25)
            axes[i].set_ylabel('Choke opening [%]',fontsize=25)
            #plt.legend(fontsize=40)
            axes[i].grid(which='major', linestyle='-')
            axes[i].set_axisbelow(True)
            fig.subplots_adjust(wspace=0.08, hspace=.18, top=0.95, bottom=0.06, left=0.04, right=0.99)

        plt.show()



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
