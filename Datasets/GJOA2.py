


import pandas as pd
from .base import *
from .visualize import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
DATA_PATH='Datasets/Data/'
FILENAME='STABLE_GJOA_OIL_NEW.csv'

X_tags=['CHK','PWH','PBH','PDC']
X_GJOA_tags=['RISER_OIL_B_PDC','RISER_OIL_B_CHK','RISER_OIL_A_PDC']

Y_GJOA_tags=['TOTAL_QOIL','TOTAL_QGAS_DEPRECATED','TOTAL_QWAT','SEP_1_QOIL','SEP_1_QWAT','SEP_1_WCT','SEP_1_QGAS','SEP_1_QLIQ','SEP_3_QWAT_1','SEP_2_QWAT']
Y_tags=['QOIL','QWAT','QGAS','PDC','PWH','PBH','CHK']

DATA_TYPE = 'mea'



well_names=['C1','C2','C3','C4','D1','B3','B1']


def fetch_gjoa_data():
    data=pd.read_csv(DATA_PATH+FILENAME)

    X, Y = data_to_X_Y(data)
    X['time'] = np.arange(0, len(X))

    X,Y=preprocesss(X, Y)
    X,Y=generate_total_export_variables(X,Y)

    #print('MAX: {}, MEAN: {}'.format(np.max(Y['GJOA_TOTAL_SUM_QOIL']), np.mean(Y['GJOA_TOTAL_SUM_QOIL'])))
    #print('Data size: {}'.format(len(Y)))

    GjoaData=DataContainer(X,Y,name='GJOA2')

    if False:

        cols=['GJOA_TOTAL_SUM_QOIL','C3_QOIL']
        fig,axes=plt.subplots(len(cols),1,sharex=True)
        #axes=[axes]

        for i,key in zip(range(0,len(cols)),cols):
            try:
                axes[i].scatter(X['time'], GjoaData.X_transformed[key], color='blue')
            except(KeyError):
                #axes[i].scatter(X['time'], GjoaData.Y_transformed[key], color='blue')
                axes[i].hist( (GjoaData.Y[key])**2)
            axes[i].set_title(key)
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel(key)
            fig.subplots_adjust(wspace=0.08, hspace=.18, top=0.95, bottom=0.06, left=0.04, right=0.99)

        plt.show()

    return GjoaData

def data_to_X_Y(data):
    X=pd.DataFrame()
    Y=pd.DataFrame()
    X['X']=np.zeros((len(data),))
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
    for tag in X_GJOA_tags:
        col = 'GJOA' + '_' + tag
        X[col] = data[col + '_' + DATA_TYPE]
    for tag in Y_GJOA_tags:
        col = 'GJOA' + '_' + tag
        Y[col] = data[col + '_' + DATA_TYPE]


    return X,Y


def calculate_sum_multiphase(Y):
    tags_OIL = []
    tags_GAS = []
    for key in well_names:
        name_oil = key + '_' + 'QOIL'
        name_qgas = key + '_' + 'QGAS'
        tags_OIL.append(name_oil)
        tags_GAS.append(name_qgas)

    sum_oil = Y[tags_OIL].sum(axis=1)
    sum_gas = Y[tags_GAS].sum(axis=1)

    return sum_oil,sum_gas



def set_chk_zero_values_to_zero(X,Y):

    for key in well_names:
        ind_gas_zero = Y[key + '_QGAS'] == 0
        ind_oil_zero = Y[key + '_QOIL'] == 0


        #X = set_index_values_to_zero(X, ind_oil_zero, key + '_CHK')
        #X = set_index_values_to_zero(X, ind_gas_zero, key + '_CHK')
        # X[key + '_CHK'][ind_gas_zero] = 0

        ind_zero = X[key + '_CHK'] < CHK_THRESHOLD

        #ind_gas_zero = Y[key + '_QGAS'] == 0
        #ind_oil_zero = Y[key + '_QOIL'] == 0
        ##ind_gas_zero=ind_gas_zero!=ind_zero
        #ind_oil_zero = ind_oil_zero != ind_zero

        #X = X[~ind_oil_zero]
        #Y = Y[~ind_oil_zero]

        #X = X[~ind_gas_zero]
        #Y = Y[~ind_gas_zero]

        #X[key+'_1_PBH']=X[key+'_PBH'].copy()

        X=set_index_values_to_zero(X, ind_zero, key + '_PWH')
        X = set_index_values_to_zero(X, ind_zero, key + '_PBH')
        X = set_index_values_to_zero(X, ind_zero, key + '_PDC')

        Y = set_index_values_to_zero(Y, ind_zero, key + '_PWH')
        Y = set_index_values_to_zero(Y, ind_zero, key + '_PBH')
        Y = set_index_values_to_zero(Y, ind_zero, key + '_PDC')

        Y = set_index_values_to_zero(Y, ind_zero, key + '_QOIL')
        Y = set_index_values_to_zero(Y, ind_zero, key + '_QGAS')

        X[key + '_CHK_zero'] = np.array([0 if x < CHK_THRESHOLD else 1 for x in X[key + '_CHK']])

    return X,Y

def preprocesss(X,Y):
    DROP = [808, 807, 173, 416, 447, 487]

    X['time'] = np.arange(0, len(X))

    if False:

        cols=[]
        TAG='PWH'
        for key in well_names:
            cols.append(key+'_'+TAG)
        fig,axes=plt.subplots(len(cols),1,sharex=True)
        #axes=[axes]

        for i,key in zip(range(0,len(cols)),cols):
            try:
                axes[i].scatter(X['time'], X[key], color='blue')
            except(KeyError):
                axes[i].scatter(X['time'], Y[key], color='blue')
            axes[i].set_title(key)
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel(key)
            fig.subplots_adjust(wspace=0.08, hspace=.18, top=0.95, bottom=0.06, left=0.04, right=0.99)

        plt.show()


    X.drop(DROP, inplace=True)
    Y.drop(DROP, inplace=True)

    #X.loc[114:129, 'GJOA_RISER_OIL_B_CHK'] = 0

    X,Y=set_chk_zero_values_to_zero(X,Y)



    return X,Y


def generate_total_export_variables(X,Y):
    sum_oil, sum_gas = calculate_sum_multiphase(Y)

    Y['GJOA_TOTAL_SUM_QOIL'] = sum_oil
    Y['GJOA_OIL_SUM_QGAS'] = sum_gas

    Y['GJOA_OIL_QGAS'] = Y['GJOA_TOTAL_QGAS_DEPRECATED'] - Y['GJOA_SEP_1_QGAS']




    # Remove bias
    Y['GJOA_OIL_QGAS'] += np.ones((len(Y),)) * 5000

    # Remove negative values
    Y = negative_values_to_zero(Y, 'GJOA_OIL_QGAS')

    return X,Y





