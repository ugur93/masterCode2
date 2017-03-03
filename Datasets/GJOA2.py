


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
    DROP=[808,807, 173,416,447,487]
    X,Y=data_to_X_Y(data)
    print(len(X))
    X.drop(DROP, inplace=True)
    Y.drop(DROP, inplace=True)
    print(len(X))
    #exit()
    #Y = add_choke_delta(Y)
    #Y = add_well_delta(Y)

    #X = add_choke_delta(X)
    #X = add_well_delta(X)
    X['time'] = np.arange(0, len(X))
    Y['non']=np.zeros((len(X),))
    for key in well_names:
        ind_zero = X[key + '_CHK'] <=5

        X[key + '_PWH'][ind_zero] = 0
        X[key + '_PBH'][ind_zero] = 0
        X[key + '_PDC'][ind_zero] = 0
        Y[key + '_PWH'][ind_zero] = 0
        Y[key + '_PBH'][ind_zero] = 0
        Y[key + '_PDC'][ind_zero] = 0
        Y[key + '_QOIL'][ind_zero]=0

        X[key + '_CHK_zero'] = np.array([0 if x <= 5 else 1 for x in X[key + '_CHK']])
        X[key + '_neg'] = -1*np.ones((len(X),))#np.array([0 if x <= 5 else 1 for x in X[key + '_CHK']])
    sum_oil, sum_gas = calculate_sum_multiphase(Y)

    Y['GJOA_OIL_QGAS'] = Y['GJOA_TOTAL_QGAS_DEPRECATED'] - Y['GJOA_SEP_1_QGAS']# + np.ones((len(Y),)) * 5000
    Y['GJOA_OIL_QGAS_OLD']=Y['GJOA_OIL_QGAS'].copy()


    #ind_zero=Y['GJOA_OIL_QGAS']<0

    #Remove bias
    Y['GJOA_OIL_QGAS']+= np.ones((len(Y),)) * 5000

    #Remove negative values
    Y = negative_values_to_zero(Y, 'GJOA_OIL_QGAS')
    #Y=Y[~ind_zero]
    #X=X[~ind_zero]

    Y['GJOA_TOTAL_SUM_QOIL'] = sum_oil
    Y['GJOA_OIL_SUM_QGAS'] = sum_gas






    if False:
        fig,axes=plt.subplots(2,1,sharex=True)
        axes[0].scatter(X['time'], Y['C1_QOIL'], color='blue')
        axes[0].set_title('C1_PWH')
        axes[1].scatter(X['time'], X['C1_CHK'], color='blue')
        axes[1].set_title('C1_CHK')

        #for i in range(len(well_names)):
        #    plt.subplot(4,2,i+1)
        #    plt.scatter(X['time'], X[well_names[i]+'_CHK'], color='black')
        #    plt.title(well_names[i]+'_CHK')
        plt.show()

        #axes[0].scatter(X['time'], Y['GJOA_OIL_QGAS'], color='red')
        #axes[0].scatter(X['time'], Y['GJOA_OIL_QGAS_OLD'], color='green')
        #axes[0].scatter(X['time'], Y['GJOA_OIL_SUM_QGAS'], color='blue')
        #axes[1].scatter(X['time'], Y['GJOA_OIL_SUM_QGAS']-Y['GJOA_OIL_QGAS_OLD'], color='blue')
        #axes[2].scatter(X['time'], Y['GJOA_OIL_SUM_QGAS']-Y['GJOA_OIL_QGAS'], color='blue')
        plt.show()

    #test_bed(X,Y,sum_gas,sum_oil)

    print('MAX: {}, MEAN: {}'.format(np.max(Y['GJOA_TOTAL_SUM_QOIL']), np.mean(Y['GJOA_TOTAL_SUM_QOIL'])))
    print('Data size: {}'.format(len(Y)))


    col='C1_PWH'
    GjoaData=DataContainer(X,Y,name='GJOA2')
    #plt.plot(GjoaData.X_transformed[col])
    #plt.show()

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





def test_bed(X,Y,sum_gas,sum_oil):
    #pass
    # plt.figure()
    # plt.scatter(X['time'], Y['GJOA_TOTAL_QGAS_DEPRECATED'], color='blue',label='GJOA_TOTAL_QGAS_DEPRECATED')
    # plt.scatter(X['time'], Y['GJOA_SEP_1_QGAS'], color='red',label='GJOA_SEP_1_QGAS')
    # plt.legend()
    # plt.plot(Y['GJOA_QGAS']-sum_oil,color='red')
    # plt.plot(sum_oil,color='blue')
    # plt.show()

    #plt.scatter(X['time'], sum_gas-Y['GJOA_OIL_QGAS'], color='blue', label='sum_gas - GJOA_OIL_QGAS')
    plt.scatter(X['time'], Y['GJOA_OIL_QGAS'], color='blue', label='GJOA_OIL_QGAS')
    plt.scatter(X['time'], sum_gas, color='black', label='Sum oil wells QGAS ')
    plt.xlabel('time')
    plt.ylabel('QGAS sm3/h')
    plt.legend()
    plt.show()
    # Y=Y[~ind]
    # X=X[~ind]
    # Y['GJOA_QGAS'] = Y['GJOA_QGAS'] #- np.mean(Y['GJOA_QGAS'] - sum_oil)
    i=1
    for tag in well_names:
        #plt.figure()
        name=tag+'_'+'PDC'
        fig,axes=plt.subplots(2,1,sharex=True)
        #plt.subplot(2,1,1)
        i+=1
        axes[0].plot(X[name])
        axes[0].set_title(name)
        name = tag + '_' + 'CHK'
        #plt.subplot(2, 1, 2)
        i += 1
        axes[1].plot(X[name])
        axes[1].set_title(name)
    plt.show()
    plot_input_to_total(X, Y, 'TOTAL_SUM_QOIL', well_names)
    plot_input_to_well(X, Y, 'QOIL', well_names)
    plt.show()

def add_choke_delta(Y):
    for name in well_names:
        Y[name+'_c_delta']=Y[name+'_PWH']-Y[name+'_PDC']
    return Y
def add_well_delta(Y):
    for name in well_names:
        Y[name+'_w_delta']=Y[name+'_PBH']-Y[name+'_PWH']
    return Y