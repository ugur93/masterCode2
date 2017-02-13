


import pandas as pd
from .base import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
DATA_PATH='Datasets/Data/'
FILENAME='STABLE_GJOA.csv'

SENSORS=['CHK','PWH','PBH','PDC','QGAS']
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




tags=['F1','B2','D3','E1']
def fetch_gjoa_data():
    data=pd.read_csv(DATA_PATH+FILENAME)


    WELL_F1=data[generate_well_headers('F1',data_type)]
    WELL_B2=data[generate_well_headers('B2',data_type)]
    WELL_D3=data[generate_well_headers('D3',data_type)]
    WELL_E1=data[generate_well_headers('E1',data_type)]

    GJOA_SEP_1=data[GJOA_QGAS_COL+data_type].to_frame('GJOA_QGAS_mea') #Rename col name

    data={'WELL_F1':WELL_F1,'WELL_B2':WELL_B2,'WELL_D3':WELL_D3,'WELL_E1':WELL_E1,'GJOA_SEP_1':GJOA_SEP_1}

    X,Y,Y_Q=data_to_X_Y(data)

    Y=add_diff(Y)

    Y=addModify(X,Y,'E1_PWH')
    X = addModify(Y, X, 'E1_PDC')
    X['time']=np.arange(0,len(X.index))

    print('MAX: {}, MEAN: {}'.format(np.max(Y['GJOA_QGAS']),np.mean(Y['GJOA_QGAS'])))

    #plt.plot(Y['GJOA_QGAS'])
    #plt.show()
    #Y=pwh_to_zero(X,Y.copy())


    for tag in tags:
        plt.plot(Y[tag+'_'+'PWH'],label=tag)
        plt.title(tag)
    plt.legend()

    #plt.show()



    ind=X['B2_CHK']<5

    #X=X[~ind]
    #Y=Y[~ind]

    #a=X['F1_CHK']*np.sqrt(2000*Y['F1_deltap'])*10.1
    #a=Y['F1_deltap']*100.5*2000
    #plt.plot(a,color='red')
    #plt.plot((Y['F1_QGAS']/X['F1_CHK'])**2,color='blue')
    #plt.show()

    #print_rank(X,'GJOA')
    GjoaData=DataContainer(X,Y,Y_Q,name='GJOA')
    #plot_input_to_well(X,Y)
    #plt.plot(GjoaData.Y_transformed['GJOA_QGAS'])
    #plt.show()

    #plot_scaled(GjoaData,'CHK')
    #splot_scaled(GjoaData, 'PWH')
    #plt.subplot(2,1,1)
    #plt.plot(GjoaData.Y_transformed)
    #plt.subplot(2,1,2)
    #plt.plot(GjoaData.Y)
    #plot_pressure(X)
    #plt.show()

    #exit()
    #print(data['WELL_F1'])
    #plot_scatter(data['WELL_E1'],'E1_CHK_mea','GJOA_QGAS',GJOA_SEP_1)
    #visualizeData(GjoaData.X_transformed)
    return GjoaData
def plot_pressure(X):
    pressures=['PDC','PWH']


    for pres in pressures:
        plt.figure()
        for tag in tags:
            name=tag+'_'+pres
            plt.plot(X[name],label=name)
        plt.legend()

    pres_y = 'PDC'
    pres_x = 'CHK'
    plt.figure()
    i=1
    for tag in tags:
        namey=tag+'_'+pres_y
        namex=tag+'_'+pres_x
        plt.subplot(2,2,i)
        plt.scatter(X[namex],X[namey])
        plt.xlabel(namex)
        plt.ylabel(namey)
        i+=1
    pres_y = 'PWH'
    pres_x = 'CHK'
    plt.figure()
    i = 1
    for tag in tags:
        namey = tag + '_' + pres_y
        namex = tag + '_' + pres_x
        plt.subplot(2, 2, i)
        plt.scatter(X[namex], X[namey])
        plt.xlabel(namex)
        plt.ylabel(namey)
        i += 1
    pres_y = 'PWH'
    pres_x = 'PDC'
    plt.figure()
    i = 1
    for tag in tags:
        namey = tag + '_' + pres_y
        namex = tag + '_' + pres_x
        plt.subplot(2, 2, i)
        plt.scatter(X[namex], X[namey])
        plt.xlabel(namex)
        plt.ylabel(namey)
        i += 1

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
                    tag_name=sensor_splitted[0]+'_'+sensor_splitted[1]
                    X[tag_name]=data[key][sensor]
                    if col=='CHK':
                        #print(X)
                        ind=X[tag_name]<0
                        X[tag_name][ind]=0
                        #X[X<0]=0

            #for col in Y_Q_COLS:
            #    if sensor_splitted[1]==col and sensor_splitted[0]!='GJOA':
            #        Y_Q[sensor_splitted[0]+'_'+sensor_splitted[1]]=data[key][sensor]
            for col in Y_COLS:
                if sensor_splitted[1] == col and sensor_splitted[0] != 'GJOA' and sensor_splitted[0]:
                    Y[sensor_splitted[0] + '_' + sensor_splitted[1]] = data[key][sensor]
    Y['GJOA_QGAS']=data['GJOA_SEP_1']
    return X,Y,Y_Q


def plot_scaled(data,ending):
    X=data.X
    X_scaled=data.X_transformed

    for i in range(4):
        plt.figure()
        plt.subplot(2,1,1)
        name=tags[i]+'_'+ending
        plt.plot(X [name],color='blue',label='Original')
        plt.title(name+' - ORIGINAL')
        plt.subplot(2,1,2)
        plt.plot(X_scaled[name],color='blue',label='Transformed')
        plt.title(name+'- Transformed')
    plt.show()

def add_diff(X):
    for key in tags:
        PWH_tag=key+'_'+'PWH'
        PDC_tag=key+'_'+'PDC'

        X[key+'_deltap']=X[PWH_tag]-X[PDC_tag]
    return X


def plot_scatter(X):
    pressures = ['PDC', 'PWH']
    for tag in tags:
        plt.figure()
        i=1
        for pres in pressures:
            tag_y = tag + '_' + pres
            tag_x=tag+'_'+'CHK'
            plt.subplot(3,1,i)
            i+=1
            plt.scatter(X[tag_x],X[tag_y], label=tag_y)
            plt.xlabel(tag_x)
            plt.ylabel(tag_y)
            plt.title(tag_y)
        tag_y = tag + '_' + 'PDC'
        tag_x = tag + '_' + 'PWH'
        plt.subplot(3, 1, i)
        i += 1
        plt.scatter(X[tag_x], X[tag_y], label=tag_y)
        plt.xlabel(tag_x)
        plt.ylabel(tag_y)
        plt.title(tag_y)
        #plt.legend()

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
def pwh_to_zero(X,Y):
    for tag in tags:
        ind=X[tag+'_CHK']<5
        Y[tag+'_PWH'][ind]=0
        #plt.subplot(2,1,1)
        #plt.plot(X[tag+'_PDC'])
        #plt.subplot(2,1,2)
        #plt.plot(X[tag+'_PWH'])
        #plt.show()
    return Y

def plot_input_to_well(X,Y):
    cols=['CHK','PDC','PWH','PBH']
    out_ending='QGAS'
    for tag in tags:
        i=1
        plt.figure()
        for col in cols:
            name_input=tag+'_'+col
            name_output=tag+'_'+out_ending
            plt.subplot(2,2,i)
            i+=1
            plt.scatter(X[name_input],Y[name_output],color='black')
            plt.title(name_input)
            plt.xlabel(name_input)
            plt.ylabel(name_output)
    plt.show()
def plot_input_to_total(X,Y):
    cols=['CHK','PDC','PWH','PBH']
    out_ending='QGAS'
    for tag in tags:
        i=1
        plt.figure()
        for col in cols:
            name_input=tag+'_'+col
            name_output='GJOA'+'_'+out_ending
            plt.subplot(2,2,i)
            i+=1
            plt.scatter(X[name_input],Y[name_output],color='black')
            plt.title(name_input)
            plt.xlabel(name_input)
            plt.ylabel(name_output)
    plt.show()