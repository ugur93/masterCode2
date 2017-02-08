import numpy as np

from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import pandas as pd
from .base import *
from matplotlib import cm

N_SAMPLES=1000
CHOKE_FREQ=int(N_SAMPLES/50)
N_WELLS=4
N_SHUTDOWNS=3
N_SHUTDOWN_STEPS=3
N_SHUTDOWN_SCALE=5
np.random.seed(151)

WELL_NAMES=['F1','B2','D3','E1']

def generateChokeConfig():
    choke_mean = np.random.randint(20, 80, 1)
    choke_min = int(choke_mean -5)
    choke_max = int(choke_mean +5)
    if choke_max > 100:
        choke_max = 100
    #print(choke_min,choke_max)
    return choke_min,choke_max
def getSimulatedChokeData():
    def shutdown(curr_choke_val):
        data=np.zeros((CHOKE_FREQ,1))
        for i in range(N_SHUTDOWN_STEPS):
            data[int(CHOKE_FREQ/N_SHUTDOWN_SCALE*(i)):int(CHOKE_FREQ/N_SHUTDOWN_SCALE*(i+1))]=\
                int(curr_choke_val/(2**(i+1)))*np.ones((int(CHOKE_FREQ/N_SHUTDOWN_SCALE),1))
        return data
    def turnOn(next_choke_val):
        data = np.zeros((CHOKE_FREQ, 1))
        for i in range(N_SHUTDOWN_STEPS):
            data[int(CHOKE_FREQ / N_SHUTDOWN_SCALE * (i+N_SHUTDOWN_STEPS-1)):int(CHOKE_FREQ / N_SHUTDOWN_SCALE * (i + N_SHUTDOWN_STEPS))] = \
                int(next_choke_val / (2 ** (N_SHUTDOWN_STEPS-i))) * np.ones((int(CHOKE_FREQ / N_SHUTDOWN_SCALE), 1))
        #   print(int(next_choke_val / (2 ** (N_SHUTDOWN_STEPS-i))))
        #print('::::::::')
        return data

    chkInputs=pd.DataFrame(index=range(0,N_SAMPLES),columns=[WELL_NAMES[i]+'_CHK' for i in range(0,N_WELLS)])
    for i in range(N_WELLS):
        choke_min,choke_max=generateChokeConfig()
        status_shutdown=False
        shutdown_samples=np.random.randint(2,N_SAMPLES/CHOKE_FREQ,N_SHUTDOWNS)

        chk_data=np.zeros((N_SAMPLES,1))
        prev_sample=0
        for current_sample in range(CHOKE_FREQ, N_SAMPLES+CHOKE_FREQ, CHOKE_FREQ):
            if status_shutdown:
                chk_data[prev_sample:current_sample]=turnOn(np.random.randint(choke_min,choke_max,1))
                status_shutdown=False
            elif current_sample/CHOKE_FREQ in shutdown_samples and status_shutdown==False:
                chk_data[prev_sample:current_sample]=shutdown(chk_data[prev_sample-1])
                status_shutdown=True
            else:
                chk_data[prev_sample:current_sample]=np.random.randint(choke_min,choke_max,1)*np.ones((CHOKE_FREQ,1))
            prev_sample=current_sample
        chkInputs[WELL_NAMES[i]+'_CHK']=chk_data
    return chkInputs

def fetchSimulatedData():

    X=getSimulatedChokeData()
    X_Q=pd.DataFrame(index=range(0,N_SAMPLES),columns=[WELL_NAMES[i]+'_QGAS' for i in range(0,N_WELLS)])

    XT=pd.DataFrame()
    Y=np.zeros((N_SAMPLES,1))
    WELL_PARAMS={}
    for i in range(N_WELLS):
        a=np.random.randint(1,10,1)
        b=np.random.randint(1, 100, 1)
        c=np.linspace(0,10,N_SAMPLES)*np.random.rand()
        noise = np.random.rand(N_SAMPLES, 1)*25
        data=f_linear(a,b,c,X[WELL_NAMES[i]+'_CHK'])+noise
        #print(data.shape)
        X_Q[WELL_NAMES[i]+'_QGAS']=data
        Y+=data
        WELL_PARAMS[WELL_NAMES[i]]={'a':a,'b':b}
        XT[WELL_NAMES[i]]=X[WELL_NAMES[i]+'_CHK']
        x_toggle=np.array([0 if val == 0 else 1 for val in X[WELL_NAMES[i]+'_CHK']])
        XT[WELL_NAMES[i]+'_b']=b*np.ones((N_SAMPLES,1))*x_toggle.reshape(N_SAMPLES,1)


    #print_rank(XT,'Simulated')


    Y=pd.DataFrame(Y,columns=['GJOA_QGAS'])

    Y=pd.concat([Y,X_Q],axis=1)

    plotData(X, X_Q, Y)

    print('Data generated with sample-size of: {}'.format(N_SAMPLES))

    #plotChokeInputs(X)
    SimData=DataContainer(X,Y,X_Q,params=WELL_PARAMS,name='Simulated',Y_SCALE=100)
    print(SimData.Y.columns)
    return SimData

def f_linear(a,b,c,x):
    x_toggle=np.array([0 if val==0 else 1 for val in x])
    return a*x.values.reshape(N_SAMPLES,1)+b*np.ones((N_SAMPLES,1))*x_toggle.reshape(N_SAMPLES,1)#+c.reshape((N_SAMPLES,1))


def plotData(X,X_Q,Y):
    plt.subplot(2,1,1)
    plotChokeInputs(X)
    plt.subplot(2,1,2)
    plotWellOutputs(X_Q)
    plt.figure()
    plt.plot(Y['GJOA_QGAS'])
    plt.show()
def plotChokeInputs(X):
    for i in range(1,N_WELLS+1):
        #plt.subplot(2,2,i)
        plt.plot(X[WELL_NAMES[i-1]+'_CHK'],label=WELL_NAMES[i-1]+'_CHK')
    plt.title('CHK INPUTS')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=4, mode="expand", borderaxespad=0., fontsize=10)


def plotWellOutputs(X_Q):
   for i in range(1,N_WELLS+1):
       #plt.subplot(2, 2, i)
       plt.plot(X_Q[WELL_NAMES[i-1]+'_QGAS'],label=WELL_NAMES[i-1]+'_QGAS')
   plt.title('WELL OUTPUTS')
   plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
              ncol=4, mode="expand", borderaxespad=0., fontsize=10)
   #plt.show()
