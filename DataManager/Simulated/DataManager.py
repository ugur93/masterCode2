


import numpy as np

from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import pandas as pd
import warnings

N_SAMPLES=1000
CHOKE_FREQ=int(N_SAMPLES/40)
N_WELLS=4
N_SHUTDOWNS=1
N_SHUTDOWN_STEPS=3
N_SHUTDOWN_SCALE=5
np.random.seed(100)

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
        return data

    chkInputs=pd.DataFrame(index=range(0,N_SAMPLES),columns=['chk'+str(i) for i in range(1,N_WELLS+1)])
    for i in range(N_WELLS):
        choke_min,choke_max=generateChokeConfig()
        status_shutdown=False
        shutdown_samples=np.random.randint(2,10,N_SHUTDOWNS)

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
        chkInputs['chk'+str(i+1)]=chk_data
    return chkInputs

def getSimulatedOilFieldData():

    chkInputs=getSimulatedChokeData()
    wellOutputs=pd.DataFrame(index=range(0,N_SAMPLES),columns=['well_'+str(i) for i in range(1,N_WELLS+1)])
    totalOutput=np.zeros((N_SAMPLES,1))
    for i in range(N_WELLS):
        a=np.random.randint(1,10,1)
        b=np.random.randint(0, 100, 1)
        c=np.linspace(0,10,N_SAMPLES)*np.random.rand()
        noise = np.random.rand(N_SAMPLES, 1)*10
        data=f_linear(a,b,c,chkInputs['chk'+str(i+1)])+noise
        #print(data.shape)
        wellOutputs['well_'+str(i+1)]=data
        totalOutput+=data


    totalOutput=pd.DataFrame(totalOutput,columns=['Q'])

    plotData(chkInputs, wellOutputs, totalOutput)

    #chkInputs=scale(chkInputs,chkInputs_scaler)
    #wellOutputs = scale(wellOutputs,wellOutputs_scaler)
    #totalOutput = scale(totalOutput,totalOutput_scaler)

    print('Data generated with sample-size of: {}'.format(N_SAMPLES))
    return chkInputs,wellOutputs,totalOutput

def f_linear(a,b,c,x):
    return a*x.values.reshape(N_SAMPLES,1)+b*np.ones((N_SAMPLES,1))+c.reshape((N_SAMPLES,1))

def scale(data,scaler):
    cols=data.columns
    data=scaler.fit_transform(data)

    return pd.DataFrame(data=data,columns=cols)


def plotData(chkInputs,wellOutputs,totalOutput):
    plotChokeInputs(chkInputs)
    plotWellOutputs(wellOutputs)
    plt.figure()
    plt.plot(totalOutput)
    plt.show()
def plotChokeInputs(chokeInputs):
    plt.subplot(2,2,1)
    plt.plot(chokeInputs['chk1'])
    plt.subplot(2,2,2)
    plt.plot(chokeInputs['chk2'])
    plt.subplot(2,2, 3)
    plt.plot(chokeInputs['chk3'])
    plt.subplot(2, 2, 4)
    plt.plot(chokeInputs['chk4'])

def plotWellOutputs(wellOutputs):
   # plt.figure()
    plt.subplot(2,2,1)
    plt.plot(wellOutputs['well_1'])
    plt.subplot(2,2,2)
    plt.plot(wellOutputs['well_2'])
    plt.subplot(2,2, 3)
    plt.plot(wellOutputs['well_3'])
    plt.subplot(2, 2, 4)
    plt.plot(wellOutputs['well_4'])
    plt.title('Well outputs')


