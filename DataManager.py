


import numpy as np

from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import pandas as pd
import warnings
def generate_data():

    x=np.linspace(-5,5,1000)

    print x.shape

    x1=np.sin(x)*np.cos(x)
    x2=np.exp(x/10)*np.sin(x)

    input={'input1':x1,'input2':x2}
    #input_test={'input1':x1,'input2':x2}

    Y=x1+x2+np.random.normal(0,0.1,1000)

    output={'main_output':Y}

    return input,output


def get_concrete_data(plot=False):
    cols = ['Cement','Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate','Age',
            'Concrete compressive strength']
    df=pd.read_excel('Concrete_Data.xls',names=cols)
    cols = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate',
            'Fine Aggregate', 'Age']
    X=pd.DataFrame(data=df[cols].values,columns=cols)
    Y=pd.DataFrame(data=df['Concrete compressive strength'],columns=['Concrete compressive strength'])

    X,Y=shuffle(X,Y,random_state=1)

    #print X[['Age','Water']]

    if plot:
        num = np.arange(1, len(cols) + 1)
        fig = plt.figure()
        fig.subplots_adjust(hspace=.5)
        for i, key in zip(num, cols):
            ax = fig.add_subplot(3, 3, i)
            ax.scatter(df[key].values, Y)
            ax.set_ylabel('Strength')
            ax.set_xlabel(key)
    X1_scaler=StandardScaler()
    X2_scaler = StandardScaler()
    Y_scaler=StandardScaler()

    input2 = X[['Age', 'Water','Cement','Superplasticizer']]
    input1 = X[['Blast Furnace Slag', 'Fly Ash' ,'Coarse Aggregate', 'Fine Aggregate']]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        input1 = X1_scaler.fit_transform(input1)
        input2=X2_scaler.fit_transform(input2)
        Y = Y_scaler.fit_transform(Y)


    input={'input1': input1,'input2':input2}
    output={'main_output':Y}
    return input,output