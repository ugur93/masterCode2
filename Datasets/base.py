
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,FunctionTransformer
import pandas as pd
import numpy as np


def print_rank(X,name):
    print('Rank {}: '.format(name))
    rank = np.linalg.matrix_rank(X)
    if rank == X.shape[1]:
        print('Input has full rank of ' + str(rank))
    else:
        print('Rank of ' + str(rank) + ' is not full rank')


def func_transform(x,scaler):
    return x/scaler


def inverse_func_transform(x,scaler):
    return x*scaler


def get_cols_that_ends_with(df,tag):

    cols=[]

    for col in df.columns:
        if col.split('_')[-1]==tag:
            cols.append(col)
    return cols


class Transformer:
    def __init__(self):
        self.PRESSURES=['PDC','PBH','PWH']
        self.QGAS=['QGAS','DEPRECATED']
        self.CHK=['CHK','time']
        self.QOIL=['QOIL','SUM']
        self.QWAT=['QWAT']

        self.SCALES={'PRESSURES':100,'QGAS':100000,'CHK':100,'QOIL':100,'QWAT':100}

    def transform(self,data):
        data_transformed=data.copy()
        #Pressures
        cols=self.get_cols_that_ends_with(data,self.PRESSURES)
        data_transformed[cols]=data_transformed[cols]/self.SCALES['PRESSURES']
        #print(cols)
        # GAS
        cols = self.get_cols_that_ends_with(data, self.QGAS)
        data_transformed[cols]= data_transformed[cols] / self.SCALES['QGAS']
        #print(cols)

        # Pressures
        cols = self.get_cols_that_ends_with(data, self.CHK)
        print(data_transformed.columns)
        print(cols)
        data_transformed[cols]= data_transformed[cols]/self.SCALES['CHK']
        #print(cols)
        # Pressures
        cols = self.get_cols_that_ends_with(data, self.QOIL)
        data_transformed[cols]= data_transformed[cols] / self.SCALES['QOIL']
        #print(data_transformed[cols])
        #print(cols)
        # Pressures
        cols = self.get_cols_that_ends_with(data, self.QWAT)
        data_transformed[cols]= data_transformed[cols] / self.SCALES['QWAT']

        return data_transformed


    def inverse_transform(self,data,tag='ALL'):
        data_transformed = data.copy()
        # Pressures
        cols = self.get_cols_that_ends_with(data, self.PRESSURES)
        data_transformed[cols]= data_transformed[cols] * self.SCALES['PRESSURES']

        # GAS
        cols = self.get_cols_that_ends_with(data, self.QGAS)

        data_transformed[cols] = data_transformed[cols] * self.SCALES['QGAS']

        # Pressures
        cols = self.get_cols_that_ends_with(data, self.CHK)
        data_transformed[cols] = data_transformed[cols] * self.SCALES['CHK']

        # Pressures
        cols = self.get_cols_that_ends_with(data, self.QOIL)
        data_transformed[cols] = data_transformed[cols] * self.SCALES['QOIL']

        # Pressures
        cols = self.get_cols_that_ends_with(data, self.QWAT)
        data_transformed[cols] = data_transformed[cols] * self.SCALES['QWAT']

        return data_transformed

    def fit_transform(self,data):
        return self.transform(data)

    def get_scale(self,type):
        if type in self.PRESSURES:
            return self.SCALES['PRESSURES']
        elif type in self.CHK:
            return self.SCALES['CHK']
        elif type in self.QGAS:
            return self.SCALES['QGAS']
        elif type in self.QOIL:
            return self.SCALES['QOIL']
        elif type in self.QWAT:
            return self.SCALES['QWAT']

    def get_cols_that_ends_with(self,data,endings):
        data_cols=data.columns
        cols_out=[]
        for col in data_cols:
            if col.split('_')[-1] in endings:
                cols_out.append(col)
        return cols_out


class DataContainer:
    def __init__(self,X,Y,name='unnamed'):
        self.name=name
        self.X=X
        self.Y=Y
        self.data_size=X.shape[0]
        self.n_cols=X.shape[1]

        self.X_transformed=None
        self.Y_transformed=None

        self.SCALER=Transformer()

        self.init_transform()

    def init_transform(self):
        self.X_transformed =self.SCALER.transform(self.X)
        self.Y_transformed =self.SCALER.transform(self.Y)

    def inverse_transform(self,data):
        return self.SCALER.inverse_transform(data)

    def transform(self,data):
        return self.SCALER.transform(data)

    def merge(self,data_X,data_Y):
        self.X=pd.concat([self.X,data_X.drop('time',1)],axis=1)
        #(self.X)
        self.Y=pd.concat([self.Y,data_Y],axis=1)
        self.init_transform()
    def __str__(self):

        s=self.name+'\n'
        s+='---------------------------\n'
        if self.params!=None:
            s+='--- Params ---\n'
            for key in self.params:
                s+=key+'\n'
                for param in self.params[key]:
                    s+=param+': '+str(self.params[key][param][0])+'\n'
                s+='\n'

        s+='--- config --- \n'
        s+='Data size: '+str(self.data_size)+'\n'
        s+='N_variables: '+str(self.n_cols)+'\n'
        s+='-------------------------------------\n'
        return s
    def get_scale(self,type):
        return self.SCALER.get_scale(type)



