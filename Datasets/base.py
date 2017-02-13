
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

        self.SCALES={'PRESSURES':100,'GAS':100000,'CHK':100,'QOIL':100,'QWAT':100}
    def transform(self,data):
        data_transformed=data.copy()
        #Pressures
        cols=self.get_cols_that_ends_with(data,self.PRESSURES)
        data_transformed[cols]=data_transformed[cols]/self.SCALES['PRESSURES']
        print(cols)
        # GAS
        cols = self.get_cols_that_ends_with(data, self.QGAS)
        data_transformed[cols]= data_transformed[cols] / self.SCALES['GAS']
        print(cols)

        # Pressures
        cols = self.get_cols_that_ends_with(data, self.CHK)
        data_transformed[cols]= data_transformed[cols] / self.SCALES['CHK']
        print(cols)
        # Pressures
        cols = self.get_cols_that_ends_with(data, self.QOIL)
        data_transformed[cols]= data_transformed[cols] / self.SCALES['QOIL']
        #print(data_transformed[cols])
        print(cols)
        # Pressures
        cols = self.get_cols_that_ends_with(data, self.QWAT)
        data_transformed[cols]= data_transformed[cols] / self.SCALES['QWAT']

        return data_transformed
    def inverse_transform(self,data):
        data_transformed = data.copy()
        # Pressures
        cols = get_cols_that_ends_with(data, self.PRESSURES)
        data_transformed = data_transformed[cols] * self.SCALES['PRESSURES']

        # GAS
        cols = get_cols_that_ends_with(data, self.QGAS)
        data_transformed = data_transformed[cols] * self.SCALES['GAS']

        # Pressures
        cols = get_cols_that_ends_with(data, self.CHK)
        data_transformed = data_transformed[cols] * self.SCALES['CHK']

        # Pressures
        cols = get_cols_that_ends_with(data, self.QOIL)
        data_transformed = data_transformed[cols] * self.SCALES['QOIL']

        # Pressures
        cols = get_cols_that_ends_with(data, self.QWAT)
        data_transformed = data_transformed[cols] * self.SCALES['QWAT']

        return data_transformed
    def fit_transform(self,data):
        return self.transform(data)
    def get_scale(self,type):
        return self.SCALES[type]


    def get_cols_that_ends_with(self,data,endings):
        data_cols=data.columns
        cols_out=[]
        for col in data_cols:
            if col.split('_')[-1] in endings:
                cols_out.append(col)
        return cols_out


class DataContainer:
    def __init__(self,X,Y,Y_Q,params=None,name='unnamed',X_SCALE=100,Y_SCALE=100000):
        #print(params)
        self.name=name
        self.X=X
        self.Y=Y
        self.Y_Q=Y_Q
        self.params=params
        self.data_size=X.shape[0]
        self.n_cols=X.shape[1]


        self.Y_SCALE=Y_SCALE#458376.372582837
        self.X_SCALE=X_SCALE
        self.Y_SCALE=100

        self.X_SCALER=Transformer()#FunctionTransformer(func=func_transform,inverse_func=inverse_func_transform,kw_args={'scaler':self.X_SCALE},inv_kw_args={'scaler':self.X_SCALE})
        self.Y_scaler=Transformer()#FunctionTransformer(func=func_transform,inverse_func=inverse_func_transform,kw_args={'scaler':self.Y_SCALE},inv_kw_args={'scaler':self.Y_SCALE})


        #self.Y_scaler=MinMaxScaler()


        self.X_transformed = None
        self.Y_transformed = None
        self.Y_Q_transformed=None

        self.transformed=False
        self.init_transform()

        #print(self.Y.columns)
        #print(self.Y_scaler.scale_)
        #print(1/2.18161332e-06)



    def init_transform(self):
        X_cols = self.X.columns
        X_scaled = self.X_SCALER.transform(self.X)

        self.X_transformed =pd.DataFrame(data=X_scaled, columns=X_cols)

        Y_cols = self.Y.columns

        Y_scaled=self.Y_scaler.transform(self.Y)

        self.Y_transformed = pd.DataFrame(data=Y_scaled, columns=Y_cols)

    def transform_Y_with_new_scale(self,scale=100):
        print('adadsdsadasd')
        print(scale)
        self.Y_scaler=FunctionTransformer(func=func_transform,inverse_func=inverse_func_transform,
                                          kw_args={'scaler':scale},inv_kw_args={'scaler':scale})
        self.Y_SCALE=scale
        Y_cols = self.Y.columns

        Y_scaled = self.Y_scaler.transform(self.Y)

        self.Y_transformed = pd.DataFrame(data=Y_scaled, columns=Y_cols)

    def inverse_transform(self,data,scaler_type):
        if scaler_type=='X':
            return self.X_SCALER.inverse_transform(data)
        else:
            return self.Y_SCALER.inverse_transform(data)

    def transform(self,data,scaler_type):
        if scaler_type=='X':
            return self.X_SCALER.transform(data)
        else:
            return self.Y_SCALER.transform(data)
    def merge(self,data,type):
        if type=='X':
            self.X=pd.concat([self.X,data],axis=1)
        else:
            self.Y=pd.concat([self.Y,data],axis=1)
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

    def get_scale(self,type,scaler):
        if scaler=='Y':
            return self.Y_scaler.get_scale(type)
        return self.X_SCALER.get_scale(type)



