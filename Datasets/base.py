
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
class DataContainer:
    def __init__(self,X,Y,Y_Q,params=None,name='unnamed',Y_SCALE=100000):
        #print(params)
        self.name=name
        self.X=X
        self.Y=Y
        self.Y_Q=Y_Q
        self.params=params
        self.data_size=X.shape[0]
        self.n_cols=X.shape[1]


        self.Y_SCALE=Y_SCALE#458376.372582837
        #self.Y_SCALE=100

        self.X_SCALER=FunctionTransformer(func=func_transform,inverse_func=inverse_func_transform,kw_args={'scaler':100},inv_kw_args={'scaler':100})
        self.Y_scaler=FunctionTransformer(func=func_transform,inverse_func=inverse_func_transform,kw_args={'scaler':self.Y_SCALE},inv_kw_args={'scaler':self.Y_SCALE})


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

        self.X_transformed = pd.DataFrame(data=X_scaled, columns=X_cols)

        Y_cols = self.Y.columns

        Y_scaled=self.Y_scaler.transform(self.Y)

        self.Y_transformed = pd.DataFrame(data=Y_scaled, columns=Y_cols)

    def transform_Y_with_new_scale(self,scale=100):
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





