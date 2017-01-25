
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler
import pandas as pd
import numpy as np


def print_rank(X,name):
    print('Rank {}: '.format(name))
    rank = np.linalg.matrix_rank(X)
    if rank == X.shape[1]:
        print('Input has full rank of ' + str(rank))
    else:
        print('Rank of ' + str(rank) + ' is not full rank')


class DataContainer:
    def __init__(self,X,Y,Y_Q,params=None,name='unnamed'):
        #print(params)
        self.name=name
        self.X=X
        self.Y=Y
        self.Y_Q=Y_Q
        self.params=params
        self.data_size=X.shape[0]
        self.n_cols=X.shape[1]

        self.X_SCALER=MaxAbsScaler()
        self.Y_SCALER=MaxAbsScaler()
        self.Y_Q_SCALER=MinMaxScaler()

        self.X_transformed = None
        self.Y_transformed = None
        self.Y_Q_transformed=None


        self.transformed=False
        self.transform()



    def transform(self):
        if not self.transformed:
            X_cols = self.X.columns
            X_scaled = self.X_SCALER.fit_transform(self.X)

            self.X_transformed = pd.DataFrame(data=X_scaled, columns=X_cols)

            Y_cols = self.Y.columns
            Y_scaled = self.Y_SCALER.fit_transform(self.Y)

            self.Y_transformed = pd.DataFrame(data=Y_scaled, columns=Y_cols)

            Y_Q_cols=self.Y_Q.columns
            Y_Q_scaled=self.Y_SCALER.transform(self.Y_Q)
            self.Y_Q_transformed=pd.DataFrame(data=Y_Q_scaled,columns=Y_Q_cols)

            self.transformed=True

    def inverse_transform_using_scaler(self,data,scaler_type):
        if self.transformed:
            if scaler_type=='X':
                return self.X_SCALER.inverse_transform(data)
            else:
                return self.Y_SCALER.inverse_transform(data)
        return None
    def transform_using_scaler(self,data,scaler_type):
        if self.transformed:
            if scaler_type=='X':
                return self.X_SCALER.transform(data)
            else:
                return self.Y_SCALER.transform(data)
        return None
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





