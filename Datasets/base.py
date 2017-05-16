
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,FunctionTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



CHK_THRESHOLD=5
def negative_values_to_zero(data,tag_name):
    ind = data[tag_name] < 0
    data.loc[ind, tag_name] = 0
    return data

def set_index_values_to_zero(df,ind,col):

    df.loc[ind,col]=0
    return df

def print_rank(X,name):
    print('Rank {}: '.format(name))
    rank = np.linalg.matrix_rank(X)
    if rank == X.shape[1]:
        print('Input has full rank of ' + str(rank))
    else:
        print('Rank of ' + str(rank) + ' is not full rank')


def get_cols_that_ends_with(df,tag):

    cols=[]

    for col in df.columns:
        if col.split('_')[-1]==tag:
            cols.append(col)
    return cols


class CustomScaler:
    def __init__(self,with_minmax=False,with_mean=False, with_std=False,with_mean_from_csv=False,csv_path='',type='X',well_names=[]):
        self.type=type
        self.well_names=well_names
        self.SCALES={
                     'PRESSURES3':100,
                      'PRESSURES2':50,
                     'QGAS':1,
                     'CHK':50,
                     'QOIL':1,
                     'QWAT':1
                     }
        self.InverseScales = {
            'PRESSURES3': 100,
            'PRESSURES2': 50,
            'QGAS': 1,#2000,
            'CHK': 50,
            'QOIL': 0.2,
            'QWAT': 1
        }
    #OIl: 0.2
    #GAS: 2000
        self.TAGS={'PRESSURES3':['PBH','PWH','delta','PDC'],
                   #'PRESSURES2':['PBH'],
                   'QGAS':['QGAS','DEPRECATED'],
                   'CHK':['CHK','time'],
                   'QOIL':['QOIL','SUM'],
                   'QWAT':['QWAT']
                   }

        self.with_mean_from_csv=False
        if with_mean_from_csv and with_mean and len(csv_path)>0:
            self.with_mean_from_csv=True
            self.mean = pd.read_csv(csv_path,squeeze=True,header=None,index_col=0)
        else:
            self.mean=None
        self.std=None

        self.minmax_scale=None
        self.minmax_min=None

        if with_minmax:
            self.with_mean=False
            self.with_std=False
        else:
            self.with_mean=with_mean
            self.with_std=with_std
        self.with_minmax=with_minmax


    def transform(self,data):
        data_transformed=data.copy()
        for tag in self.TAGS:
            cols = self.get_cols_that_ends_with(data, self.TAGS[tag])
            if len(cols)>0:

                if self.with_mean and tag not in ['QGAS','QOIL','PRESSURES']:
                    data_transformed[cols] -= self.mean[cols]
                elif self.with_minmax:
                    data_transformed[cols] -= self.minmax_min[cols]

                #Scaling
                if self.with_std and tag not in ['QGAS','QOIL','PRESSURES']:
                    data_transformed[cols] /= self.std[cols]
                elif self.with_minmax:
                    data_transformed[cols] /= self.minmax_scale[cols]
                else:
                    data_transformed[cols] /=self.SCALES[tag]



        #print(data_transformed)
        return data_transformed

    def change_scale_of_shifted_variables(self):
        for key in self.well_names:
            self.std[key + '_shifted_PDC'] = self.std[key + '_PDC']
            self.std[key + '_shifted_PWH'] = self.std[key + '_PWH']
            self.std[key + '_shifted_PBH'] = self.std[key + '_PBH']
            if self.type == 'X':
                self.std[key + '_shifted_CHK'] = self.std[key + '_CHK']
                self.mean[key + '_shifted_CHK'] = self.mean[key + '_CHK']

            self.mean[key + '_shifted_PDC'] = self.mean[key + '_PDC']
            self.mean[key + '_shifted_PWH'] = self.mean[key + '_PWH']
            self.mean[key + '_shifted_PBH'] = self.mean[key + '_PBH']

           # self.mean[key + '_delta_PDC'] = self.mean[key + '_PDC']
           # self.mean[key + '_delta_PWH'] = self.mean[key + '_PWH']
           # self.mean[key + '_delta_PBH'] = self.mean[key + '_PBH']

    def inverse_transform(self,data):

        data_transformed = data.copy()
        for tag in self.TAGS:
            cols = self.get_cols_that_ends_with(data, self.TAGS[tag])
            if len(cols) > 0:
                #Scaling
                if self.with_std and tag not in ['QGAS','QOIL','PRESSURES']:
                    data_transformed[cols] *= self.std[cols]
                elif self.with_minmax:
                    data_transformed[cols] *= self.minmax_scale[cols]
                else:
                    #print('hereree')
                    data_transformed[cols] *=self.InverseScales[tag]


                if self.with_mean and tag not in ['QGAS','QOIL','PRESSURES']:
                    data_transformed[cols] += self.mean[cols]
                elif self.with_minmax:
                    data_transformed[cols] += self.minmax_min[cols]

        return data_transformed

    def fit_transform(self,data):
        data_transformed = data.copy()
        if not self.with_mean_from_csv:
            self.mean=data_transformed.mean()
            #self.mean.to_csv('Models/NeuralNetworks/ConfigFilesUseful/GJOA_GAS_MEAN.csv')
            #exit()
        self.std = data_transformed.std()


        self.std.replace(0,1,inplace=True)
        #print(self.std)
        self.minmax_scale=data_transformed.max()-data_transformed.min()
        self.minmax_min=data_transformed.min()
        #self.change_scale_of_shifted_variables()


        return self.transform(data)

    def get_scale(self,type):
        for tag in self.TAGS:
            if type in self.TAGS[tag]:
                return self.SCALES[tag]

    def get_cols_that_ends_with(self,data,endings):
        data_cols=data.columns
        cols_out=[]
        for col in data_cols:
            if col.split('_')[-1] in endings:
                cols_out.append(col)
        return cols_out


class DataContainer:
    def __init__(self,X,Y,name='unnamed',csv_path='',well_names=[]):
        self.name=name
        self.X=X
        self.Y=Y

        self.X.fillna(0,inplace=True)
        self.Y.fillna(0,inplace=True)
        self.data_size=X.shape[0]
        self.n_cols=X.shape[1]

        self.X_transformed=None
        self.Y_transformed=None

        self.SCALER_X=CustomScaler(with_mean=True,with_mean_from_csv=False,csv_path=csv_path,with_std=False,with_minmax=False,type='X',well_names=well_names)
        self.SCALER_Y = CustomScaler(with_minmax=False,with_mean=True,with_std=False,type='Y',well_names=well_names)

        self.init_transform()
        #print(np.max(self.X_transformed))
        #print(np.mean(self.X_transformed))
        #exit()
        #self.X_transformed.fillna(0, inplace=True)
        #self.Y_transformed.fillna(0, inplace=True)

    def init_transform(self):
        self.X_transformed =self.SCALER_X.fit_transform(self.X)

        #cols=self.X.columns
        self.Y_transformed =self.SCALER_Y.fit_transform(self.Y)
        #self.X_transformed=pd.DataFrame(data=self.X_transformed,columns=cols)

    def inverse_transform(self,data,scaler):
        if scaler=='X':
            return self.SCALER_X.inverse_transform(data)
        else:
            return self.SCALER_Y.inverse_transform(data)

    def transform(self,data,scaler):
        if scaler=='X':
            return self.SCALER_X.transform(data)
        else:
            return self.SCALER_Y.transform(data)

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

    def get_scale(self,scaler,type):

        if scaler=='X':
            return self.SCALER_X.get_scale(type)
        else:
            return self.SCALER_Y.get_scale(type)

    def get_mean(self,scaler,cols):

        if scaler=='X':
            return self.SCALER_X.mean[cols]
        else:
            return self.SCALER_Y.mean[cols]


