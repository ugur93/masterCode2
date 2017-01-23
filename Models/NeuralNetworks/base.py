from keras.models import Sequential
from keras.layers import Dense, Activation, Merge, Input, merge,Layer,Dropout
from keras.models import Model
from keras.utils.visualize_util import plot
from sklearn.svm import SVR
from keras.regularizers import l2

from keras.callbacks import Callback, EarlyStopping
from keras.constraints import nonneg,unitnorm,maxnorm
from keras import backend as K
import tensorflow as tf
import numpy as np


def generateInceptionModule(input_layer, n_inception,n_depth, n_width, l2_weight):

    inception_outputs=[]

    for i in range(n_inception):
        out_temp=Dense(n_width, activation='relu', W_regularizer=l2(l2_weight))(input_layer)
        for j in range(n_depth-1):
            out_temp = Dense(n_width, activation='relu', W_regularizer=l2(l2_weight))(out_temp)
        inception_outputs.append(out_temp)
    output_merged = merge(inception_outputs, mode='concat')
    #module_output = Dense(1)(output_merged)

    return output_merged

def addLayers(input_layer,n_depth,n_width,l2_weight):
    output_layer=Dense(n_width, activation='relu', W_regularizer=l2(l2_weight))(input_layer)
    for i in range(n_depth-1):
        output_layer = Dense(n_width, activation='relu', W_regularizer=l2(l2_weight))(output_layer)
    return output_layer

def generateInputModule(n_depth,n_width,l2_weight,name,n_input,n_inception=0):
    input_layer = Input(shape=(n_input,), dtype='float32', name=name)
    #input_dropout_layer=Dropout(0.1)(input_layer)
    if n_inception>1:
        temp_output=generateInceptionModule(input_layer,n_inception,n_depth,n_width,l2_weight)
    else:
        temp_output=addLayers(input_layer,n_depth,n_width,l2_weight)
    output_layer=Dense(1,W_constraint=nonneg())(temp_output)

    aux_input=Input(shape=(n_input,), dtype='float32', name='aux-'+name)
    merged_output=merge([aux_input,output_layer],mode='mul')
    return aux_input,input_layer,merged_output,output_layer



class EpochVerbose(Callback):
    def on_train_begin(self, logs={}):
        self.nb_epoch=self.params['nb_epoch']
        self.current_epoch=0
    def on_epoch_end(self, epoch, logs={}):
        self.current_epoch+=1
        self.print_status(logs)
    def print_status(self,logs):
        s='On epoch: {0:1d}/{1:1d} -- {2:0.2f}%'.format(self.current_epoch,self.nb_epoch,self.current_epoch/self.nb_epoch*100)
        s+='-- Loss: {}'.format(logs.get('loss'))
        print('\r', end='', flush=True)
        print(s,end='',flush=True)
    def on_train_end(self, logs={}):
        print('')


class LossHistory(Callback):
    def on_train_begin(self,logs={}):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def plotModel(model,file_name):
    plot(model, to_file='Models/NeuralNetworks/model_figures/'+file_name+'.png', show_shapes=True)


def addToggledInput(input):
    new_input=input.copy()
    for key in input.keys():
        toggled_input=np.array([0 if x==0 else 1 for x in input[key]])
        new_input.update({'aux-'+key:toggled_input})
    return new_input
def dfToDict(df):
    data={}
    for key in df.columns:
        data[key]=df[key]
    return data
