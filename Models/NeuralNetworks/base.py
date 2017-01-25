from keras.models import Sequential
from keras.layers import Dense, Activation, Merge, Input, merge,Layer,Dropout
from keras.models import Model
try:
    from keras.utils.visualize_util import plot
except(AttributeError):
    print('pydot.find_graphviz() not available, can avoid this problem by commenting out it from visualize_util.py file')
from sklearn.svm import SVR
from keras.regularizers import l2

from keras.callbacks import Callback, EarlyStopping
from keras.constraints import nonneg,unitnorm,maxnorm
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from keras.optimizers import SGD


def generate_inception_module(input_layer, n_inception,n_depth, n_width, l2_weight):

    inception_outputs=[]

    for i in range(n_inception):
        out_temp=Dense(n_width, init='glorot_normal', activation='relu', W_regularizer=l2(l2_weight),b_regularizer=l2(l2_weight),bias=True)(input_layer)
        for j in range(n_depth-1):
            out_temp = Dense(n_width, init='glorot_normal',activation='relu', W_regularizer=l2(l2_weight),b_regularizer=l2(l2_weight),bias=True)(out_temp)
        inception_outputs.append(out_temp)
    output_merged = merge(inception_outputs, mode='sum')
    return output_merged


def add_layers(input_layer,n_depth,n_width,l2_weight):
    output_layer=Dense(n_width, activation='relu',init='glorot_normal', W_regularizer=l2(l2_weight),b_regularizer=l2(l2_weight),bias=True)(input_layer)
    #output_layer=Activation('relu')(output_layer)
    for i in range(n_depth-1):
        output_layer = Dense(n_width, activation='relu', init='glorot_normal', W_regularizer=l2(l2_weight),b_regularizer=l2(l2_weight),bias=True)(output_layer)
        #output_layer=Activation('relu')(output_layer)
    return output_layer


def generate_input_module(n_depth, n_width, l2_weight,name, n_input, thresholded_output, n_inception=0):

    input_layer = Input(shape=(n_input,), dtype='float32', name=name)

    if n_depth == 0:
        temp_output = input_layer
    else:
        if n_inception>1:
            temp_output = generate_inception_module(input_layer,n_inception,n_depth,n_width,l2_weight)
            temp_output = add_layers(temp_output, 1, n_width, l2_weight)
            temp_output = generate_inception_module(temp_output, n_inception, n_depth, n_width, l2_weight)
            temp_output = add_layers(temp_output, 1, n_width, l2_weight)
        else:
            temp_output=add_layers(input_layer,n_depth,n_width,l2_weight)

    if thresholded_output:
        output_layer = Dense(1,init='glorot_normal', W_regularizer=l2(l2_weight), b_regularizer=l2(l2_weight),bias=True)(temp_output)
        aux_input, merged_output = add_thresholded_output(output_layer,n_input,name)
    else:
        output_layer = Dense(1,init='glorot_normal', W_regularizer=l2(l2_weight), b_regularizer=l2(l2_weight),bias=True, name=name + '_out')(temp_output)

        merged_output = output_layer
        aux_input = input_layer

    return aux_input,input_layer,merged_output,output_layer


def add_thresholded_output(output_layer,n_input,name):
    aux_input = Input(shape=(1,), dtype='float32', name='aux_' + name)
    return aux_input, merge([aux_input, output_layer], mode='mul', name=name + '_out')


class EpochVerbose(Callback):

    def on_train_begin(self, logs={}):
        self.nb_epoch=self.params['nb_epoch']
        self.current_epoch=0

    def on_epoch_end(self, epoch, logs={}):
        self.current_epoch+=1
        self.print_status(logs)

    def print_status(self,logs):
        s='On epoch: {0:1d}/{1:1d} -- {2:0.2f}%'.format(self.current_epoch,self.nb_epoch,self.current_epoch/self.nb_epoch*100)
        s+=' -- Loss: {0:0.5f}'.format(logs.get('loss'))
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
    try:
        plot(model, to_file='Models/NeuralNetworks/model_figures/'+file_name+'.png', show_shapes=True)
    except(NameError):
        print('Model not plotted')


def addToggledInput(X,thresholds):
    new_X=X.copy()
    for key in X.keys():
        toggled_X=np.array([0 if x<thresholds[key] else 1 for x in X[key][:,0]])
        new_X.update({'aux_'+key:toggled_X})
    return new_X


def df2dict(df,input_tags,data_type):
    data={}
    if data_type=='X':
        for key in input_tags:
            input_data=()
            for i in range(len(input_tags[key])):
                tag=input_tags[key][i]
                if type(tag) is not int:
                    input_data+=(df[tag],)
            data[key]=np.vstack(input_data).T
    else:
        for key in df.columns:
            data[key]=df[key]
    return data

def addDummyOutput(X,Y):

    new_Y=Y.copy()
    for key in X.keys():
        new_Y.update({key+'_out':Y['GJOA_QGAS']})
    return new_Y




