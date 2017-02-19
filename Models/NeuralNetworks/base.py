from keras.models import Sequential
from keras.layers import Dense, Activation, Merge, Input, merge,Layer,Dropout,MaxoutDense
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
from sklearn import metrics
import keras

INIT='glorot_uniform'
def generate_inception_module(input_layer, n_inception,n_depth, n_width, l2_weight):
    inception_outputs=[]
    for i in range(n_inception):
        out_temp=Dense(n_width, init=INIT, activation='relu', W_regularizer=l2(l2_weight),bias=True)(input_layer)
        for j in range(n_depth-1):
            out_temp = Dense(n_width, init=INIT,activation='relu', W_regularizer=l2(l2_weight),bias=True)(out_temp)
        inception_outputs.append(out_temp)
    output_merged = merge(inception_outputs, mode='concat')
    return output_merged


def add_layers(input_layer,n_depth,n_width,l2_weight):
    if n_depth==0:
        return input_layer
    output_layer=Dense(n_width, activation='relu',init=INIT, W_regularizer=l2(l2_weight),bias=True)(input_layer)
    for i in range(n_depth-1):
        output_layer = Dense(n_width, activation='relu', init=INIT, W_regularizer=l2(l2_weight),bias=True)(output_layer)
    return output_layer


def add_layers_maxout(input_layer,n_depth,n_width,l2_weight):
    if n_depth==0:
        return input_layer
    output_layer = MaxoutDense(n_width, init=INIT, W_regularizer=l2(l2_weight), bias=True)(input_layer)
    # output_layer=Activation('relu')(output_layer)
    for i in range(n_depth - 1):
        output_layer = MaxoutDense(n_width, init=INIT, W_regularizer=l2(l2_weight), bias=True)(output_layer)
        # output_layer=Activation('relu')(output_layer)
    return output_layer


def add_thresholded_output(output_layer,n_input,name):
    aux_input = Input(shape=(1,), dtype='float32', name='OnOff_' + name)
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


#######################################INPUT#############################################################################


def add_OnOff_state_input(X,X_dict,thresholds):
    new_X=X_dict.copy()
    for key in thresholds.keys():
        OnOff_X=np.array([0 if x<=thresholds[key] else 1 for x in X[key+'_CHK']])
        new_X.update({'OnOff_'+key:OnOff_X})
    return new_X


def df2dict(df,input_tags,output_tags,data_type):
    data={}
    if data_type=='X':
        for key in input_tags:
            data[key]=df[input_tags[key]].as_matrix()
    else:
        for key in output_tags:
            data[key] = df[output_tags[key]].as_matrix()
    return data


def find_tag_that_ends_with(lst,end):
    for tag in lst:
        if tag.split('_')[-1]==end:
            return tag
    return False

def layer_to_ordered_tag_list(tags,layers):
    ordered_list=[]
    layer_names=[]
    for layer_name,_,_ in layers:
        layer_names.append(layer_name)
        if layer_name.split('_')[0]!='OnOff':
            for tag in tags[layer_name]:
                ordered_list.append(tag)
    return ordered_list,layer_names
def output_tags_to_index(output_tags,output_layers):
    output_tag_index={}
    output_tag_ordered_list=[]
    i=0
    for layer_name,_,_ in output_layers:
        for tag_name in output_tags[layer_name]:
            output_tag_index[tag_name]=i
            output_tag_ordered_list.append(tag_name)
            i+=1

    n_outputs=i

    return output_tag_index,output_tag_ordered_list,n_outputs


def tags_to_list(tags):

    tags_list=[]
    for key in tags:
        for tag in tags[key]:
            tags_list.append(tag)
    return tags_list


def ends_with(tag,endings):

    return tag.split('_')[-1] in endings
