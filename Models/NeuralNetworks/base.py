from keras.models import Sequential
from keras.layers import Dense, ThresholdedReLU,UpSampling2D, ZeroPadding1D,GaussianDropout,Activation, Merge, Input, merge,GlobalMaxPooling1D,Layer,Dropout,MaxoutDense,BatchNormalization,GaussianNoise,Convolution1D,MaxPooling1D,Flatten,LocallyConnected1D,UpSampling1D,AveragePooling1D,Convolution2D,MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
try:
    from keras.utils.visualize_util import plot
except(AttributeError):
    print('pydot.find_graphviz() not available, can avoid this problem by commenting out it from visualize_util.py file')
from sklearn.svm import SVR
from keras.regularizers import l2

from keras.callbacks import Callback, EarlyStopping
from keras.constraints import nonneg,unitnorm,maxnorm
import numpy as np
import warnings
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
        out_temp=Dense(n_width, init=INIT, activation='relu', W_regularizer=l2(l2_weight),b_regularizer=l2(l2_weight),bias=True)(input_layer)
        for j in range(n_depth-1):
            out_temp = Dense(n_width, init=INIT,activation='relu', W_regularizer=l2(l2_weight),b_regularizer=l2(l2_weight),bias=True)(out_temp)
        inception_outputs.append(out_temp)
    output_merged = merge(inception_outputs, mode='concat')
    return output_merged


def add_layers(input_layer,n_depth,n_width,l2_weight):
    if n_depth==0:
        return input_layer
    output_layer=Dense(n_width, activation='relu',init=INIT,bias=True,W_constraint=maxnorm(1))(input_layer)
    for i in range(n_depth-1):
        #output_layer = BatchNormalization()(output_layer)
        #output_layer = GaussianNoise(0.05)(output_layer)
        output_layer = Dense(n_width, activation='relu', init=INIT, W_constraint=maxnorm(1),bias=True)(output_layer)
    #output_layer = BatchNormalization()(output_layer)
    return output_layer


def add_layers_maxout(input_layer,n_depth,n_width,l2_weight):
    if n_depth==0:
        return input_layer
    output_layer = MaxoutDense(n_width, init=INIT, W_regularizer=l2(l2_weight),b_regularizer=l2(l2_weight), bias=True)(input_layer)
    # output_layer=Activation('relu')(output_layer)
    for i in range(n_depth - 1):
        output_layer = MaxoutDense(n_width, init=INIT, W_regularizer=l2(l2_weight),b_regularizer=l2(l2_weight), bias=True)(output_layer)
        # output_layer=Activation('relu')(output_layer)
    return output_layer


def add_thresholded_output(output_layer,n_input,name):
    aux_input = Input(shape=(1,), dtype='float32', name='OnOff_' + name)
    return aux_input, merge([aux_input, output_layer], mode='mul', name=name + '_out')

class CustomEarlyStopping(Callback):

    """Stop training when a monitored quantity has stopped improving.
    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
    """

    def __init__(self, monitor='val_loss',
                 min_delta=0, patience=0, verbose=0, mode='auto'):
        super(CustomEarlyStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights=0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        self.wait = 0  # Allow instances to be re-used
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Early stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.best_weights=self.model.get_weights()
            self.wait = 0
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)
            self.wait += 1

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping with best loss: %f' % (self.stopped_epoch,self.best))

class EpochVerbose(Callback):

    def on_train_begin(self, logs={}):
        self.nb_epoch=self.params['nb_epoch']
        self.current_epoch=0

    def on_epoch_end(self, epoch, logs={}):
        self.current_epoch+=1
        self.print_status(logs)

    def print_status(self,logs):
        s='On epoch: {0:1d}/{1:1d} -- {2:0.2f}%'.format(self.current_epoch,self.nb_epoch,self.current_epoch/self.nb_epoch*100)
        s+=' -- Loss: {0:0.5f} -- Val_loss: {1:0.5f}'.format(logs.get('loss'),logs.get('val_loss'))
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
    #print(thresholds)
    new_X=X_dict.copy()
    for key in thresholds:
        OnOff_X=np.array([0 if x<thresholds[key] else 1 for x in X[key+'_CHK']])
        OnOff_X=OnOff_X.reshape((len(OnOff_X),1))
        new_X.update({'OnOff_'+key:OnOff_X})
    return new_X
def add_output_threshold_input(X,X_dict,thresholds):
    new_X=X_dict.copy()
    N=len(X)
    for key in thresholds:
        OnOff_X_out_thresh=np.ones((N,1))*thresholds[key]
        new_X.update({'OnOff_ZERO_THRES_'+key:OnOff_X_out_thresh})
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


def layer_config_to_string(sub_network_names,sub_network_config):
    s=' '
    for i in range(len(sub_network_names)):
        s+='\n\n'
        s+=names_to_arrowed_string(sub_network_names[i])
        s+='\n'
        s+=str(sub_network_config[i])
    return s

def names_to_arrowed_string(network_names):
    s=network_names[0]
    for i in range(1,len(network_names)):
        s+=' -> '+network_names[i]
    return s

def find_next_file_number(model_name,file_data):

    next_index='-1'
    print(file_data)
    for line in file_data:
        name,index=line.split(':')
        print(name,model_name)
        if name==model_name:
            next_index=int(index)+1
            new_line=model_name+': '+str(next_index)
            old_line_index=file_data.index(line)
            file_data[old_line_index]=new_line
            break
    if next_index=='-1':
        new_line=model_name+': 1'
        file_data.append(new_line)
        next_index=1
    s=''
    for line in file_data:
        s+=line+'\n'

    return next_index,s


