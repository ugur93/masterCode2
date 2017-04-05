from keras.models import Sequential
from keras.layers import Dense,ThresholdedReLU,UpSampling2D, ZeroPadding1D,GaussianDropout,Activation, Merge,merge, Input,GlobalMaxPooling1D,Layer,Dropout,MaxoutDense,BatchNormalization,GaussianNoise,Convolution1D,MaxPooling1D,Flatten,LocallyConnected1D,UpSampling1D,AveragePooling1D,Convolution2D,MaxPooling2D
from keras.models import Model
from keras.layers.advanced_activations import PReLU
from keras.losses import mean_absolute_error,mean_absolute_percentage_error,mean_squared_error
from keras.layers.merge import add,multiply,Add,concatenate,average,Multiply,Concatenate
from keras.optimizers import Adam
from keras.initializers import glorot_uniform,glorot_normal,RandomUniform
try:
    from keras.utils import plot_model
except(AttributeError):
    print('pydot.find_graphviz() not available, can avoid this problem by commenting out it from visualize_util.py file')

from keras.regularizers import l2,l1

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





OIL_WELLS_QOIL_OUTPUT_TAGS={
                'C1_out': ['C1_QOIL'],
                'C2_out': ['C2_QOIL'],
                'C3_out': ['C3_QOIL'],
                'C4_out': ['C4_QOIL'],
                'D1_out': ['D1_QOIL'],
                'B3_out': ['B3_QOIL'],
                'B1_out': ['B1_QOIL'],
                'GJOA_TOTAL': ['GJOA_TOTAL_SUM_QOIL']
            }

OIL_WELLS_QGAS_OUTPUT_TAGS= {

                'C1_out': ['C1_QGAS'],
                'C2_out': ['C2_QGAS'],
                'C3_out': ['C3_QGAS'],
                'C4_out': ['C4_QGAS'],
                'D1_out': ['D1_QGAS'],
                'B3_out': ['B3_QGAS'],
                'B1_out': ['B1_QGAS'],

                'GJOA_TOTAL': ['GJOA_OIL_QGAS']
            }








SEED=1235
np.random.seed(seed=None)
INIT='glorot_normal'
print(SEED)
#INIT=RandomUniform(minval=-1,maxval=1,seed=1)
bINIT='zeros'
import keras.backend as K
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
def smooth_huber_loss(y_true, y_pred, w):
    """Regression loss function, smooth version of Huber loss function. """
    return K.mean(w * K.log(K.cosh(y_true - y_pred)))


def huber(y_true, y_pred):
    delta=0.1
    diff = y_true - y_pred
    a = 0.5 * (diff**2)
    b = delta * (abs(diff) - delta / 2.0)
    loss = K.switch(abs(diff) <= delta, a, b)
    return loss.sum()
def generate_inception_module(input_layer, n_inception,n_depth, n_width, l2_weight):
    inception_outputs=[]
    for i in range(n_inception):
        out_temp=Dense(n_width, init=INIT, activation='relu', W_regularizer=l2(l2_weight),b_regularizer=l2(l2_weight),bias=True)(input_layer)
        for j in range(n_depth-1):
            out_temp = Dense(n_width, init=INIT,activation='relu', W_regularizer=l2(l2_weight),b_regularizer=l2(l2_weight),bias=True)(out_temp)
        inception_outputs.append(out_temp)
    output_merged = merge(inception_outputs, mode='concat')
    return output_merged

def generate_pressure_sub_model(input_layer,name,init,l2weight,depth,n_width,dp_rate):
        i=0
        sub_model = Dense(n_width, kernel_regularizer=l2(l2weight), activation='relu',name=name+'_'+str(i),kernel_initializer=init)(input_layer)

        for i in range(1,depth):
            if dp_rate>0:
                sub_model=Dropout(dp_rate,name=name+'_dp_'+str(i))(sub_model)
            sub_model = Dense(n_width,kernel_regularizer=l2(l2weight), activation='relu',name=name+'_'+str(i),kernel_initializer=init)(sub_model)

        return sub_model
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
    return aux_input, Multiply(name=name + '_out')([aux_input, output_layer])

class CustomEarlyStopping(Callback):

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
        self.nb_epoch=self.params['epochs']
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
        plot_model(model, to_file='Models/NeuralNetworks/model_figures/'+file_name+'.png', show_shapes=True)
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
        new_X.update({'OnOff_1_' + key: OnOff_X})
        new_X.update({'OnOff_2_' + key: OnOff_X})
        new_X.update({'OnOff_PDC_' + key: OnOff_X})
        new_X.update({'OnOff_PWH_' + key: OnOff_X})
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
        for key in output_tags   :
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
    #print(file_data)
    for line in file_data:
        name,index=line.split(':')
        #print(name,model_name)
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


