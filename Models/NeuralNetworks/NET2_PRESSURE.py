
from .base import *
from .base_class import NN_BASE

import keras.backend as K
#Bra:
#self.n_depth = 2
#self.n_width = 20
#self.l2weight =0.0005
def abs(x):
    return K.abs(x)
class SSNET2(NN_BASE):


    def __init__(self):

        self.model_name='NCNET2-QGAS'
        self.SCALE=100

        self.output_layer_activation = 'linear'
        # Input module config
        self.n_inception = 0 #(n_inception, n_depth inception)
        self.n_depth = 2
        self.n_width = 20
        self.l2weight =0.0001
        self.add_thresholded_output=True

        self.input_tags=['CHK','PDC']
        #Training config
        self.optimizer = 'adam' #SGD(momentum=0.9,nesterov=True)
        self.loss = 'mse'
        self.nb_epoch = 2000 #15000
        self.batch_size = 64
        self.verbose = 0

        self.output_tags = {
            'F1_out': ['F1_QGAS'],
            'B2_out': ['B2_QGAS'],
            'D3_out': ['D3_QGAS'],
            'E1_out': ['E1_QGAS'],
            'GJOA_QGAS': ['GJOA_QGAS']
        }
        self.well_names=['F1','B2','D3','E1']

        self.input_tags={}
        tags=['CHK','PDC','PWH','PBH']
        for name in self.well_names:
            self.input_tags[name]=[]
            for tag in tags:
                self.input_tags[name].append(name+'_'+tag)
            #self.input_tags[name].append('time')
        self.loss_weights = {
            'F1_out': 0.0,
            'B2_out': 0.0,
            'D3_out': 0.0,
            'E1_out': 0.0,
            'GJOA_QGAS': 1.0
        }
        super().__init__()

    def generate_input_module(self,n_depth, n_width, l2_weight, name, n_input, thresholded_output, n_inception=0):

        input_layer = Input(shape=(n_input,), dtype='float32', name=name)
        # temp_output=Dropout(0.1)(input_layer)

        if n_depth == 0:
            temp_output = input_layer
        else:
            if n_inception > 1:
                # temp_output = add_layers(input_layer, 1, n_width, l2_weight)
                temp_output = generate_inception_module(input_layer, n_inception, n_depth, n_width, l2_weight)
                #temp_output = add_layers(temp_output, 1, n_width, l2_weight)
                #temp_output = generate_inception_module(temp_output, n_inception, n_depth, n_width, l2_weight)
                #temp_output = add_layers(temp_output, 1, n_width, l2_weight)
            else:
                temp_output = add_layers(input_layer, n_depth, n_width, l2_weight)

        if thresholded_output:
            output_layer = Dense(1, init=INIT, W_regularizer=l2(l2_weight),b_regularizer=l2(l2_weight),bias=True)(temp_output)
            output_layer = Activation(self.output_layer_activation)(output_layer)
            # output_layer = Dense(1,init=INIT, W_regularizer=l2(l2_weight), b_regularizer=l2(l2_weight),bias=True)(temp_output)
            aux_input, merged_output = add_thresholded_output(output_layer, n_input, name)
        else:
            output_layer = Dense(1, init=INIT, W_regularizer=l2(l2_weight), b_regularizer=l2(l2_weight), bias=True,
                                 name=name + '_out')(temp_output)

            merged_output = output_layer
            aux_input = input_layer

        return aux_input, input_layer, merged_output, output_layer

    def initialize_model(self):
        print('Initializing %s' % (self.model_name))

        for key in self.well_names:
            n_input = len(self.input_tags[key])
            aux_input,input,merged_out,out=self.generate_input_module(n_depth=self.n_depth, n_width=self.n_width,
                                                                    n_input=n_input, n_inception=self.n_inception,
                                                                    l2_weight=self.l2weight, name=key,thresholded_output=self.add_thresholded_output)
            self.aux_inputs.append(aux_input)
            self.inputs.append(input)
            self.merged_outputs.append(merged_out)
            self.outputs.append(out)

        merged_input = merge(self.merged_outputs, mode='sum', name='GJOA_QGAS')

        self.merged_outputs.append(merged_input)
        inputs = self.inputs

        if self.add_thresholded_output:
            inputs+=self.aux_inputs
        self.model = Model(input=inputs, output=self.merged_outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=self.loss_weights)
    def update_model(self):
        self.nb_epoch=10000
        self.output_layer_activation='relu'
        self.aux_inputs=[]
        self.inputs=[]
        self.merged_outputs=[]
        self.outputs=[]

        old_model=self.model
        self.initialize_model()
        weights=old_model.get_weights()
        self.model.set_weights(weights)