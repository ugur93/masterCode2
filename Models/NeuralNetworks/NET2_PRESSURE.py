
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

        self.model_name='GJOA_GAS_WELLS_2017_PRETRAINED_OLD_DATA2'
        self.SCALE=100

        self.output_layer_activation = 'linear'
        # Input module config
        self.n_depth = 2
        self.n_width = 50
        self.l2weight =0.0001
        self.add_thresholded_output=True

        self.input_tags=['CHK','PDC']
        #Training config
        self.optimizer = 'adam'
        self.loss = 'mae'
        self.nb_epoch = 1
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
        tags=['CHK','PWH','PBH','PDC']
        for name in self.well_names:
            self.input_tags[name]=[]
            for tag in tags:
                self.input_tags[name].append(name+'_'+tag)
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


        temp_output = Dense(self.n_width, activation='relu',W_regularizer=l2(self.l2weight), init=INIT,
                            bias=True)(input_layer)
        #temp_output=Dropout(0.05)(temp_output)
        temp_output = Dense(self.n_width, activation='relu', W_regularizer=l2(self.l2weight), init=INIT,
                           bias=True)(temp_output)
        #temp_output=Dropout(0.05)(temp_output)

        output_layer = Dense(1, init=INIT, activation=self.output_layer_activation,W_regularizer=l2(self.l2weight),bias=True)(temp_output)
        aux_input, merged_output = add_thresholded_output(output_layer, n_input, name)


        return aux_input, input_layer, merged_output, output_layer

    def initialize_model(self):
        print('Initializing %s' % (self.model_name))

        aux_inputs=[]
        inputs=[]
        merged_outputs=[]
        outputs=[]

        for key in self.well_names:
            n_input = len(self.input_tags[key])
            aux_input,input,merged_out,out=self.generate_input_module(n_depth=self.n_depth, n_width=self.n_width,
                                                                    n_input=n_input, n_inception=0,
                                                                    l2_weight=self.l2weight, name=key,thresholded_output=self.add_thresholded_output)
            aux_inputs.append(aux_input)
            inputs.append(input)
            merged_outputs.append(merged_out)
            outputs.append(out)

        merged_input = merge(merged_outputs, mode='sum', name='GJOA_QGAS')

        merged_outputs.append(merged_input)


        if self.add_thresholded_output:
            inputs+=aux_inputs
        self.model = Model(input=inputs, output=merged_outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=self.loss_weights)
    def update_model(self):
        self.nb_epoch=5000
        self.output_layer_activation='relu'
        self.aux_inputs=[]
        self.inputs=[]
        self.merged_outputs=[]
        self.outputs=[]

        old_model=self.model
        self.initialize_model()
        weights=old_model.get_weights()
        self.model.set_weights(weights)

    def load_weights_from_file(self,PATH):
        self.model.load_weights(PATH)


    def generate_input_module_incept(self, n_depth, n_width, l2_weight, name, n_input, thresholded_output,
                              n_inception=0):
        K.set_image_dim_ordering('th')
        input_layer = Input(shape=(n_input,), dtype='float32', name=name)

        #mod_dense = Flatten()(input_layer)
        mod_dense = Dense(self.n_width, activation='relu',W_regularizer=l2(self.l2weight), init=INIT,
                          bias=True)(input_layer)
        #for i in range(1, self.n_depth):
        #    mod_dense = Dense(self.n_width, activation='relu',W_regularizer=l2(self.l2weight), init=INIT,
        #                      bias=True)(mod_dense)

        #mod_conv = Dropout(0.1)(input_layer)

        mod_conv = Dense(20, activation='relu', init=INIT,
                         bias=True,W_regularizer=l2(self.l2weight))(input_layer)

        mod_conv = Dropout(0.1)(mod_conv)

        #mod_conv = Dense(20, activation='relu', init=INIT,
        #                 bias=True,W_regularizer=l2(self.l2weight))(mod_conv)

        #mod_conv = Dropout(0.1)(mod_conv)

        #mod_conv = Flatten()(mod_conv)
        main_model = merge([mod_conv, mod_dense], mode='concat')

        output_layer = Dense(1, init=INIT,W_regularizer=l2(self.l2weight),
                             activation=self.output_layer_activation,
                             bias=True)(main_model)

        aux_input, merged_output = add_thresholded_output(output_layer, n_input, name)

        return aux_input, input_layer, merged_output, output_layer
