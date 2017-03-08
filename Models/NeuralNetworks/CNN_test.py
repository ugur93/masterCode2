from .base import *
from .base_class import NN_BASE
import keras.backend as K
K.set_image_dim_ordering('th')

import keras.backend as K


class CNN_GJOAOIL(NN_BASE):


    def __init__(self):

        self.model_name='CNN-QOIL'


        self.output_layer_activation='linear'

        # Training config
        self.optimizer = 'adam'#SGD(momentum=0.5,nesterov=True)
        self.loss = 'mse'
        self.nb_epoch = 10000
        self.batch_size = 64
        self.verbose = 0

        #Model config
        self.SCALE=100

        #Input module config
        self.n_inception =0
        self.n_depth = 2
        self.n_depth_incept=1
        self.n_width_incept=20
        self.n_width = 20
        self.l2weight = 0.00001

        self.make_same_model_for_all=True
        self.add_thresholded_output=True

        self.input_tags = {}
        self.well_names = ['C2']#,'C2', 'C3', 'C4','B1','B3','D1']#
        tags = ['CHK','PBH','PWH','PDC']
        self.input_tags['Main_input'] = []
        for name in self.well_names:
            for tag in tags:
                if (name=='C2' or name=='D1') and tag=='PBH':
                    pass
                else:
                    self.input_tags['Main_input'].append(name + '_' + tag)
        print(self.input_tags)




        pressure_tags=['QGAS']
        pressure_outputs=[]
        for name in self.well_names:
            for tag in pressure_tags:

                col=name+'_'+tag
                pressure_outputs.append(col)
        name = self.well_names[0]
        self.output_tags = {

            'MAIN_OUT': [name+'_QOIL'],

        }
        self.loss_weights = {

            'MAIN_OUT': 1.0,
        }



        super().__init__()

    def initialize_model(self):
        print('Initializing %s' % (self.model_name))
        name=self.well_names[0]
        main_input=Input(shape=(1,len(self.input_tags['Main_input'])),dtype='float32',name='Main_input')
        #main_model=GaussianNoise(0.01)(input)


        mod_dense=Dense(50, activation='relu', W_regularizer=l2(self.l2weight))(main_input)
        mod_dense = Dense(50, activation='relu', W_regularizer=l2(self.l2weight))(mod_dense)
        mod_dense=Convolution1D(20,2,activation='relu',border_mode='same')(mod_dense)
        mod_dense = Flatten()(mod_dense)
        #mod_conv=LocallyConnected1D(100,2,border_mode='valid',activation='relu', W_regularizer=l2(self.l2weight))(main_input)
        #mod_conv = LocallyConnected1D(100, 2, border_mode='valid', activation='relu', W_regularizer=l2(self.l2weight))(
        #    mod_conv)
        #mod_conv=MaxPooling1D(2)(mod_conv)
        #mod_conv=Flatten()(mod_conv)

        #main_model=merge([mod_conv,mod_dense],mode='concat')
        aux_input = Input(shape=(1,), dtype='float32', name='OnOff_' + name)


        main_model = Dense(len(self.output_tags['MAIN_OUT']), activation='relu',
                           W_regularizer=l2(self.l2weight))(mod_dense)
        merged_out=merge([aux_input, main_model], mode='mul', name='MAIN_OUT')


        self.model = Model(input=[main_input,aux_input], output=[merged_out])
        self.model.compile(optimizer=self.optimizer, loss=self.loss)


    def generate_input_module(self,n_depth, n_width, l2_weight, name, n_input, thresholded_output, n_inception=0):

        input_layer = Input(shape=(n_input,), dtype='float32', name=name)
        temp_output=GaussianNoise(0.01)(input_layer)
        # temp_output=Dropout(0.1)(input_layer)

        if n_depth == 0 and n_inception==0:
            temp_output = input_layer
        else:
            if n_inception > 1:
                #temp_output = add_layers(input_layer, 1, n_width, l2_weight)
                temp_output = generate_inception_module(input_layer, n_inception, self.n_depth_incept, self.n_width_incept, l2_weight)
                temp_output = add_layers(temp_output, n_depth, n_width, l2_weight)
                #temp_output = generate_inception_module(temp_output, n_inception, n_depth, n_width, l2_weight)
                #temp_output = add_layers(temp_output, 1, n_width, l2_weight)
            else:
                temp_output = add_layers(temp_output, n_depth, n_width, l2_weight)

        if thresholded_output:
            #output_layer = Dense(1, init=INIT, W_regularizer=l2(l2_weight), b_regularizer=l2(l2_weight), bias=True)(
            #    temp_output)
            output_layer = Dense(1,init=INIT,activation=self.output_layer_activation,W_regularizer=l2(l2_weight),b_regularizer=l2(l2_weight),bias=True)(temp_output)
            #output_layer = MaxoutDense(1, init=INIT, W_regularizer=l2(l2_weight),b_regularizer=l2(l2_weight), bias=True)(temp_output)
            #output_layer=Activation('relu')(output_layer)
            aux_input, merged_output = add_thresholded_output(output_layer, n_input, name)
        else:
            output_layer = Dense(1, init=INIT, W_regularizer=l2(l2_weight),b_regularizer=l2(l2_weight), bias=True,
                                 name=name + '_out')(temp_output)

            merged_output = output_layer
            aux_input = input_layer

        return aux_input, input_layer, merged_output, output_layer
