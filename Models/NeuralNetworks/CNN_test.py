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
        self.l2weight = 0.0001

        self.make_same_model_for_all=True
        self.add_thresholded_output=True

        self.input_tags = {}
        self.well_names = ['C1','C2', 'C3', 'C4','B1','B3','D1']#
        tags = ['CHK']#,'PBH','PWH','PDC']
        self.input_tags['Main_input'] = []
        for name in self.well_names:
            for tag in tags:
                if (name=='C2' or name=='D1') and tag=='PBH':
                    pass
                else:
                    self.input_tags['Main_input'].append(name + '_' + tag)
        print(self.input_tags)




        pressure_tags=['PWH']
        pressure_outputs=[]
        for name in self.well_names:
            for tag in pressure_tags:
                #if name=='C2' or name=='D1':
                #    pass
                #else:
                col=name+'_'+tag
                pressure_outputs.append(col)
        self.output_tags = {
            #'C1_out':['C1_QOIL'],
            #'C2_out':['C2_QOIL'],
            #'C3_out':['C3_QOIL'],
            #'C4_out':['C4_QOIL'],
            #'D1_out':['D1_QOIL'],
            #'B3_out':['B3_QOIL'],
            #'B1_out':['B1_QOIL'],
            #'GJOA_TOTAL': ['GJOA_TOTAL_QOIL_SUM']

            #'C1_out': ['C1_QGAS'],
            #'C2_out': ['C2_QGAS'],
            #'C3_out': ['C3_QGAS'],
            #'C4_out': ['C4_QGAS'],
            #'D1_out': ['D1_QGAS'],
            #'B3_out': ['B3_QGAS'],
            #'B1_out': ['B1_QGAS'],

            #'F1_out': ['F1_QGAS'],
            #'B2_out': ['B2_QGAS'],
            #'D3_out': ['D3_QGAS'],
            #'E1_out': ['E1_QGAS'],
            'MAIN_OUT':pressure_outputs,
            'conv':['non']
            #'GJOA_TOTAL':['GJOA_OIL_QGAS']
        }
        self.loss_weights = {
            #'B1_out':  0.0,
            #'B3_out':  0.0,
            #'C2_out':  0.0,
            #'C3_out':  0.0,
            #'D1_out':  0.0,
            #'C4_out':  0.0,
            #'F1_out': 0.0,
            #'B2_out': 0.0,
            #'D3_out': 0.0,
            #'E1_out': 0.0,
            #'A1_out':0.0,
            'MAIN_OUT': 1.0,
            #'Riser_out': 0.0,
            'conv':  0.0
        }



        super().__init__()

    def initialize_model(self):
        print('Initializing %s' % (self.model_name))

        input=Input(shape=(1,len(self.input_tags['Main_input'])),dtype='float32',name='Main_input')
        #main_model=GaussianNoise(0.01)(input)
        #main_model = Dense(20, activation='relu', W_regularizer=l2(self.l2weight))(input)
        #main_model = Dense(20, activation='relu', W_regularizer=l2(self.l2weight))(main_model)
        main_model=Convolution1D(10,10,border_mode='same',activation='linear', W_regularizer=l2(self.l2weight))(input)
        other=Flatten()(main_model)
        other=Dense(1,name='conv')(other)
        #main_model=MaxPooling1D(pool_length=1)(main_model)
        #main_model = Convolution1D(32, 2, border_mode='same', activation='relu', W_regularizer=l2(self.l2weight))(
        #    main_model)
        #main_model = Convolution1D(32, 2, border_mode='same', activation='relu', W_regularizer=l2(self.l2weight))(
        #    main_model)
        #main_model = MaxPooling1D(pool_length=1)(main_model)
        #main_model = Dropout(0.01)(main_model)
        #main_model = Dense(20, activation='relu', W_regularizer=l2(self.l2weight))(main_model)
        #main_model = Convolution1D(32, 2, border_mode='same', activation='relu', W_regularizer=l2(self.l2weight))(main_model)
        #main_model = MaxPooling1D(pool_length=1)(main_model)
        #main_model = Dropout(0.01)(main_model)

        #main_model = Dropout(0.01)(main_model)
        #main_model = Convolution1D(124, 2, border_mode='same', activation='relu')(main_model)
        #main_model = Dropout(0.01)(main_model)
        #main_model = Convolution1D(124, 2, border_mode='same', activation='relu')(main_model)
        #
        #main_model = Dropout(0.1)(main_model)
        #main_model = Convolution1D(64, 2, border_mode='same', activation='relu')(main_model)
        #main_model = MaxPooling1D(pool_length=1)(main_model)
        #main_model = Convolution1D(64, 2, border_mode='same', activation='relu')(main_model)
        #main_model = MaxPooling1D(pool_length=1)(main_model)



        #main_model = Convolution1D(32, 3, border_mode='same', activation='relu')(main_model)
        #main_model=MaxPooling1D(pool_length=1)(main_model)
        #main_model=Dropout(0.1)(main_model)
        #main_model = Convolution1D(len(self.output_tags['MAIN_OUT']), 2, border_mode='same', activation='relu',name='MAIN_OUT')(main_model)
        main_model=Flatten()(main_model)
        #main_model = Dense(20, activation='relu', W_regularizer=l2(self.l2weight))(main_model)
        main_model=Dense(len(self.output_tags['MAIN_OUT']),activation='linear',name='MAIN_OUT',W_regularizer=l2(self.l2weight))(main_model)
        self.model = Model(input=input, output=[main_model,other])
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
