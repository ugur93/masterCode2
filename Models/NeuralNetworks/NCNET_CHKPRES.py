
from .base import *
from .base_class import NN_BASE



def sub(inputs):
    s = inputs[0]
    for i in range(1, len(inputs)):
        s -= inputs[i]
    return s

class SSNET3_PRESSURE(NN_BASE):


    def __init__(self):

        self.model_name='GJOA_GAS_WELLS_CHK_PRES'
        # Training config
        self.optimizer = 'adam'  # SGD(momentum=0.9,nesterov=True)
        self.loss = 'mse'
        self.nb_epoch = 30000
        self.batch_size = 1000
        self.verbose = 0

        self.n_inputs=5
        self.n_outputs=1
        self.SCALE=100000
        # Input module config
        self.n_inception = 0 #(n_inception, n_depth inception)
        self.n_depth = 2
        self.n_width = 20
        self.l2weight = 0.0001
        self.add_thresholded_output=False



        self.input_name='E1'
        self.well_names=['F1','B2','D3','E1']
        tags=['CHK']#,'PDC','PBH','PWH']

        self.input_tags={'Main_input':[]}
        for key in self.well_names:
            for tag in tags:
                self.input_tags['Main_input'].append(key+'_'+tag)

        self.n_inputs = len(self.input_tags['Main_input'])
        self.n_outputs=1

        self.output_tags = {}
        #self.input_tags['aux_pbh'] = []
        #self.input_tags['aux_pwh'] = []
        for name in self.well_names:
            self.output_tags[name + '_out'] = [name + '_PWH']
            self.input_tags['aux_' + name] = [name + '_CHK_zero']

        super().__init__()
    def initialize_model(self):
        print('Initializing %s' % (self.model_name))

        chk_input = Input(shape=(1,len(self.input_tags['Main_input'])), dtype='float32', name='Main_input')
        chk_input_noise = BatchNormalization()(chk_input)
        #chk_input_noise=Dense(self.n_width,activation='relu',W_regularizer=l2(self.l2weight))(chk_input_noise)

        # chk_input_noise=Convolution1D(20,2,activation='relu',border_mode='same')(chk_input_noise)
        # chk_input_noise=UpSampling1D(2)(chk_input_noise)
        # chk_input_noise=MaxPooling1D(pool_length=2)(chk_input_noise)
        # chk_input_noise=Flatten()(chk_input_noise)
        # chk_input_noise=Dropout(0.05)(chk_input_noise)
        inputs = [chk_input]
        outputs = []

        for key in self.well_names:

                sub_model = Dense(self.n_width, activation='relu', W_regularizer=l2(self.l2weight))(chk_input_noise)
                # pdc_model = GaussianNoise(0.01)(pdc_model)
                for i in range(1,self.n_depth):
                    sub_model = Dense(self.n_width, activation='relu', W_regularizer=l2(self.l2weight))(sub_model)
                    # pdc_model = Dense(self.n_width, activation='relu', W_regularizer=l2(self.l2weight))(pdc_model)

                sub_model=Convolution1D(20,2,activation='relu',border_mode='full')(sub_model)
                sub_model=UpSampling1D(3)(sub_model)
                sub_model=MaxPooling1D(pool_length=3)(sub_model)
                sub_model=Flatten()(sub_model)
                #sub_model=Dropout(0.01)(sub_model)
                sub_model = Dense(len(self.output_tags[key + '_out']), W_regularizer=l2(self.l2weight))(sub_model)
                aux_input = Input(shape=(len(self.input_tags['aux_' + key]),), dtype='float32',
                                  name='aux_' + key)
                sub_model_out = merge([sub_model, aux_input], mode='mul', name=key + '_out')

                outputs.append(sub_model_out)
                inputs.append(aux_input)

        # pwh_model=Dense(self.n_width,activation='relu',W_regularizer=l2(self.l2weight))(chk_input_noise)
        # pwh_model = GaussianNoise(0.01)(pwh_model)
        # pwh_model = Dense(self.n_width, activation='relu', W_regularizer=l2(self.l2weight))(pwh_model)
        # pwh_model = Dense(self.n_width, activation='relu', W_regularizer=l2(self.l2weight))(pwh_model)
        # pwh_model = Dense(self.n_width, activation='relu', W_regularizer=l2(self.l2weight))(pwh_model)
        # pwh_model = Dense(self.n_width, activation='relu', W_regularizer=l2(self.l2weight))(pwh_model)
        # pwh_model=Dense(len(self.output_tags['PWH_out']),W_regularizer=l2(self.l2weight))(pwh_model)

        # aux_pwh = Input(shape=(len(self.input_tags['aux_pwh']),), dtype='float32', name='aux_pwh')
        # pwh_merged = merge([pwh_model, aux_pwh], mode='mul',name='PWH_out')

        # main_model=merge([pdc_merged,pwh_model],mode='sum',name='c_delta_out')

        #
        # main_model=Dense(len(self.output_tags['MAIN_OUT']),activation='linear',name='MAIN_OUT',W_regularizer=l2(self.l2weight))(main_model)
        self.model = Model(input=inputs, output=outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

