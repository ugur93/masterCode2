
from .base import *
from .base_class import NN_BASE

import keras.backend as K

def sub(inputs):
    s = inputs[0]
    for i in range(1, len(inputs)):
        s -= inputs[i]
    return s
PRES='PWH'

def neg(x):
    return -1*x
class SSNET3_PRESSURE(NN_BASE):


    def __init__(self,Data):



        self.model_name='GJOA_OIL_DO2_'+PRES
        self.out_act='linear'
        # Training config
        self.optimizer = 'adam'  # SGD(momentum=0.9,nesterov=True)
        self.loss = 'mse'
        self.nb_epoch = 20
        self.batch_size = 64
        self.verbose = 0

        self.n_inputs=5
        self.n_outputs=1
        self.SCALE=100000
        # Input module config
        self.n_inception = 0 #(n_inception, n_depth inception)
        self.n_depth = 2
        self.n_width = 20
        self.l2weight = 0.0000
        self.add_thresholded_output=False



        self.input_name='E1'
        self.well_names=['F1','B2','D3','E1']
        chk_names=['C1', 'C2', 'C3', 'C4', 'B1', 'B3', 'D1']
        if PRES=='PBH':
            self.well_names=['C1','C3', 'C4','B1','B3']
        else:
            self.well_names = ['C1', 'C2', 'C3', 'C4', 'B1', 'B3', 'D1']
        tags=['CHK']#,'PBH','PWH']

        self.input_tags={'Main_input':[]}
        for key in chk_names:
            for tag in tags:
                self.input_tags['Main_input'].append(key+'_'+tag)
        self.input_tags['Main_input'].append('GJOA_RISER_OIL_B_CHK')
        self.n_inputs = len(self.input_tags['Main_input'])
        self.n_outputs=1

        self.output_tags = {}
        #self.output_tags['MAIN_out']=[]
        #self.input_tags['aux_pbh'] = []
        #self.input_tags['aux_pwh'] = []
        for name in self.well_names:
            #self.output_tags['MAIN_out'].append(name + '_'+PRES)

            #self.output_tags['MAIN_out'].append(name + '_PBH')
            #self.output_tags['MAIN_out'].append(name + '_PDC')
            #self.output_tags['MAIN_out'].append(name + '_PWH')

            self.output_tags[name + '_out'] = [name + '_'+PRES]
            self.input_tags['aux_' + name] = [name + '_CHK_zero']
        self.initialize_zero_thresholds(Data)
        super().__init__()
    def initialize_model(self):
        print('Initializing %s' % (self.model_name))

        chk_input = Input(shape=(1,len(self.input_tags['Main_input'])), dtype='float32', name='Main_input')
        #
        inputs = [chk_input]
        outputs = []

        for key in self.well_names:
                #sub_model=Dropout(0.5)(chk_input)
                #sub_model=UpSampling1D(2)(chk_input)
                #sub_model=LocallyConnected1D(20,2,activation='relu',W_constraint=maxnorm(1))(sub_model)
                #sub_model=MaxPooling1D(1)(sub_model)
                #sub_model = Dense(20, activation='relu',W_constraint=maxnorm(1))(sub_model)
                sub_model = Flatten()(chk_input)
                #sub_model = Dropout(0.1)(sub_model)
                sub_model = Dense(50, activation='relu', W_constraint=maxnorm(1),b_constraint=maxnorm(1))(sub_model)



                sub_model = Dense(1, W_regularizer=l2(self.l2weight),activation='relu')(sub_model)
                aux_input = Input(shape=(len(self.input_tags['aux_' + key]),), dtype='float32',name='aux_' + key)
                sub_model_out = merge([sub_model, aux_input], mode='mul', name=key + '_out')

                outputs.append(sub_model_out)
                inputs.append(aux_input)

        self.model = Model(input=inputs, output=outputs)
        self.model.compile(optimizer=Adam(), loss=self.loss)

    def update_model(self):
        self.nb_epoch=10000
        self.out_act='linear'
        #self.aux_inputs=[]
        #self.inputs=[]
        #self.merged_outputs=[]
        #self.outputs=[]

        old_model=self.model
        self.initialize_model()
        weights=old_model.get_weights()
        self.model.set_weights(weights)