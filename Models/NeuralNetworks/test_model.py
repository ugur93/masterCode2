from .base import *
from .base_class import NN_BASE
import keras.backend as K
K.set_image_dim_ordering('th')

import keras.backend as K

class Test_model(NN_BASE):


    def __init__(self):

        self.model_name='Test_pressure_arch'


        self.output_layer_activation='linear'

        # Training config
        self.optimizer = 'adam'#SGD(momentum=0.5,nesterov=True)
        self.loss = 'mse'
        self.nb_epoch = 10000
        self.batch_size =1000
        self.verbose = 0

        #Model config
        self.SCALE=100

        #Input module config
        self.n_inception =0
        self.n_depth = 3
        self.n_depth_incept=1
        self.n_width_incept=20
        self.n_width = 100
        self.l2weight = 0.0001

        self.make_same_model_for_all=True
        self.add_thresholded_output=True

        self.input_tags = {}
        self.well_names = ['C1','C2', 'C3', 'C4','B1','B3','D1']#
        tags = ['CHK']#,'PBH','PWH','PDC']
        self.input_tags['Main_input'] = []
        for name in self.well_names:
            for tag in tags:
                if (name=='C2' or name=='D1'):# and tag=='PBH':
                    pass
                else:
                    self.input_tags['Main_input'].append(name + '_' + tag)
        #self.input_tags['Main_input'].append('time')
        print(self.input_tags)




        pressure_tags=['PW']
        PDC_outputs=[]
        PWH_outputs=[]
        delta_out=[]
        self.input_tags['aux_pbh']=[]
        self.input_tags['aux_pwh'] = []
        names=['C3']

        self.output_tags={}
        for name in self.well_names:

            if name!='C2' and name!='D1':
                #PDC_outputs.append(name+'_PBH')
                #PWH_outputs.append(name+'_PWH')
                self.output_tags[name+'_out']=[name+'_PBH']
                self.input_tags['aux_'+name+'_pbh']=[name + '_CHK_zero']
                #self.input_tags['aux_pwh'].append(name + '_CHK_zero')
                delta_out.append(name+'_c_delta')

        self.loss_weights = {

            'PBH_out':1.0,
            'PWH_out':1.0,
            #'c_delta_out':1.0
        }



        super().__init__()

    def initialize_model(self):
        print('Initializing %s' % (self.model_name))

        chk_input=Input(shape=(len(self.input_tags['Main_input']),),dtype='float32',name='Main_input')
        chk_input_noise=GaussianNoise(0.01)(chk_input)
        #chk_input_noise=Dense(self.n_width,activation='relu',W_regularizer=l2(self.l2weight))(chk_input_noise)

        #chk_input_noise=Convolution1D(20,2,activation='relu',border_mode='same')(chk_input_noise)
        #chk_input_noise=UpSampling1D(2)(chk_input_noise)
        #chk_input_noise=MaxPooling1D(pool_length=2)(chk_input_noise)
        #chk_input_noise=Flatten()(chk_input_noise)
        #chk_input_noise=Dropout(0.05)(chk_input_noise)
        inputs=[chk_input]
        outputs=[]

        for key in self.well_names:
            if key != 'C2' and key != 'D1':
                sub_model=Dense(self.n_width,activation='relu',W_regularizer=l2(self.l2weight))(chk_input_noise)
                #pdc_model = GaussianNoise(0.01)(pdc_model)
                for i in range(self.n_depth):
                    sub_model = Dense(self.n_width, activation='relu', W_regularizer=l2(self.l2weight))(sub_model)
                    #pdc_model = Dense(self.n_width, activation='relu', W_regularizer=l2(self.l2weight))(pdc_model)

                if key == 'B11':
                    sub_model = Dense(len(self.output_tags[key+'_out']), W_regularizer=l2(self.l2weight),name=key+'_out')(sub_model)
                    outputs.append(sub_model)
                else:
                    sub_model = Dense(len(self.output_tags[key + '_out']), W_regularizer=l2(self.l2weight))(sub_model)
                    aux_input=Input(shape=(len(self.input_tags['aux_'+key+'_pbh']),),dtype='float32',name='aux_'+key+'_pbh')
                    sub_model_out=merge([sub_model,aux_input],mode='mul',name=key+'_out')

                    outputs.append(sub_model_out)
                    inputs.append(aux_input)



        #pwh_model=Dense(self.n_width,activation='relu',W_regularizer=l2(self.l2weight))(chk_input_noise)
        #pwh_model = GaussianNoise(0.01)(pwh_model)
        #pwh_model = Dense(self.n_width, activation='relu', W_regularizer=l2(self.l2weight))(pwh_model)
        #pwh_model = Dense(self.n_width, activation='relu', W_regularizer=l2(self.l2weight))(pwh_model)
        #pwh_model = Dense(self.n_width, activation='relu', W_regularizer=l2(self.l2weight))(pwh_model)
        #pwh_model = Dense(self.n_width, activation='relu', W_regularizer=l2(self.l2weight))(pwh_model)
        #pwh_model=Dense(len(self.output_tags['PWH_out']),W_regularizer=l2(self.l2weight))(pwh_model)

        #aux_pwh = Input(shape=(len(self.input_tags['aux_pwh']),), dtype='float32', name='aux_pwh')
        #pwh_merged = merge([pwh_model, aux_pwh], mode='mul',name='PWH_out')

        #main_model=merge([pdc_merged,pwh_model],mode='sum',name='c_delta_out')

        #
        #main_model=Dense(len(self.output_tags['MAIN_OUT']),activation='linear',name='MAIN_OUT',W_regularizer=l2(self.l2weight))(main_model)
        self.model = Model(input=inputs, output=outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

