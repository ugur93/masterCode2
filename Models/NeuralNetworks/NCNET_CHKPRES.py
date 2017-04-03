
from .base import *
from .base_class import NN_BASE

import keras.backend as K


class SSNET3_PRESSURE(NN_BASE):


    def __init__(self,n_depth=2 ,n_width=100,l2w=0.00001,seed=3014):



        self.model_name='GJOA_OIL_WELLS_PDC_MODEL'
        self.out_act='linear'

        # Training config
        optimizer ='adam'
        loss = 'mae'
        nb_epoch = 10000
        batch_size = 64
        dp_rate=0


        chk_names=['C1', 'C2', 'C3', 'C4', 'B1', 'B3', 'D1']
        self.well_names = ['C1', 'C2', 'C3', 'C4', 'B3','B1','D1']
        #self.well_names=['C1','C3', 'C4','B1','B3']


        self.input_tags={'PRESSURE_INPUT':[]}
        for key in chk_names:
            for tag in ['CHK']:
                self.input_tags['PRESSURE_INPUT'].append(key+'_'+tag)
        for key in ['C1','C3', 'C4','B1','B3']:
                self.input_tags['PRESSURE_INPUT'].append(key + '_' + 'PBH')
        #self.input_tags['PRESSURE_INPUT'].append('GJOA_RISER_OIL_B_CHK')
        self.input_tags['RISER_B_CHK_INPUT']=['GJOA_RISER_OIL_B_CHK']

        self.output_tags = {}

        for name in self.well_names:

            #self.output_tags[name + '_PWH_out'] = [name + '_' + 'PWH']
            self.output_tags[name + '_PDC_out'] = [name + '_' + 'PDC']

        self.output_zero_thresholds = {}

        super().__init__(n_width=n_width, n_depth=n_depth, l2_weight=l2w, seed=seed,
                         optimizer=optimizer, loss=loss, nb_epoch=nb_epoch, batch_size=batch_size,dp_rate=dp_rate)


    def initialize_model2(self):
        print('Initializing %s' % (self.model_name))

        chk_input = Input(shape=(len(self.input_tags['Main_input']),), dtype='float32', name='Main_input')

        outputs=[]
        inputs=[chk_input]

        sub_model1 = Dense(20, W_regularizer=l2(self.l2weight), activation='relu')(chk_input)


        for key in self.well_names:
            #sub_model_temp = Dense(20, W_regularizer=l2(self.l2weight), activation='relu')(chk_input)
            #sub_model_temp = Dropout(0.05)(sub_model1)
            sub_model_temp = Dense(len(self.output_tags[key + '_out']),
                                   W_regularizer=l2(self.l2weight), activation=self.out_act, name=key + '_out')(sub_model1)

            aux_input = Input(shape=(len(self.input_tags['aux_' + key]),), dtype='float32', name='OnOff_' + key)

            sub_model_out = merge([sub_model_temp, aux_input], mode='mul', name=key + '_out')
            #sub_model_out = Dense(1, W_regularizer=l2(self.l2weight), activation=self.out_act)(sub_model_out)

            outputs.append(sub_model_out)
            inputs.append(aux_input)

        #sub_model=Dense(20,W_regularizer=l2(self.l2weight),activation='relu')(sub_model)


        #sub_model=Dense(len(self.output_tags['Main_output']),W_regularizer=l2(self.l2weight),name='Main_output')(sub_model)

        self.model = Model(input=inputs, output=outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
    def initialize_model(self):
        print('Initializing %s' % (self.model_name))

        all_chk_input = Input(shape=(len(self.input_tags['PRESSURE_INPUT']),), dtype='float32', name='PRESSURE_INPUT')
        riser_chk_input = Input(shape=(len(self.input_tags['RISER_B_CHK_INPUT']),), dtype='float32', name='RISER_B_CHK_INPUT')

        all_and_riser_chk_input=Concatenate(name='RISER_MERGE')([all_chk_input,riser_chk_input])

        output_layers = {}
        outputs = []
        inputs = [all_chk_input,riser_chk_input]

        #
        #sub_model_all = Dense(50, W_regularizer=l2(self.l2weight), activation='relu',name='pres_sub_all')(all_chk_input)
        #sub_model_all = Dense(50, W_regularizer=l2(self.l2weight), activation='relu',name='pres_sub_all2')(sub_model_all)
        #sub_model_all = Dense(50, W_regularizer=l2(self.l2weight), activation='relu',name='pres_sub_all3')(sub_model_all)



        for key in self.well_names:
            #sub_model_PWH=self.generate_sub_model(all_chk_input,name=key+'_PWH')
            sub_model_PDC=self.generate_sub_model(all_and_riser_chk_input,name=key+'_PDC')

            #PWH_out = Dense(1,
            #                kernel_regularizer=l2(self.l2weight), activation='linear', name=key + '_PWH_out',kernel_initializer=self.init)(sub_model_PWH)

            #
            PDC_out = Dense(1,
                            kernel_regularizer=l2(self.l2weight), activation='linear', name=key + '_PDC_out',kernel_initializer=self.init)(sub_model_PDC)

            #aux_input_PDC = Input(shape=(1,), dtype='float32', name='OnOff_PDC_' + key)
            #aux_input_PWH = Input(shape=(1,), dtype='float32', name='OnOff_PWH_' + key)

            #PWH_out = Multiply(name=key + '_PWH_pred')([PWH_out, aux_input_PWH])
            #PDC_out = Multiply(name=key + '_PDC_pred')([PDC_out, aux_input_PDC])


            #output_layers[key] = PRESSURE_OUT
            #outputs.append(PWH_out)
            outputs.append(PDC_out)

            #inputs.append(aux_input_PDC)
            #inputs.append(aux_input_PWH)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def update_model(self):
        self.nb_epoch=10000
        self.out_act='relu'
        #self.aux_inputs=[]
        #self.inputs=[]
        #self.merged_outputs=[]
        #self.outputs=[]

        old_model=self.model
        self.initialize_model()
        weights=old_model.get_weights()
        self.model.set_weights(weights)

    def generate_sub_model(self,input_layer,name,l2w=0,depth=2):
        i=0
        sub_model = Dense(self.n_width, kernel_regularizer=l2(self.l2weight), activation='relu',name=name+'_'+str(i),kernel_initializer=self.init)(input_layer)

        for i in range(1,depth):
            #sub_model=Dropout(0.1,name=name+'_dp_'+str(i))(sub_model)
            sub_model = Dense(self.n_width,kernel_regularizer=l2(self.l2weight), activation='relu',name=name+'_'+str(i),kernel_initializer=self.init)(sub_model)

        return sub_model