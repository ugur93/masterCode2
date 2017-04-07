
from .base import *
from .base_class import NN_BASE

import keras.backend as K


class PRESSURE_PDC(NN_BASE):


    def __init__(self,n_depth=4 ,n_width=50,l2w=0.002 ,seed=3014,dp_rate=0):



        self.model_name='GJOA_OIL_WELLS_PDC_MODEL_FINAL'
        self.out_act='linear'

        # Training config
        optimizer ='adam'
        loss = huber
        nb_epoch = 10000
        batch_size = 64
        dp_rate=0


        chk_names=['C1', 'C2', 'C3', 'C4', 'B1', 'B3', 'D1']
        self.well_names = ['C1', 'C2', 'C3', 'C4', 'B3','B1','D1']



        self.input_tags={'PRESSURE_INPUT':[]}
        for key in chk_names:
            for tag in ['CHK']:
                self.input_tags['PRESSURE_INPUT'].append(key+'_'+tag)
        #for key in ['C1','C3', 'C4','B1','B3']:
        #        self.input_tags['PRESSURE_INPUT'].append(key + '_' + 'PBH')
        self.input_tags['RISER_B_CHK_INPUT']=['GJOA_RISER_OIL_B_CHK']

        self.output_tags = {}

        for name in self.well_names:
            self.output_tags[name + '_PDC_out'] = [name + '_' + 'PDC']

        self.output_zero_thresholds = {}

        super().__init__(n_width=n_width, n_depth=n_depth, l2_weight=l2w, seed=seed,
                         optimizer=optimizer, loss=loss, nb_epoch=nb_epoch, batch_size=batch_size,dp_rate=dp_rate)



    def initialize_model(self):
        print('Initializing %s' % (self.model_name))

        all_chk_input = Input(shape=(len(self.input_tags['PRESSURE_INPUT']),), dtype='float32', name='PRESSURE_INPUT')
        riser_chk_input = Input(shape=(len(self.input_tags['RISER_B_CHK_INPUT']),), dtype='float32', name='RISER_B_CHK_INPUT')

        all_and_riser_chk_input=Concatenate(name='RISER_MERGE')([all_chk_input,riser_chk_input])

        output_layers = {}
        outputs = []
        inputs = [all_chk_input,riser_chk_input]






        for key in self.well_names:

            sub_model_PDC=generate_pressure_sub_model(all_and_riser_chk_input,name=key+'_PDC',depth=self.n_depth,
                                                      n_width=self.n_width,dp_rate=self.dp_rate,init=self.init,l2weight=self.l2weight)

            PDC_out = Dense(1,
                            kernel_regularizer=l2(self.l2weight), activation='linear', name=key + '_PDC_out',kernel_initializer=self.init)(sub_model_PDC)


            outputs.append(PDC_out)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def update_model(self):
        self.nb_epoch=10000
        self.out_act='relu'

        old_model=self.model
        self.initialize_model()
        weights=old_model.get_weights()
        self.model.set_weights(weights)




class PRESSURE_PWH(NN_BASE):


    def __init__(self,n_depth=2 ,n_width=80,l2w=0.0003 ,seed=3014,dp_rate=0):



        self.model_name='GJOA_OIL_WELLS_PWH_MODEL2'
        self.out_act='linear'

        # Training config
        optimizer ='adam'
        loss = 'mae'
        nb_epoch = 10000
        batch_size = 64
        dp_rate=0


        chk_names=['C1', 'C2', 'C3', 'C4', 'B1', 'B3', 'D1']
        self.well_names = ['C1', 'C2', 'C3', 'C4', 'B3','B1','D1']



        self.input_tags={'PRESSURE_INPUT':[]}
        for key in chk_names:
            for tag in ['CHK']:
                self.input_tags['PRESSURE_INPUT'].append(key+'_'+tag)
        for key in ['C1','C3', 'C4','B1','B3']:
                self.input_tags['PRESSURE_INPUT'].append(key + '_' + 'PBH')


        self.output_tags = {}

        for name in self.well_names:
            self.output_tags[name + '_PWH_out'] = [name + '_' + 'PWH']

        self.output_zero_thresholds = {}

        super().__init__(n_width=n_width, n_depth=n_depth, l2_weight=l2w, seed=seed,
                         optimizer=optimizer, loss=loss, nb_epoch=nb_epoch, batch_size=batch_size,dp_rate=dp_rate)



    def update_model(self):
        self.nb_epoch=10000
        self.out_act='relu'


        old_model=self.model
        self.initialize_model()
        weights=old_model.get_weights()
        self.model.set_weights(weights)

    def initialize_model(self):
        print('Initializing %s' % (self.model_name))

        all_chk_input = Input(shape=(len(self.input_tags['PRESSURE_INPUT']),), dtype='float32',
                              name='PRESSURE_INPUT')

        outputs = []
        inputs = [all_chk_input]

        for key in self.well_names:
            sub_model_PWH=generate_pressure_sub_model(all_chk_input,name=key+'_PWH',depth=self.n_depth,n_width=self.n_width,dp_rate=self.dp_rate,init=self.init,l2weight=self.l2weight)


            PWH_out = Dense(1,
                            kernel_regularizer=l2(self.l2weight), activation='linear', name=key + '_PWH_out',kernel_initializer=self.init)(sub_model_PWH)


            outputs.append(PWH_out)


        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)



class PRESSURE_PBH(NN_BASE):


    def __init__(self,n_depth=2 ,n_width=100,l2w=0.00001 ,seed=3014,dp_rate=0):



        self.model_name='GJOA_OIL_WELLS_PBH_MODEL2'
        self.out_act='linear'

        # Training config
        optimizer ='adam'
        loss = huber
        nb_epoch = 10000
        batch_size = 64
        dp_rate=dp_rate


        chk_names=['C1', 'C2', 'C3', 'C4', 'B1', 'B3', 'D1']
        self.well_names = ['C1','C3', 'C4','B1','B3']



        self.input_tags={'CHK_INPUT':[]}
        for key in chk_names:
            for tag in ['CHK']:
                self.input_tags['CHK_INPUT'].append(key+'_'+tag)
        #for key in ['C1','C3', 'C4','B1','B3']:
         #       self.input_tags['PRESSURE_INPUT'].append(key + '_' + 'PBH')
        #self.input_tags['CHK_INPUT'].append('GJOA_RISER_OIL_B_CHK')


        self.output_tags = {}

        for name in self.well_names:
            self.output_tags[name + '_PBH_out'] = [name + '_' + 'PBH']

        self.output_zero_thresholds = {}

        super().__init__(n_width=n_width, n_depth=n_depth, l2_weight=l2w, seed=seed,
                         optimizer=optimizer, loss=loss, nb_epoch=nb_epoch, batch_size=batch_size,dp_rate=dp_rate)



    def update_model(self):
        self.nb_epoch=10000
        self.out_act='relu'


        old_model=self.model
        self.initialize_model()
        weights=old_model.get_weights()
        self.model.set_weights(weights)

    def initialize_model(self):
        print('Initializing %s' % (self.model_name))

        all_chk_input = Input(shape=(len(self.input_tags['CHK_INPUT']),), dtype='float32',
                              name='CHK_INPUT')

        outputs = []
        inputs = [all_chk_input]

        for key in self.well_names:
            sub_model_PWH=generate_pressure_sub_model(all_chk_input,name=key+'_PBH',depth=self.n_depth,n_width=self.n_width,dp_rate=self.dp_rate,init=self.init,l2weight=self.l2weight)


            PWH_out = Dense(1,
                            kernel_regularizer=l2(self.l2weight), activation='linear', name=key + '_PBH_out',kernel_initializer=self.init)(sub_model_PWH)


            outputs.append(PWH_out)


        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

