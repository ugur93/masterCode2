
from .base import *
from .base_class import NN_BASE

import keras.backend as K
#PDC: Best params:{'l2w': 0.0021204081632653061, 'seed': 3014, 'n_width': 50, 'n_depth': 4}
#PBH: Best params:{'l2w': 0.0001, 'seed': 3014, 'n_depth': 4, 'n_width': 90}


class PRESSURE(NN_BASE):


    def __init__(self,n_depth=2 ,n_width=100,l2w=0.001,seed=3014,dp_rate=0,tag='PWH',act='relu',n_epoch=10000):


        #PWH: {'l2w': 0.00040000000000000002, 'n_depth': 2, 'n_width': 40, 'seed': 3014}
        #PDC: {'l2w': 0.00020000000000000001, 'n_depth': 2, 'n_width': 40, 'seed': 3014}
        #PBH {'l2w': 5.0000000000000002e-05, 'n_depth': 2, 'n_width': 30, 'seed': 3014}
        #PWH: {'l2w': 6.0000000000000002e-05, 'n_depth': 2, 'n_width': 50, 'seed': 3014}
        self.model_name='GJOA_OIL_WELLS_'+tag
        self.out_act='linear'

        self.tag=tag
        self.type='ALPHA'

        self.n_lookback=10
        do_lookback=False

        # Training config
        optimizer ='adam'
        loss = huber
        nb_epoch = n_epoch
        batch_size = 64
        dp_rate=0


        self.chk_names=['C1', 'C2', 'C3', 'C4', 'B1', 'B3', 'D1']

        if self.tag=='PBH':
            self.well_names = ['C1', 'C3', 'C4', 'B3', 'B1']

        else:

            self.well_names = ['C1', 'C2', 'C3', 'C4', 'B3','B1','D1']

        self.delta_in=False
        self.input_tags={}


        self.input_tags['CHK_INPUT_NOW']=[]
        self.input_tags['CHK_INPUT_PREV'] = []

        if True:
            self.input_tags['CHK_INPUT_NOW'].append('GJOA_RISER_OIL_B_CHK')
            self.input_tags['CHK_INPUT_PREV'].append('GJOA_RISER_OIL_B_shifted_CHK')
        for key in self.chk_names:
            for tag in ['CHK']:
                self.input_tags['CHK_INPUT_NOW'].append(key + '_' + tag)
                self.input_tags['CHK_INPUT_PREV'].append(key + '_shifted_' + tag)

        self.output_tags = {}
        for name in self.well_names:
            self.output_tags[name + '_'+self.tag+'_out'] = [name + '_'+self.tag]

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

        chk_input_prev = Input(shape=(len(self.input_tags['CHK_INPUT_PREV']),), dtype='float32',
                              name='CHK_INPUT_PREV')
        chk_input_now = Input(shape=(len(self.input_tags['CHK_INPUT_NOW']),), dtype='float32',
                          name='CHK_INPUT_NOW')

        chk_input=Concatenate(name='ALL_CHOKES')([chk_input_now,chk_input_prev])
        outputs = []
        inputs = [chk_input_now]

        for key in self.well_names:


            sub_model_PWH=generate_pressure_sub_model(chk_input_now,name=key+'_'+self.tag,depth=self.n_depth,n_width=self.n_width,dp_rate=self.dp_rate,init=self.init,l2weight=self.l2weight)


            PWH_out = Dense(1,
                            kernel_regularizer=l2(self.l2weight), activation='linear', name=key + '_'+self.tag+'_out',
                            kernel_initializer=self.init)(sub_model_PWH)

            outputs.append(PWH_out)



        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        #print(self.model.summary())
        #exit()

class PRESSURE_DELTA(NN_BASE):


    def __init__(self,n_depth=1 ,n_width=100,l2w=0.0001,seed=3014,dp_rate=0,tag='PWH',data='OIL',act='relu',n_epoch=10000):


        #PWH: {'l2w': 0.00040000000000000002, 'n_depth': 2, 'n_width': 40, 'seed': 3014}
        #PDC: {'l2w': 0.00020000000000000001, 'n_depth': 2, 'n_width': 40, 'seed': 3014}
        #PBH {'l2w': 5.0000000000000002e-05, 'n_depth': 2, 'n_width': 30, 'seed': 3014}
        #PWH: {'l2w': 6.0000000000000002e-05, 'n_depth': 2, 'n_width': 50, 'seed': 3014} 
        self.model_name='GJOA_'+data+'_WELLS_'+tag+'_ALL_DATA'
        self.out_act='linear'

        self.tag=tag
        self.type = 'DELTA'

        # Training config
        optimizer ='adam'
        loss =huber
        nb_epoch = n_epoch
        batch_size = 64
        dp_rate=0



        if data=='GAS':
            self.chk_names =['F1','B2','D3','E1']
            self.well_names=['F1','B2','D3','E1']
        else:
            self.chk_names = ['C1', 'C2', 'C3', 'C4', 'B1', 'B3', 'D1']
            if self.tag == 'PBH':
                self.well_names = ['C1', 'C3', 'C4', 'B3', 'B1']
            else:
                self.well_names = ['C1', 'C2', 'C3', 'C4', 'B3', 'B1', 'D1']
        self.delta_in=False
        self.input_tags={}

        if self.delta_in:
            self.input_tags['CHK_DELTA']=[]
            self.input_tags['CHK_DELTA'].append('GJOA_RISER_delta_CHK')
            for key in self.chk_names:
                for tag in ['CHK']:
                    self.input_tags['CHK_DELTA'].append(key + '_delta_' + tag)
        else:
            self.input_tags['CHK_INPUT_NOW']=[]
            self.input_tags['CHK_INPUT_PREV'] = []
            #for key in self.well_names:
            #self.input_tags['CHK_VAL_'+'C1']=[]
            if True:
                self.input_tags['CHK_INPUT_NOW'].append('GJOA_RISER_OIL_B_CHK')
                self.input_tags['CHK_INPUT_PREV'].append('GJOA_RISER_OIL_B_shifted_CHK')
            for key in self.chk_names:
                for tag in ['CHK']:
                    self.input_tags['CHK_INPUT_NOW'].append(key + '_' + tag)
                    #for key2 in self.well_names:
                    #self.input_tags['CHK_VAL_'+key].append(key + '_delta_' + 'PDC')
                    #self.input_tags['CHK_INPUT'].append(key + '_' + tag)
                    #self.input_tags['CHK_INPUT'].append(key + '_delta_' + tag)
                    self.input_tags['CHK_INPUT_PREV'].append(key + '_shifted_' + tag)
                    #self.input_tags['CHK_DELTA'].append(key + '_delta_' + tag)
                    self.input_tags['SHIFTED_PRESSURE_'+self.tag+'_' + key] = [key + '_shifted_'+self.tag]
            #for key in ['C1','C3', 'C4','B1','B3']:
            #        self.input_tags['PRESSURE_INPUT'].append(key + '_' + 'PBH')
            #self.input_tags['PRESSURE_INPUT'].append('time')



        self.output_tags = {}
        #self.output_tags['CHK_DELTA']=['GJOA_RISER_delta_CHK','C1_delta_CHK','C2_delta_CHK','C3_delta_CHK','C4_delta_CHK','B1_delta_CHK','B3_delta_CHK','D1_delta_CHK']
        for name in self.well_names:
            if self.delta_in:
                self.output_tags[name + '_' + self.tag + '_out'] = [name + '_delta_' + self.tag]
            else:
                self.output_tags[name + '_'+self.tag+'_out2'] = [name + '_'+self.tag]
                self.output_tags[name + '_' + self.tag + '_out'] = [name + '_delta_' + self.tag]

        self.output_zero_thresholds = {}

        self.loss_weights={}
        for key in self.output_tags.keys():
            if key.split('_')[-1]=='out2':
                self.loss_weights[key]=1.0
            else:
                self.loss_weights[key]=0.0


        super().__init__(n_width=n_width, n_depth=n_depth, l2_weight=l2w, seed=seed,
                         optimizer=optimizer, loss=loss, nb_epoch=nb_epoch, batch_size=batch_size,dp_rate=dp_rate)

    def update_model(self):
        self.nb_epoch=10000
        self.out_act='relu'


        old_model=self.model
        self.initialize_model()
        weights=old_model.get_weights()
        self.model.set_weights(weights)

    def initialize_model2(self):
        print('Initializing %s' % (self.model_name))


        chk_delta = Input(shape=(len(self.input_tags['CHK_DELTA']),), dtype='float32',
                              name='CHK_DELTA')



        outputs = []
        inputs = [chk_delta]

        for key in self.well_names:
            sub_model_PWH=generate_pressure_sub_model(chk_delta,name=key+'_'+self.tag,depth=self.n_depth,n_width=self.n_width,dp_rate=self.dp_rate,init=self.init,l2weight=self.l2weight)





            #shifted_pressure_input = Input(shape=(len(self.input_tags['SHIFTED_PRESSURE_'+self.tag+'_' + key]),), dtype='float32',
            #                               name='SHIFTED_PRESSURE_'+self.tag+'_' + key)

            PWH_out = Dense(1,
                            kernel_regularizer=l2(self.l2weight), activation='linear', name=key + '_'+self.tag+'_out',
                            kernel_initializer=self.init)(sub_model_PWH)

            #PWH_out = Add(name=key + '_'+self.tag+'_out2')([PWH_out, shifted_pressure_input])
            outputs.append(PWH_out)
            #inputs.append(shifted_pressure_input)


        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        #print(self.model.summary())
        #exit()


    def initialize_model(self):
        print('Initializing %s' % (self.model_name))

        chk_input_now = Input(shape=(len(self.input_tags['CHK_INPUT_NOW']),), dtype='float32',
                              name='CHK_INPUT_NOW')
        chk_input_prev = Input(shape=(len(self.input_tags['CHK_INPUT_PREV']),), dtype='float32',
                              name='CHK_INPUT_PREV')

        chk_delta=Add(name='CHK_DELTA')([chk_input_prev,chk_input_now])

        #chk_delta=ThresholdedReLU(theta=0.001)(chk_delta)


        outputs = []
        inputs = [chk_input_now,chk_input_prev]

        for key in self.well_names:


            sub_model_PWH=generate_pressure_sub_model(chk_delta,name=key+'_'+self.tag,depth=self.n_depth,n_width=self.n_width,dp_rate=self.dp_rate,init=self.init,l2weight=self.l2weight)

            #chk_val = Input(shape=(len(self.input_tags['CHK_VAL_'+key]),), dtype='float32',
            #                name='CHK_VAL_'+key)
            #sub_model_PWH = Concatenate(name='CHK_DELTA_'+key)([sub_model_PWH, chk_val])

            shifted_pressure_input = Input(shape=(len(self.input_tags['SHIFTED_PRESSURE_'+self.tag+'_' + key]),), dtype='float32',
                                           name='SHIFTED_PRESSURE_'+self.tag+'_' + key)

            PWH_out = Dense(1,
                            kernel_regularizer=l2(self.l2weight), activation='linear', name=key + '_'+self.tag+'_out',
                            kernel_initializer=self.init,use_bias=True)(sub_model_PWH)

            PWH_out2 = Add(name=key + '_'+self.tag+'_out2')([PWH_out, shifted_pressure_input])
            outputs.append(PWH_out2)
            outputs.append(PWH_out)
            inputs.append(shifted_pressure_input)
            #inputs.append(chk_val)


        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss,loss_weights=self.loss_weights)


    def initialize_model3(self):
        print('Initializing %s' % (self.model_name))

        chk_input_now = Input(shape=(len(self.input_tags['CHK_INPUT_NOW']),), dtype='float32',
                              name='CHK_INPUT_NOW')
        chk_input_prev = Input(shape=(len(self.input_tags['CHK_INPUT_PREV']),), dtype='float32',
                               name='CHK_INPUT_PREV')

        chk_input=Input(shape=(len(self.input_tags['CHK_INPUT']),), dtype='float32',
                               name='CHK_INPUT')

        #chk_delta = Add(name='CHK_DELTA')([chk_input_prev, chk_input_now])

        outputs = []
        inputs = [chk_input]

        for key in self.well_names:
            prev_pressure_input = Input(shape=(len(self.input_tags['SHIFTED_PRESSURE_' + self.tag + '_' + key]),),
                                           dtype='float32',
                                           name='SHIFTED_PRESSURE_' + self.tag + '_' + key)

            main_input=Concatenate(name='MAIN_INPUT_'+key)([chk_input,prev_pressure_input])

            sub_model_PWH = generate_pressure_sub_model(main_input, name=key + '_' + self.tag, depth=self.n_depth,
                                                        n_width=self.n_width, dp_rate=self.dp_rate, init=self.init,
                                                        l2weight=self.l2weight)



            PWH_out = Dense(1,
                            kernel_regularizer=l2(self.l2weight), activation='linear',
                            name=key + '_' + self.tag + '_out2',
                            kernel_initializer=self.init)(sub_model_PWH)

            #PWH_out2 = Add(name=key + '_' + self.tag + '_out2')([PWH_out, shifted_pressure_input])
            outputs.append(PWH_out)
            # outputs.append(PWH_out)
            inputs.append(prev_pressure_input)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

