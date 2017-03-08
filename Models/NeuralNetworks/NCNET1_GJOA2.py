
from .base import *
from .base_class import NN_BASE

import keras.backend as K
K.set_image_dim_ordering('th')
#GOOD

#incept 3
# dept 1
# width 20
#l2 0.0001
#opt rmsprop

def abs(x):
    return K.abs(x)
class NCNET1_GJOA2(NN_BASE):



    def __init__(self,maxnorm_hidden=1,maxnorm_out=1,n_depth=3,n_width=20,p_dropout1=0.1,p_dropout2=0.1):

        self.model_name='NCNET2-QOIL_GAS_depth2_w100_mnconv'


        self.output_layer_activation='linear'
        print(maxnorm_hidden)
        # Training config
        self.optimizer = 'adam'#SGD(momentum=0.5,nesterov=True)
        self.loss = 'mse'
        self.nb_epoch = 100
        self.batch_size = 64
        self.verbose = 0

        #Model config
        self.SCALE=100

        #Input module config
        self.n_inception =0
        self.n_depth = n_depth
        self.n_depth_incept=1
        self.n_width_incept=20
        self.n_width = n_width
        self.maxnorm_hidden=maxnorm_hidden
        self.maxnorm_out=maxnorm_out
        self.l2weight = 0.00001
        self.p_dropout1=p_dropout1
        self.p_dropout2=p_dropout2

        self.make_same_model_for_all=True
        self.add_thresholded_output=True

        self.input_tags = {}
        self.well_names = ['C1','C2', 'C3', 'C4','B1','B3','D1']#
        tags = ['CHK','PBH','PWH','PDC']
        for name in self.well_names:
            self.input_tags[name] = []
            for tag in tags:
                if (name=='C2' or name=='D1') and tag=='PBH':
                    pass
                else:
                    self.input_tags[name].append(name + '_' + tag)
            self.input_tags['OnOff_'+name]=[name+'_CHK_zero']
            #self.input_tags[name].append('GJOA_RISER_OIL_B_CHK')
        #self.input_tags['A1']=['C1_CHK','C1_PWH','C1_PDC','C1_PBH']
        #self.input_tags['B1']=['B1_CHK']
        #self.input_tags['C1']=['C1_CHK']
        print(self.input_tags)

        OUT='GASs'
        if OUT=='GAS':
            self.output_tags = {
                #'C1_out':['C1_QOIL'],
                #'C2_out':['C2_QOIL'],
                #'C3_out':['C3_QOIL'],
                #'C4_out':['C4_QOIL'],
                #'D1_out':['D1_QOIL'],
                #'B3_out':['B3_QOIL'],
                #'B1_out':['B1_QOIL'],
                #'GJOA_TOTAL': ['GJOA_TOTAL_SUM_QOIL']

                'C1_out': ['C1_QGAS'],
                'C2_out': ['C2_QGAS'],
                'C3_out': ['C3_QGAS'],
                'C4_out': ['C4_QGAS'],
                'D1_out': ['D1_QGAS'],
                'B3_out': ['B3_QGAS'],
                'B1_out': ['B1_QGAS'],

                #'F1_out': ['F1_QGAS'],
                #'B2_out': ['B2_QGAS'],
                #'D3_out': ['D3_QGAS'],
                #'E1_out': ['E1_QGAS'],

                'GJOA_TOTAL':['GJOA_OIL_QGAS']
            }
        else:
            self.output_tags = {
                 'C1_out':['C1_QOIL'],
                 'C2_out':['C2_QOIL'],
                 'C3_out':['C3_QOIL'],
                 'C4_out':['C4_QOIL'],
                 'D1_out':['D1_QOIL'],
                 'B3_out':['B3_QOIL'],
                 'B1_out':['B1_QOIL'],
                 'GJOA_TOTAL': ['GJOA_TOTAL_SUM_QOIL']
            }
        self.loss_weights = {
            'B1_out':  0.0,
            'B3_out':  0.0,
            'C2_out':  0.0,
            'C3_out':  0.0,
            'D1_out':  0.0,
            'C4_out':  0.0,
            #'F1_out': 0.0,
            #'B2_out': 0.0,
            #'D3_out': 0.0,
            #'E1_out': 0.0,
            #'A1_out':0.0,
            'GJOA_TOTAL': 1.0,
            #'Riser_out': 0.0,
            'C1_out':  0.0
        }
        if len(tags)>1:
            self.model_config={
                'B1': (2, 20, 0.0001),
                'B3': (1, 10, 0.0001),
                'C1': (2, 10, 0.0001),
                'C2': (3, 20, 0.0001),
                'C3': (3, 20, 0.00015),
                'C4': (2, 20, 0.0001),
                'D1': (2, 20, 0.0001)
            }
        else:
            self.model_config = {
                'B1': (2, 20, 0.0001),
                'B3': (1, 20, 0.001),
                'C1': (2, 10, 0.0001),
                'C2': (3, 20, 0.0001),
                'C3': (2, 20, 0.0001),
                'C4': (2, 20, 0.0001),
                'D1': (2, 20, 0.0001)
            }


        super().__init__()

    def initialize_model(self):
        print('Initializing %s' % (self.model_name))
        n_depth=self.n_depth
        n_width=self.n_width
        l2w=self.l2weight
        for key in self.well_names:
            if not self.make_same_model_for_all:
                n_depth=self.model_config[key][0]
                n_width=self.model_config[key][1]
                l2w=self.model_config[key][2]
            n_input=len(self.input_tags[key])
            aux_input,input,merged_out,out=self.generate_input_module(n_depth=n_depth, n_width=n_width,
                                                                    n_input=n_input, n_inception=self.n_inception,
                                                                    l2_weight=l2w, name=key,thresholded_output=self.add_thresholded_output)
            self.aux_inputs.append(aux_input)
            self.inputs.append(input)
            self.merged_outputs.append(merged_out)
            self.outputs.append(out)


        merged_input = merge(self.merged_outputs, mode='sum', name='GJOA_TOTAL')

        self.merged_outputs.append(merged_input)
        inputs = self.inputs

        if self.add_thresholded_output:
            inputs+=self.aux_inputs

        self.model = Model(input=inputs, output=self.merged_outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=self.loss_weights)
        #weights=self.model.get_weights()
        #weights=np.normalize(weights)
        #self.model.set_weights(weights)

    def generate_input_module(self,n_depth, n_width, l2_weight, name, n_input, thresholded_output, n_inception=0):
        K.set_image_dim_ordering('th')
        input_layer = Input(shape=(1,n_input), dtype='float32', name=name)


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

                #temp_output=ZeroPadding1D(1)(input_layer)


                #temp_output=MaxPooling1D(2)(temp_output)
                #temp_output = Dropout(0.1)(temp_output)
                #temp_output = Dense(self.n_width, activation='relu', W_regularizer=l2(self.l2weight),init=INIT, bias=True)(temp_output)
                #temp_output = MaxoutDense(self.n_width, W_constraint=maxnorm(self.maxnorm_hidden),init=INIT, bias=True)(temp_output)
                #temp_output = Dense(self.n_width, activation='relu', W_constraint=maxnorm(self.maxnorm_hidden),init=INIT, bias=True)(input_layer)
                #temp_output = Dense(self.n_width, activation='relu', W_regularizer=l2(self.l2weight), init=INIT,bias=True)(input_layer)
                #temp_output = Convolution1D(20, 2, activation='relu', border_mode='same',W_constraint=maxnorm(self.maxnorm_hidden))(input_layer)
                #temp_output = Flatten()(input_layer)
                temp_output = Dense(self.n_width, activation='relu', W_constraint=maxnorm(self.maxnorm_hidden), init=INIT, bias=True)(
                    input_layer)
                for i in range(1,self.n_depth):

                    temp_output = Dense(self.n_width, activation='relu', W_constraint=maxnorm(self.maxnorm_hidden),init=INIT,bias=True)(temp_output)
                    #temp_output = Dense(self.n_width, activation='relu', W_constraint=maxnorm(self.maxnorm_hidden), init=INIT,bias=True)(temp_output)
                    #temp_output = Dropout(0.1)(temp_output)
                #temp_output = Convolution1D(20, 2, activation='relu', border_mode='same',W_constraint=maxnorm(self.maxnorm_hidden))(temp_output)

                #temp_output = MaxPooling1D(2)(temp_output)
                temp_output = Flatten()(temp_output)
                #temp_output = Dropout(0.1)(temp_output)

                #conv33 = Convolution1D(3, 3, border_mode='same', activation='relu',
                #                       W_constraint=maxnorm(self.maxnorm_hidden))(temp_output)
                #conv55 = Convolution1D(5, 5, border_mode='same', activation='relu',
                #                       W_constraint=maxnorm(self.maxnorm_hidden))(temp_output)
                #ap = AveragePooling1D(1)(input_layer)
                #conv11 = Convolution1D(1, 1, border_mode='same', activation='relu',
                #                       W_constraint=maxnorm(self.maxnorm_hidden))(ap)

                #temp_output = merge([conv11, conv33, conv55], mode='concat')




        if thresholded_output:
            #output_layer = Dense(1, init=INIT, W_regularizer=l2(l2_weight), b_regularizer=l2(l2_weight), bias=True)(
            #    temp_output)
            #output_layer = Dense(1,init=INIT, W_constraint=maxnorm(self.maxnorm_hidden),activation=self.output_layer_activation,bias=True)(temp_output)
            output_layer = Dense(1, init=INIT,activation=self.output_layer_activation, bias=True)(temp_output)
            #output_layer = MaxoutDense(1, init=INIT, W_regularizer=l2(l2_weight),b_regularizer=l2(l2_weight), bias=True)(temp_output)
            #output_layer=Activation('relu')(output_layer)
            aux_input, merged_output = add_thresholded_output(output_layer, n_input, name)
        else:
            output_layer = Dense(1, init=INIT, W_regularizer=l2(l2_weight),b_regularizer=l2(l2_weight), bias=True,
                                 name=name + '_out')(temp_output)

            merged_output = output_layer
            aux_input = input_layer

        return aux_input, input_layer, merged_output, output_layer

    def generate_input_module2(self, n_depth, n_width, l2_weight, name, n_input, thresholded_output, n_inception=0):
        K.set_image_dim_ordering('th')
        input_layer = Input(shape=(1,n_input), dtype='float32', name=name)



        mod_dense = Flatten()(input_layer)

        for i in range(0, self.n_depth):
            mod_dense = Dense(self.n_width, activation='relu',W_constraint=maxnorm(self.maxnorm_hidden), init=INIT,
                                bias=True)(mod_dense)

        mod_conv = Dense(20, activation='relu', W_constraint=maxnorm(self.maxnorm_hidden), init=INIT,
                         bias=True)(input_layer)
        mod_conv = Dense(20, activation='relu', W_constraint=maxnorm(self.maxnorm_hidden), init=INIT,
                         bias=True)(mod_conv)
        #mod_conv = Convolution1D(100, 2, border_mode='full', activation='relu', W_constraint=maxnorm(self.maxnorm_hidden))(input_layer)
        #mod_conv = Convolution1D(100, 2, border_mode='full', activation='relu',
        #                         W_constraint=maxnorm(self.maxnorm_hidden))(mod_conv)
        #mod_conv = Convolution1D(50, 1, border_mode='valid', activation='relu',W_constraint=maxnorm(self.maxnorm_hidden))(mod_conv)
        #mod_conv = LocallyConnected1D(50, 2, border_mode='valid', activation='relu', W_constraint=maxnorm(self.maxnorm_hidden))(mod_conv)
        #mod_conv=MaxPooling1D(2)(mod_conv)
        #mod_conv=Dropout(0.1)(mod_conv)

        mod_conv = Flatten()(mod_conv)
        main_model = merge([mod_conv, mod_dense], mode='concat')


        output_layer = Dense(1, init=INIT, W_constraint=maxnorm(self.maxnorm_hidden), activation=self.output_layer_activation,
                             bias=True)(mod_dense)

        aux_input, merged_output = add_thresholded_output(output_layer, n_input, name)


        return aux_input, input_layer, merged_output, output_layer

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
    def update_model_2(self):
        self.nb_epoch=20
        self.output_layer_activation='relu'

        self.loss_weights = {
            'B1_out': 0.5,
            'B3_out': 0.5,
            'C2_out': 0.5,
            'C3_out': 0.5,
            'D1_out': 0.5,
            'C4_out': 0.5,
            # 'F1_out': 0.0,
            # 'B2_out': 0.0,
            # 'D3_out': 0.0,
            # 'E1_out': 0.0,
            # 'A1_out':0.0,
            'GJOA_TOTAL': 1.0,
            # 'Riser_out': 0.0,
            'C1_out': 0.5
        }
        self.aux_inputs=[]
        self.inputs=[]
        self.merged_outputs=[]
        self.outputs=[]

        old_model=self.model
        self.initialize_model()
        weights=old_model.get_weights()
        self.model.set_weights(weights)




class NCNETX_GJOA2(NN_BASE):


    def __init__(self):

        self.model_name='NCNETX_2'

        # Training config
        self.optimizer = 'adam'#SGD(momentum=0.5,nesterov=True)
        self.loss = 'mse'
        self.nb_epoch = 10000
        self.batch_size = 1000
        self.verbose = 0

        #Model config
        self.SCALE=100

        #Input module config
        self.n_inception =0 #(n_inception, n_depth inception)
        self.n_depth = 1
        self.n_depth_incept=2
        self.n_width_incept=20
        self.n_width = 10
        self.l2weight = 0.001
        self.add_thresholded_output=True

        self.input_tags = {}
        input_name='C1'
        self.well_name = ['C1', 'C3', 'C4', 'B3', 'B1','D1','C2']
        tags = ['CHK','PDC','PBH','PWH']
        self.input_tags[input_name] = []
        for name in self.well_name:
            if name!='C1':
                self.input_tags[name]=[name+'_CHK']
            for tag in tags:
                if (name=='C2') and tag=='PBH':
                    #tag='PWH'
                    pass
                else:
                    self.input_tags[input_name].append(name + '_' + tag)
            #self.input_tags[name].append('time')
        print(self.input_tags)

            #self.input_tags[name].append('GJOA_RISER_OIL_B_CHK')

        #self.input_tags['D1']=['D1_CHK','D1_PDC','D1_PWH']
       # self.input_tags['C2'] = ['C2_CHK', 'C2_PDC', 'C2_PWH']
        #self.input_tags['Riser']=['GJOA_RISER_OIL_B_CHK','GJOA_RISER_OIL_B_PDC']


        self.output_tags = {
            'C1_out':['C1_QOIL'],
            'C2_out':['C2_QOIL'],
            'C3_out':['C3_QOIL'],
            'C4_out':['C4_QOIL'],
            'D1_out':['D1_QOIL'],
            'B3_out':['B3_QOIL'],
            'B1_out':['B1_QOIL'],
            #'Riser_out':['GJOA_TOTAL_QOIL_SUM'],
            'GJOA_TOTAL':['GJOA_TOTAL_QOIL_SUM']
        }
        self.loss_weights = {
            'B1_out': 0.0,
            'B3_out': 0.0,
            'C2_out': 0.0,
            'C3_out': 0.0,
            'D1_out': 0.0,
            'C4_out': 0.0,
            'GJOA_TOTAL': 1.0,
            #'Riser_out': 0.0,
            'C1_out': 0.0
        }

        super().__init__()

    def initialize_model(self):
        print('Initializing %s' % (self.model_name))
        main_input = Input(shape=(len(self.input_tags['C1']),), dtype='float32', name='C1')
        for name in self.well_name:
            #n_input=len(self.input_tags[key])
            aux_input,_,merged_out,out=self.generate_input_module(input_layer=main_input,n_depth=self.n_depth, n_width=self.n_width,
                                                                    n_input=0, n_inception=self.n_inception,
                                                                    l2_weight=self.l2weight, name=name,thresholded_output=self.add_thresholded_output)
            self.aux_inputs.append(aux_input)
            #self.inputs.append(input)
            self.merged_outputs.append(merged_out)
            self.outputs.append(out)


        merged_input = merge(self.merged_outputs, mode='sum', name='GJOA_TOTAL')

        self.merged_outputs.append(merged_input)
        inputs = [main_input]

        if self.add_thresholded_output:
            inputs+=self.aux_inputs

        self.model = Model(input=inputs, output=self.merged_outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=self.loss_weights)


    def generate_input_module(self,input_layer,n_depth, n_width, l2_weight, name, n_input, thresholded_output, n_inception=0):


        # temp_output=Dropout(0.1)(input_layer)

        if n_depth == 0:
            temp_output = input_layer
        else:
            if n_inception > 1:
                #temp_output = add_layers(input_layer, 1, n_width, l2_weight)
                temp_output = generate_inception_module(input_layer, n_inception, self.n_depth_incept, self.n_width_incept, l2_weight)
                temp_output = add_layers(temp_output, n_depth, n_width, l2_weight)
                #temp_output = generate_inception_module(temp_output, n_inception, n_depth, n_width, l2_weight)
                #temp_output = add_layers(temp_output, 1, n_width, l2_weight)
            else:
                temp_output = add_layers(input_layer, n_depth, n_width, l2_weight)

        if thresholded_output:
            output_layer = Dense(1,init='glorot_normal',activation='relu',W_regularizer=l2(l2_weight), bias=True)(temp_output)
            # output_layer = Dense(1,init=INIT, W_regularizer=l2(l2_weight), b_regularizer=l2(l2_weight),bias=True)(temp_output)
            aux_input, merged_output = add_thresholded_output(output_layer, n_input, name)
        else:
            output_layer = Dense(1,init='glorot_normal',activation='relu', W_regularizer=l2(l2_weight), bias=True,
                                 name=name + '_out')(temp_output)

            merged_output = output_layer
            aux_input = input_layer

        return aux_input, [], merged_output, output_layer