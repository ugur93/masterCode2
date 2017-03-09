
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


#FOR OIL DATASET
#2x30 Dense
#Maxnorm 4 for all
#nbepoch=100
#batch=64

def abs(x):
    return K.abs(x)
class NCNET1_GJOA2(NN_BASE):



    def __init__(self,maxnorm1=2,maxnorm2=1,maxnorm3=1,n_depth=2,n_width=50):

        self.model_name='NCNET2-QOIL_GAS_depth2_w100_mnconv_32'


        self.output_layer_activation='linear'
        #print(maxnorm_hidden)
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
        #self.maxnorm_hidden=maxnorm_hidden
        #self.maxnorm_out=maxnorm_out
        self.maxnorm=[maxnorm1,maxnorm2,maxnorm3]
        self.l2weight = 0.000001


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
            #self.input_tags['OnOff_'+name]=[name+'_CHK_zero']
        print(self.input_tags)

        OUT='GASs'
        if OUT=='GAS':
            self.output_tags = {


                'C1_out': ['C1_QGAS'],
                'C2_out': ['C2_QGAS'],
                'C3_out': ['C3_QGAS'],
                'C4_out': ['C4_QGAS'],
                'D1_out': ['D1_QGAS'],
                'B3_out': ['B3_QGAS'],
                'B1_out': ['B1_QGAS'],


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

            'GJOA_TOTAL': 1.0,

            'C1_out':  0.0
        }


        super().__init__()

    def initialize_model(self):
        print('Initializing %s' % (self.model_name))

        aux_inputs=[]
        inputs=[]
        merged_outputs=[]
        outputs=[]
        #merged_outputs=[]

        n_depth=self.n_depth
        n_width=self.n_width
        l2w=self.l2weight
        for key in self.well_names:
            n_input=len(self.input_tags[key])
            aux_input,input,merged_out,out=self.generate_input_module(n_depth=n_depth, n_width=n_width,
                                                                    n_input=n_input, n_inception=self.n_inception,
                                                                    l2_weight=l2w, name=key,thresholded_output=self.add_thresholded_output)
            aux_inputs.append(aux_input)
            inputs.append(input)
            merged_outputs.append(merged_out)
            outputs.append(out)


        merged_input = merge(merged_outputs, mode='sum', name='GJOA_TOTAL')

        merged_outputs.append(merged_input)
        all_inputs = inputs

        if self.add_thresholded_output:
            all_inputs+=aux_inputs

        self.model = Model(input=all_inputs, output=merged_outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=self.loss_weights)

    def generate_input_module(self,n_depth, n_width, l2_weight, name, n_input, thresholded_output, n_inception=0):
        K.set_image_dim_ordering('th')
        input_layer = Input(shape=(1,n_input), dtype='float32', name=name)

        #temp_output = UpSampling1D(2)(input_layer)
        #temp_output = Convolution1D(10, 2, activation='relu', border_mode='same', W_constraint=maxnorm(self.maxnorm[0]))(input_layer)
        #temp_output = Convolution1D(20, 2, activation='relu', border_mode='same', W_constraint=maxnorm(self.maxnorm[0]))(temp_output)

        #temp_output=UpSampling1D(5)(temp_output)
        #temp_output=MaxPooling1D(5)(temp_output)

        temp_output = Flatten()(input_layer)

        temp_output = Dropout(0.2)(temp_output)
        temp_output = Dense(self.n_width, activation='relu', W_constraint=maxnorm(self.maxnorm[0]),init=INIT,bias=True)(temp_output)

        #temp_output = Convolution1D(20, 2, activation='relu', border_mode='same',
        #                            W_constraint=maxnorm(self.maxnorm[1]))(temp_output)
        temp_output=Dropout(0.5)(temp_output)
        temp_output = Dense(self.n_width, activation='relu',W_constraint=maxnorm(self.maxnorm[0]), init=INIT,bias=True)(temp_output)
        temp_output = Dropout(0.5)(temp_output)
        #temp_output = Dense(self.n_width, activation='relu', W_constraint=maxnorm(self.maxnorm[1]), init=INIT,bias=True)(temp_output)

        #temp_output = Convolution1D(10, 2, activation='relu', border_mode='same')(temp_output)
        #temp_output = UpSampling1D(2)(temp_output)
        #temp_output = MaxPooling1D(2)(temp_output)

       # temp_output = Dropout(0.5)(temp_output)
        #temp_output = Dense(self.n_width, activation='relu', W_constraint=maxnorm(self.maxnorm_hidden), init=INIT,
        #                    bias=True)(
        #    temp_output)
        #temp_output = Dense(self.n_width, activation='relu', W_constraint=maxnorm(self.maxnorm_hidden), init=INIT,bias=True)(temp_output)
            #temp_output = Dropout(0.1)(temp_output)

        #temp_output = Convolution1D(20, 2, activation='relu', border_mode='same',
        #                            W_constraint=maxnorm(self.maxnorm_hidden))(temp_output)


        #temp_output = Flatten()(temp_output)
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
            output_layer = Dense(1, init=INIT,W_constraint=maxnorm(self.maxnorm[0]),activation=self.output_layer_activation, bias=True)(temp_output)
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
            mod_dense = Dense(self.n_width, activation='relu',W_constraint=maxnorm(self.maxnorm[0]), init=INIT,
                                bias=True)(mod_dense)

        #mod_conv = Dense(10, activation='relu', W_constraint=maxnorm(self.maxnorm[0]), init=INIT,
        #                bias=True)(input_layer)
        mod_conv = Dropout(0.2)(input_layer)
        mod_conv = Dense(40, activation='relu', W_constraint=maxnorm(self.maxnorm[0]), init=INIT,
                         bias=True)(mod_conv)

        mod_conv=Dropout(0.5)(mod_conv)
        mod_conv = Dense(40, activation='relu', W_constraint=maxnorm(self.maxnorm[0]), init=INIT,
                         bias=True)(mod_conv)
        mod_conv = Dropout(0.5)(mod_conv)

        #mod_conv = Convolution1D(20, 2, border_mode='same', activation='relu', W_constraint=maxnorm(self.maxnorm[0]))(
        #    mod_conv)
        #mod_conv = Dense(20, activation='relu', W_constraint=maxnorm(self.maxnorm[0]), init=INIT,
        #                 bias=True)(mod_conv)
        #mod_conv = Convolution1D(50, 2, border_mode='same', activation='relu', W_constraint=maxnorm(self.maxnorm[0]))(mod_conv)
        #mod_conv = Convolution1D(20, 2, border_mode='same', activation='relu',
        #                         W_constraint=maxnorm(self.maxnorm[0]))(mod_conv)
        #mod_conv = Convolution1D(50, 1, border_mode='valid', activation='relu',W_constraint=maxnorm(self.maxnorm_hidden))(mod_conv)
        #mod_conv = LocallyConnected1D(50, 2, border_mode='valid', activation='relu', W_constraint=maxnorm(self.maxnorm_hidden))(mod_conv)
        #mod_conv=UpSampling1D(6)(mod_conv)
        #mod_conv=MaxPooling1D(6)(mod_conv)
        #mod_conv=Dropout(0.1)(mod_conv)


        mod_conv = Flatten()(mod_conv)
        main_model = merge([mod_conv, mod_dense], mode='concat')


        output_layer = Dense(1, init=INIT, W_constraint=maxnorm(self.maxnorm[0]), activation=self.output_layer_activation,
                             bias=True)(main_model)

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


