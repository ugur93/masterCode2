
from .base import *
from .base_class import NN_BASE

import keras.backend as K
#Bra:
#self.n_depth = 2
#self.n_width = 20
#self.l2weight =0.0005
#n_depth = 2
#n_width = 50
#l2w = 0.00005
#seed = 9035
#n_depth = 3
#        n_width = 20
#        l2w =0.0004
#        seed=9035
def abs(x):
    return K.abs(x)

SIM=False
class SSNET2(NN_BASE):


    def __init__(self,n_width=90,n_depth=2,l2w=0.0001,dp_rate=0,seed=3014,output_act='relu',n_epoch=10000):


        self.SCALE=100

        self.output_layer_activation =output_act
        # Input module config



        self.input_tags=['CHK','PBH','PWH','PDC']
        #Training config
        optimizer = 'adam'
        loss =huber
        nb_epoch = n_epoch
        batch_size = 64
        dp_rate=0
        self.add_onoff_state=True

        self.model_name='GJOA_GAS_WELLS_QGAS_HUBER_MODEL_FINAL'
        #self.model_name = 'SIM_DATA_WITHOUT_ONOFF'
        #self.model_name = 'GJOA_GAS_WELLS_{}_D{}_W{}_L2{}'.format(loss,n_depth,n_width,l2w)

        if SIM:
            self.output_tags = SIM_OUTPUT_TAGS
            self.well_names = ['A', 'B', 'C', 'D']
            self.loss_weights = {
                'A_out': 0.0,
                'B_out': 0.0,
                'C_out': 0.0,
                'D_out': 0.0,
                'Total_production': 1.0
            }
        else:
            self.output_tags = GAS_WELLS_QGAS_OUTPUT_TAGS
            self.well_names = ['F1', 'B2', 'D3', 'E1']
            self.loss_weights = {
                'F1_out': 0.0,
                'B2_out': 0.0,
                'D3_out': 0.0,
                'E1_out': 0.0,
                'GJOA_QGAS': 1.0
            }


        #

        self.input_tags={}
        tags=['CHK','PBH','PWH','PDC']
        for name in self.well_names:
            self.input_tags[name]=[]
            for tag in tags:
                self.input_tags[name].append(name+'_'+tag)




        super().__init__(n_width=n_width, n_depth=n_depth, l2_weight=l2w, seed=seed,
                         optimizer=optimizer, loss=loss, nb_epoch=nb_epoch, batch_size=batch_size,dp_rate=dp_rate)


    def initialize_model(self):
        print('Initializing %s' % (self.model_name))
        print('Training with params:\n n_width: {}\n n_depth: {}\n l2_weight: {}\n'
              'dp_rate: {}\n seed: {}\n loss: {}\n optimizer: {}\n nb_epoch: {}\n'
              'batch_size: {}\n output_activation: {}'.format(self.n_width, self.n_depth, self.l2weight, self.dp_rate,
                                                              self.seed, self.optimizer, self.loss, self.nb_epoch,
                                                              self.batch_size, self.output_layer_activation))

        aux_inputs=[]
        inputs=[]
        merged_outputs=[]
        outputs=[]

        for key in self.well_names:
            n_input = len(self.input_tags[key])
            aux_input,input,merged_out,out=self.generate_input_module(n_depth=self.n_depth, n_width=self.n_width,
                                                                    n_input=n_input, n_inception=0,
                                                                    l2_weight=self.l2weight, name=key,thresholded_output=self.add_thresholded_output)
            aux_inputs.append(aux_input)
            inputs.append(input)
            merged_outputs.append(merged_out)
            outputs.append(out)

        merged_input = Add( name='GJOA_QGAS')(merged_outputs)

        all_outputs = merged_outputs + [merged_input]
       # merged_outputs.append(merged_input)


        if self.add_onoff_state:
            inputs+=aux_inputs
        self.model = Model(inputs=inputs, outputs=all_outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=self.loss_weights)

    def generate_input_module(self,n_depth, n_width, l2_weight, name, n_input, thresholded_output, n_inception=0):

        input_layer = Input(shape=(n_input,), dtype='float32', name=name)

        #temp_output=BatchNormalization()(input_layer)
        # temp_output=Dropout(0.1)(input_layer)


        temp_output = Dense(self.n_width, activation='relu',kernel_regularizer=l2(self.l2weight), kernel_initializer=self.init,
                            use_bias=True)(input_layer)
        #temp_output=Dropout(0.05)(temp_output)
        for i in range(1,self.n_depth):
            if self.dp_rate>0:
                temp_output = Dropout(self.dp_rate)(temp_output)
            temp_output = Dense(self.n_width, activation='relu', kernel_regularizer=l2(self.l2weight), kernel_initializer=self.init,
                            use_bias=True)(temp_output)
        #temp_output=Dropout(0.05)(temp_output)
        if self.add_onoff_state:
            output_layer = Dense(1, kernel_initializer=self.init,activation=self.output_layer_activation,kernel_regularizer=l2(self.l2weight),use_bias=True)(temp_output)
            aux_input, merged_output = add_thresholded_output(output_layer, n_input, name)
        else:
            output_layer = Dense(1, kernel_initializer=self.init, activation=self.output_layer_activation,
                                 kernel_regularizer=l2(self.l2weight), use_bias=True,name=name+'_out')(temp_output)
            merged_output=output_layer
            aux_input=input_layer
            #aux_input, merged_output = add_thresholded_output(output_layer, n_input, name)

        return aux_input, input_layer, merged_output, output_layer

    def update_model(self,activation='relu',epoch=10000):
        self.nb_epoch=epoch
        self.output_layer_activation=activation
        self.aux_inputs=[]
        self.inputs=[]
        self.merged_outputs=[]
        self.outputs=[]

        old_model=self.model
        self.initialize_model()
        weights=old_model.get_weights()
        self.model.set_weights(weights)

    def load_weights_from_file(self,PATH):
        self.model.load_weights(PATH)



