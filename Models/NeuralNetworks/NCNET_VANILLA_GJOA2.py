
from .base import *
from .base_class import NN_BASE



#GOOD

#incept 3
# dept 1
# width 20
#l2 0.0001
#opt rmsprop
class NCNET_VANILLA(NN_BASE):


    def __init__(self):

        name='NCNET1_VANILLA_GJOA2'


        self.n_inputs=7
        self.n_outputs=1

        # Input module config
        self.n_inception = 0 #(n_inception, n_depth inception)
        self.n_depth = 2
        self.n_width = 40
        self.l2weight = 0.001
        self.add_thresholded_output=True

        self.output_tags = {
            'C1_out': ['C1_QOIL'],
            'C2_out': ['C2_QOIL'],
            'C3_out': ['C3_QOIL'],
            'C4_out': ['C4_QOIL'],
            'D1_out':['D1_QOIL'],
            'B3_out':['B3_QOIL'],
            'B1_out':['B1_QOIL'],
            'GJOA_TOTAL': ['GJOA_TOTAL_QOIL']

        }


        self.input_tags={
            'C1':['C1_CHK'],
            'C2':['C2_CHK'],
            'C3':['C3_CHK'],
            'C4':['C4_CHK'],
            'D1':['D1_CHK'],
            'B3':['B3_CHK'],
            'B1':['B1_CHK']
        }

        self.output_tags={
            'TOTAL_OUT':['GJOA_TOTAL_QOIL']
        }

        well_names=['C1','C2','C3','C4','D1','B3','B1']
        tags=['CHK']
        self.input_tags={'MAIN_INPUT':[]}

        for key in well_names:
            for tag in tags:
                self.input_tags['MAIN_INPUT'].append(key+'_'+tag)


        self.loss_weights=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]

        #Training config
        optimizer = 'rmsprop' #SGD(momentum=0.9,nesterov=True)
        loss = 'mse'
        nb_epoch = 10000
        batch_size = 64
        verbose = 0

        train_params={'optimizer':optimizer,'loss':loss,'nb_epoch':nb_epoch,'batch_size':batch_size,'verbose':verbose}

        super().__init__(name,train_params)

    def initialize_model(self):
        print('Initializing %s' % (self.model_name))

        main_input = Input(shape=(self.n_inputs,), dtype='float32', name='MAIN_INPUT')

        main_model=add_layers(main_input,self.n_depth,self.n_width,self.l2weight)

        out=Dense(1,name='TOTAL_OUT')(main_model)

        self.model = Model(input=main_input, output=out)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def generate_input_module(self,n_depth, n_width, l2_weight, name, n_input, thresholded_output, n_inception=0):

        input_layer = Input(shape=(n_input,), dtype='float32', name=name)
        # temp_output=Dropout(0.1)(input_layer)

        if n_depth == 0:
            temp_output = input_layer
        else:
            if n_inception > 1:
                #temp_output = add_layers(input_layer, 1, n_width, l2_weight)
                temp_output = generate_inception_module(input_layer, n_inception, n_depth, n_width, l2_weight)
                temp_output = add_layers(temp_output, 2, n_width, l2_weight)
                #temp_output = generate_inception_module(temp_output, n_inception, n_depth, n_width, l2_weight)
                #temp_output = add_layers(temp_output, 1, n_width, l2_weight)
            else:
                temp_output = add_layers(input_layer, n_depth, n_width, l2_weight)

        if thresholded_output:
            output_layer = Dense(1)(temp_output)
            # output_layer = Dense(1,init=INIT, W_regularizer=l2(l2_weight), b_regularizer=l2(l2_weight),bias=True)(temp_output)
            aux_input, merged_output = add_thresholded_output(output_layer, n_input, name)
        else:
            output_layer = Dense(1, init=INIT, W_regularizer=l2(l2_weight), b_regularizer=l2(l2_weight), bias=True,
                                 name=name + '_out')(temp_output)

            merged_output = output_layer
            aux_input = input_layer

        return aux_input, input_layer, merged_output, output_layer