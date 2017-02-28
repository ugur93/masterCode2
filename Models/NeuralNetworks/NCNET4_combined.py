
from .base import *
from .base_class import NN_BASE



def sub(inputs):
    s = inputs[0]
    for i in range(1, len(inputs)):
        s -= inputs[i]
    return s

class NET4_COMBINED(NN_BASE):


    def __init__(self):

        self.model_name='Combined'

        self.output_layer_activation = 'linear'
        # Training config
        self.optimizer = 'adam'  # SGD(momentum=0.9,nesterov=True)
        self.loss = 'mse'
        self.nb_epoch = 10000
        self.batch_size = 64
        self.verbose = 0

        self.n_inputs=5
        self.n_outputs=1
        self.SCALE=100000
        # Input module config
        self.n_inception = 0 #(n_inception, n_depth inception)
        self.n_depth = 1
        self.n_conv=1
        self.n_width = 20
        self.l2weight = 0.0001
        self.add_thresholded_output=True
        self.n_out=1



        self.input_tags = {}
        self.well_names = ['C1', 'C2', 'C3', 'C4', 'B1', 'B3', 'D1']  #
        tags = ['CHK', 'PBH', 'PWH', 'PDC']
        for name in self.well_names:
            self.input_tags[name] = []
            for tag in tags:
                if (name == 'C2' or name == 'D1') and tag == 'PBH':
                    pass
                else:
                    self.input_tags[name].append(name + '_' + tag)
                    # self.input_tags[name].append('GJOA_RISER_OIL_B_CHK')
        # self.input_tags['A1']=['C1_CHK','C1_PWH','C1_PDC','C1_PBH']
        # self.input_tags['B1']=['B1_CHK']
        # self.input_tags['C1']=['C1_CHK']
        print(self.input_tags)


        self.output_tags = {
            'C1_OIL':['C1_QOIL'],
            'C2_OIL':['C2_QOIL'],
            'C3_OIL':['C3_QOIL'],
            'C4_OIL':['C4_QOIL'],
            'D1_OIL':['D1_QOIL'],
            'B3_OIL':['B3_QOIL'],
            'B1_OIL':['B1_QOIL'],
            'GJOA_OIL': ['GJOA_TOTAL_SUM_QOIL'],

            'C1_GAS': ['C1_QGAS'],
            'C2_GAS': ['C2_QGAS'],
            'C3_GAS': ['C3_QGAS'],
            'C4_GAS': ['C4_QGAS'],
            'D1_GAS': ['D1_QGAS'],
            'B3_GAS': ['B3_QGAS'],
            'B1_GAS': ['B1_QGAS'],

            'GJOA_GAS': ['GJOA_OIL_SUM_QGAS']
        }

        self.loss_weights = {
            'C1_OIL':0.0,
            'C2_OIL':0.0,
            'C3_OIL':0.0,
            'C4_OIL':0.0,
            'D1_OIL':0.0,
            'B3_OIL':0.0,
            'B1_OIL':0.0,
            'GJOA_OIL': 1.0,

            'C1_GAS': 0.0,
            'C2_GAS': 0.0,
            'C3_GAS': 0.0,
            'C4_GAS': 0.0,
            'D1_GAS': 0.0,
            'B3_GAS': 0.0,
            'B1_GAS': 0.0,

            'GJOA_GAS': 1.0
        }
        print('HEREEE')
        super().__init__()
    def initialize_model(self):
        print('Initializing %s' % (self.model_name))
        n_depth=self.n_depth
        n_width=self.n_width
        l2w=self.l2weight


        aux_inputs=[]
        sub_inputs=[]
        sub_outputs_gas=[]
        sub_outputs_oil=[]

        for key in self.well_names:
            n_input=len(self.input_tags[key])
            aux_input,sub_input,sub_output_gas,sub_output_oil=self.generate_input_module(n_depth=n_depth, n_width=n_width,
                                                                    n_input=n_input, n_inception=self.n_inception,
                                                                    l2_weight=l2w, name=key,thresholded_output=self.add_thresholded_output)
            aux_inputs.append(aux_input)
            sub_inputs.append(sub_input)
            sub_outputs_gas.append(sub_output_gas)
            sub_outputs_oil.append(sub_output_oil)
            #self.outputs.append(out)


        merged_gas = merge(sub_outputs_gas, mode='sum', name='GJOA_GAS')
        merged_oil = merge(sub_outputs_oil, mode='sum', name='GJOA_OIL')

        all_outputs=[merged_gas,merged_oil]+sub_outputs_gas+sub_outputs_oil


        if self.add_thresholded_output:
            sub_inputs+=aux_inputs

        self.model = Model(input=sub_inputs, output=all_outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=self.loss_weights)


    def generate_input_module(self,n_depth, n_width, l2_weight, name, n_input, thresholded_output, n_inception=0):
        sub_input = Input(shape=(1,n_input), dtype='float32', name=name)
        sub_input_2=Flatten()(sub_input)

        temp_output = add_layers(sub_input_2, n_depth, n_width, l2_weight)



        output_layer = Dense(self.n_out,init=INIT,activation=self.output_layer_activation,W_regularizer=l2(l2_weight),bias=True)(temp_output)
        #aux_input, merged_output = add_thresholded_output(output_layer, n_input, name)
        aux_input = Input(shape=(1,), dtype='float32', name='OnOff_' + name)
        merged_output=merge([aux_input, output_layer], mode='mul')

        sub_output_gas=Dense(1,activation=self.output_layer_activation,W_regularizer=l2(l2_weight),name=name+'_GAS')(merged_output)
        sub_output_oil = Dense(1, activation=self.output_layer_activation, W_regularizer=l2(l2_weight),name=name+'_OIL')(merged_output)


        return aux_input, sub_input,sub_output_gas,sub_output_oil

    def update_model(self):
        self.nb_epoch = 10000
        self.output_layer_activation = 'relu'
        self.aux_inputs = []
        self.inputs = []
        self.merged_outputs = []
        self.outputs = []

        old_model = self.model
        self.initialize_model()
        weights = old_model.get_weights()
        self.model.set_weights(weights)