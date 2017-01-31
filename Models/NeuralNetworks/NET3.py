
from .base import *
from .base_class import NN_BASE

class SSNET3(NN_BASE):


    def __init__(self):

        name='SSNET3'

        # Input module config
        self.IM_n_inception = 3 #(n_inception, n_depth inception)
        self.IM_n_depth = 2
        self.IM_n_width = 25
        self.l2weight = 0.0001
        self.add_thresholded_output=True

        self.input_tags=['CHK']
        self.output_tags=['QGAS']

        self.model_outputs=[]
        #Training config
        optimizer = 'adam' #SGD(momentum=0.9,nesterov=True)
        loss = 'mse'
        nb_epoch = 5000
        batch_size = 1000
        verbose = 0

        train_params={'optimizer':optimizer,'loss':loss,'nb_epoch':nb_epoch,'batch_size':batch_size,'verbose':verbose}

        aux_weights=0.2
        self.loss_weights={'GJOA_QGAS':1.0,'F1_out':0.0,'B2_out':0.0,'D3_out':0.0,'E1_out':0.0,
                           'F1_aux_out':aux_weights,'B2_aux_out':aux_weights,'D3_aux_out':aux_weights,'E1_aux_out':aux_weights}

        super().__init__(name,train_params)



    def generate_input_module(self,name,n_input,n_inception,n_depth,n_width,l2_weight):
        input_layer = Input(shape=(n_input,), dtype='float32', name=name)
        out_1 = add_layers(input_layer, n_depth, n_width, l2_weight)
        #out_1 = generate_inception_module(input_layer, n_inception,n_depth, n_width, l2_weight)
        out_1=add_layers(out_1,1,1,l2_weight)
        #out_1=Dense(1)(out_1)
        aux_out=Dense(1,name=name+'_aux_out')(out_1)

        merged_layer=merge([out_1,input_layer],mode='concat')

        out_1=add_layers(merged_layer,n_depth,n_width,l2_weight)
        #out_1=generate_inception_module(merged_layer, n_inception,n_depth, n_width, l2_weight)
        out_1=Dense(1,W_regularizer=l2(l2_weight),b_regularizer=l2(l2_weight))(out_1)

        aux_input, merged_output = add_thresholded_output(out_1, n_input, name)

        return input_layer,aux_input,aux_out,out_1,merged_output
    def initialize_model(self):

        i=1
        for key in self.input_tags:
            self.input_tags[key][0]=i
            i+=2
        print('Initializing %s' % (self.model_name))
        for key in self.input_tags:
            input_layer, aux_input, aux_output, output_layer, merged_output=self.generate_input_module(n_depth=self.IM_n_depth, n_width=self.IM_n_width,
                                                                    n_input=self.n_inputs, n_inception=self.IM_n_inception,
                                                                    l2_weight=self.l2weight, name=key)
            self.aux_inputs.append(aux_input)
            self.inputs.append(input_layer)
            self.merged_outputs.append(merged_output)

            self.model_outputs.append(merged_output)
            self.model_outputs.append(aux_output)
            self.outputs.append(output_layer)

        merged_input = merge(self.merged_outputs, mode='sum', name='GJOA_QGAS')

        self.model_outputs.insert(0,merged_input)
        inputs = self.inputs
        print(self.model_outputs)
        if self.add_thresholded_output:
            inputs+=self.aux_inputs

        self.model = Model(input=inputs, output=self.model_outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=self.loss_weights)
        #print(self.model.get_config())
        #print(self.model.summary())
