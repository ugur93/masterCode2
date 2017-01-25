
from .base import *
from .base_class import NN_BASE










class SSNET2(NN_BASE):


    def __init__(self):

        name='SSNET2'

        # Input module config
        self.IM_n_inception = 3 #(n_inception, n_depth inception)
        self.IM_n_depth = 1
        self.IM_n_width = 15
        self.l2weight = 0.00005
        self.add_thresholded_output=True

        self.input_tags=['CHK','PWH']
        #Training config
        optimizer = 'rmsprop' #SGD(momentum=0.9,nesterov=True)
        loss = 'mse'
        nb_epoch = 10000
        batch_size = 1000
        verbose = 0

        train_params={'optimizer':optimizer,'loss':loss,'nb_epoch':nb_epoch,'batch_size':batch_size,'verbose':verbose}

        super().__init__(name,train_params)

    def initialize_model(self):
        print('Initializing %s' % (self.model_name))




        for key in self.input_tags:
            aux_input,input,merged_out,out=generate_input_module(n_depth=self.IM_n_depth, n_width=self.IM_n_width,
                                                                    n_input=self.n_inputs, n_inception=self.IM_n_inception,
                                                                    l2_weight=self.l2weight, name=key,thresholded_output=self.add_thresholded_output)
            self.aux_inputs.append(aux_input)
            self.inputs.append(input)
            self.merged_outputs.append(merged_out)
            self.outputs.append(out)

        merged_input = merge(self.merged_outputs, mode='sum', name='GJOA_QGAS')

        self.merged_outputs.insert(0,merged_input)
        inputs = self.inputs

        if self.add_thresholded_output:
            inputs+=self.aux_inputs

        self.model = Model(input=inputs, output=self.merged_outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=[1.0, 0.0, 0.0, 0.0, 0.0])
        #print(self.get_config())
        #print(self.model.summary())
