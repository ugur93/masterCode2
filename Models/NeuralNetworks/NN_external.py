
from .base import *
from keras.models import load_model
from .base_class import NN_BASE


class SSNET_EXTERNAL(NN_BASE):

    def __init__(self,filename):

        name = 'SSNET_EXTERNAL'

        self.filename=filename
        self.PATH = '/Users/UAC/GITFOLDERS/MasterThesisCode/Models/NeuralNetworks/SavedModels/'

        # Input module config
        self.IM_n_inception = 0  # (n_inception, n_depth inception)
        self.IM_n_depth = 2
        self.IM_n_width = 25
        self.l2weight = 0.00001
        self.add_thresholded_output = True

        self.n_inputs = 4
        self.n_outputs = 5

        self.input_tags = ['CHK', 'PWH', 'PDC', 'PBH']
        # Training config
        optimizer = 'adam'  # SGD(momentum=0.9,nesterov=True)
        loss = 'mse'
        nb_epoch = 10000  # 15000
        batch_size = 1000
        verbose = 0

        train_params = {'optimizer': optimizer, 'loss': loss, 'nb_epoch': nb_epoch, 'batch_size': batch_size,
                        'verbose': verbose}

        self.output_tags = {
            'F1_out': ['F1_QGAS'],
            'B2_out': ['B2_QGAS'],
            'D3_out': ['D3_QGAS'],
            'E1_out': ['E1_QGAS'],
            'GJOA_QGAS': ['GJOA_QGAS']
        }
        self.input_tags = {
            'F1': ['F1_CHK'],
            'B2': ['B2_CHK'],
            'D3': ['D3_CHK'],
            'E1': ['E1_CHK']
        }

        self.output_index = output_tags_to_index(self.output_tags)
        print(self.output_index)
        self.loss_weights = [0.0, 0.0, 0.0, 0.0, 1.0]

        super().__init__(name, train_params)

    def generate_input_module(self,n_depth, n_width, l2_weight, name, n_input, thresholded_output, n_inception=0):

        input_layer = Input(shape=(n_input,), dtype='float32', name=name)
        # temp_output=Dropout(0.1)(input_layer)
        temp_output = add_layers(input_layer, 1, 4, l2_weight)
        if n_depth == 0:
            temp_output = input_layer
        else:
            if n_inception > 1:
                # temp_output = add_layers(input_layer, 1, n_width, l2_weight)
                temp_output = generate_inception_module(temp_output, n_inception, n_depth, n_width, l2_weight)
                temp_output = add_layers(temp_output, 1, n_width, l2_weight)
                #temp_output = generate_inception_module(temp_output, n_inception, n_depth, n_width, l2_weight)
                #temp_output = add_layers(temp_output, 1, n_width, l2_weight)
            else:
                temp_output = add_layers(temp_output, n_depth, n_width, l2_weight)

        if thresholded_output:
            output_layer = Dense(1, init=INIT, W_regularizer=l2(l2_weight), b_regularizer=l2(l2_weight))(temp_output)
            # output_layer = Dense(1,init=INIT, W_regularizer=l2(l2_weight), b_regularizer=l2(l2_weight),bias=True)(temp_output)
            aux_input, merged_output = add_thresholded_output(output_layer, n_input, name)
        else:
            output_layer = Dense(1, init=INIT, W_regularizer=l2(l2_weight), b_regularizer=l2(l2_weight), bias=True,
                                 name=name + '_out')(temp_output)

            merged_output = output_layer
            aux_input = input_layer

        return aux_input, input_layer, merged_output, output_layer

    def initialize_model(self):
        print('Initializing %s' % (self.model_name))
        ex_model = load_model(self.PATH + self.filename + '.h5')

        for key in self.input_tags:
            aux_input,input,merged_out,out=self.generate_input_module(n_depth=self.IM_n_depth, n_width=self.IM_n_width,
                                                                    n_input=1, n_inception=self.IM_n_inception,
                                                                    l2_weight=self.l2weight, name=key,thresholded_output=self.add_thresholded_output)
            self.aux_inputs.append(aux_input)
            self.inputs.append(input)
            self.merged_outputs.append(merged_out)
            self.outputs.append(out)

        merged_input = merge(self.merged_outputs, mode='sum', name='GJOA_QGAS')

        self.merged_outputs.append(merged_input)
        inputs = self.inputs
        print(self.merged_outputs)

        if self.add_thresholded_output:
            inputs+=self.aux_inputs

        new_model = Model(input=inputs, output=self.merged_outputs)
        new_model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=self.loss_weights)
        #print(self.get_config())


        new_weights=np.array(new_model.get_weights()[0:8])
        old_weights=np.array(ex_model.get_weights())
        new_weights=np.hstack((new_weights,old_weights))
        new_model.set_weights(new_weights)

        self.model=new_model





