
from .base import *
from .base_class import NN_BASE



#GOOD

#incept 3
# dept 1
# width 20
#l2 0.0001
#opt rmsprop
class NCNET1_GJOA2(NN_BASE):


    def __init__(self):

        self.model_name='NCNET1_2'

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
        self.n_depth = 2
        self.n_depth_incept=2
        self.n_width_incept=20
        self.n_width = 20
        self.l2weight = 0.0001
        self.add_thresholded_output=True

        self.input_tags = {}
        all_tags=[]
        well_name = ['C1', 'C3', 'C4', 'B3', 'B1','D1','C2']
        tags = ['CHK','PWH','PDC','PBH']
        for name in well_name:
            self.input_tags[name] = []
            for tag in tags:
                if (name=='C2' or name=='D1') and tag=='PBH':
                    #tag='PWH'
                    pass
                else:
                    #all_tags.append(name + '_' + tag)
                    self.input_tags[name].append(name + '_' + tag)
        #for key in self.input_tags.keys():
            #self.input_tags[key]=all_tags
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
            'GJOA_TOTAL':['GJOA_TOTAL_QOIL']
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

        for key in self.input_tags:
            n_input=len(self.input_tags[key])
            aux_input,input,merged_out,out=self.generate_input_module(n_depth=self.n_depth, n_width=self.n_width,
                                                                    n_input=n_input, n_inception=self.n_inception,
                                                                    l2_weight=self.l2weight, name=key,thresholded_output=self.add_thresholded_output)
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


    def generate_input_module(self,n_depth, n_width, l2_weight, name, n_input, thresholded_output, n_inception=0):

        input_layer = Input(shape=(n_input,), dtype='float32', name=name)
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
            output_layer = Dense(1,W_regularizer=l2(l2_weight), bias=True)(temp_output)
            # output_layer = Dense(1,init=INIT, W_regularizer=l2(l2_weight), b_regularizer=l2(l2_weight),bias=True)(temp_output)
            aux_input, merged_output = add_thresholded_output(output_layer, n_input, name)
        else:
            output_layer = Dense(1, init=INIT, W_regularizer=l2(l2_weight), bias=True,
                                 name=name + '_out')(temp_output)

            merged_output = output_layer
            aux_input = input_layer

        return aux_input, input_layer, merged_output, output_layer


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
            output_layer = Dense(1,W_regularizer=l2(l2_weight), bias=True)(temp_output)
            # output_layer = Dense(1,init=INIT, W_regularizer=l2(l2_weight), b_regularizer=l2(l2_weight),bias=True)(temp_output)
            aux_input, merged_output = add_thresholded_output(output_layer, n_input, name)
        else:
            output_layer = Dense(1, init=INIT, W_regularizer=l2(l2_weight), bias=True,
                                 name=name + '_out')(temp_output)

            merged_output = output_layer
            aux_input = input_layer

        return aux_input, [], merged_output, output_layer