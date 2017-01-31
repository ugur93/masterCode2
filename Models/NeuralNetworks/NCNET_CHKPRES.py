
from .base import *
from .base_class import NN_BASE



def sub(inputs):
    s = inputs[0]
    for i in range(1, len(inputs)):
        s -= inputs[i]
    return s

class SSNET3_PRESSURE(NN_BASE):


    def __init__(self):

        name='SSNET3'


        self.n_inputs=4
        self.n_outputs=1

        # Input module config
        self.n_inception = 0 #(n_inception, n_depth inception)
        self.n_depth = 2
        self.n_width = 20
        self.l2weight = 0.001
        self.add_thresholded_output=True

        #self.output_tags = {
        #    'MAIN_OUTPUT': ['F1_PWH','F1_PDC','B2_PWH','B2_PDC','D3_PWH','D3_PDC','E1_PWH','E1_PDC']
        #}

        self.output_tags = {
            #'MAIN_OUTPUT':['F1_PDC','B2_PDC','D3_PDC','E1_PDC'],
            'GJOA_QGAS':['GJOA_QGAS'],
            #'MAIN_OUTPUT': ['F1_deltap', 'B2_deltap', 'D3_deltap', 'E1_deltap']
            #'MAIN_OUTPUT':['F1_PDC','B2_PDC','D3_PDC','E1_PDC']
            #'MAIN_OUTPUT':['F1_PDC','B2_PDC']
            #'MAIN_OUTPUT': ['F1_PWH', 'F1_PDC', 'B2_PWH', 'B2_PDC', 'D3_PWH', 'D3_PDC', 'E1_PWH', 'E1_PDC']
         }


        self.input_tags={
            'CHK':['B2_CHK','F1_CHK','D3_CHK','E1_CHK'],
            #'PDC':['B2_PDC']
            #'B2':['B2_CHK','B2_PDC'],
            #'D3':['D3_CHK'],
            #'E1':['E1_CHK']
        }
        self.input_name='E1'
        well_names=['F1','B2','D3','E1']
        tags=['CHK']

        self.input_tags={'CHK':[]}
        for key in well_names:
            for tag in tags:
                self.input_tags['CHK'].append(key+'_'+tag)
        print(self.input_tags)
       # self.input_tags = {
       #     'MAIN_IN': ['F1_CHK','B2_CHK','D3_CHK','E1_CHK'],
            #'MAIN_IN': ['F1_PWH','B2_PWH','D3_PWH','E1_PWH']
       # }

        self.loss_weights=[0.0,0.0,0.0,0.0,1.0]

        #Training config
        optimizer = 'adam' #SGD(momentum=0.9,nesterov=True)
        loss = 'mse'
        nb_epoch = 5000
        batch_size = 64
        verbose = 0

        train_params={'optimizer':optimizer,'loss':loss,'nb_epoch':nb_epoch,'batch_size':batch_size,'verbose':verbose}

        super().__init__(name,train_params)

    def initialize_model(self):
        print('Initializing %s' % (self.model_name))

        outs=[]
        for name in self.input_tags.keys():
            input=Input(shape=(self.n_inputs,), dtype='float32', name=name)
            out=add_layers(input, n_depth=1, n_width=20, l2_weight=self.l2weight)

            self.inputs.append(input)
            outs.append(out)


        #main_input = merge(outs, mode='concat')


        main_input=Input(shape=(self.n_inputs,), dtype='float32', name='CHK')

        #input_chk=Input(shape=(1,), dtype='float32', name='CHK')
        #model_chk=add_layers(input_chk,n_width=10,n_depth=2,l2_weight=self.l2weight)
        #model_chk = add_layers(model_chk, n_width=1, n_depth=1, l2_weight=self.l2weight)

        #input_pres = Input(shape=(1,), dtype='float32', name='PDC')
        #model_pres = add_layers(input_pres, n_width=10, n_depth=2, l2_weight=self.l2weight)
        #model_pres = add_layers(model_pres, n_width=1, n_depth=1, l2_weight=self.l2weight)
        #main_input=[input_chk,input_pres]

        #main_model=merge([model_chk,model_pres],mode='concat')

        #main_model=Dense(20)(main_input)


        #main_model = generate_inception_module(main_input, 3, 1, 10, self.l2weight)
        main_model = add_layers(main_input, self.n_depth,self.n_width, self.l2weight)
        #main_output=merge([PWH_out,PDC_out],mode='sum',name='MAIN_OUTPUT')
        main_output = Dense(self.n_outputs,name='GJOA_QGAS')(main_model)

        self.model = Model(input=main_input, output=main_output)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        plotModel(self.model, 'CHK_TO_PRESSURE')
