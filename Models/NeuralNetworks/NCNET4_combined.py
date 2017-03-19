
from .base import *
from .base_class import NN_BASE



def sub(inputs):
    s = inputs[0]
    for i in range(1, len(inputs)):
        s -= inputs[i]
    return s

class NET4_W_PRESSURE(NN_BASE):


    def __init__(self,pressure_weights_path):

        self.model_name='NCNET3_PBH_INPUT'

        self.output_layer_activation = 'linear'
        # Training config
        self.optimizer = 'adam'  # SGD(momentum=0.9,nesterov=True)
        self.loss = 'mse'
        self.nb_epoch = 100
        self.batch_size = 64
        self.verbose = 0

        self.pressure_weights_path=pressure_weights_path

        self.n_inputs=5
        self.n_outputs=1
        self.SCALE=100000
        # Input module config
        self.n_inception = 0 #(n_inception, n_depth inception)
        self.n_depth = 1
        self.n_conv=1
        self.n_width = 20
        self.l2weight = 0.000005
        self.add_thresholded_output=True
        self.n_out=1



        self.input_tags = {}
        self.well_names = ['C1', 'C2', 'C3', 'C4', 'B1', 'B3', 'D1']  #
        tags = ['CHK']#, 'PBH', 'PWH', 'PDC']
        for name in self.well_names:
            if name not in ['C2','D1']:
                self.input_tags[name+'_PBH_IN']=[name+'_PBH']
            self.input_tags[name] = []
            for tag in tags:
                if (name == 'C2' or name == 'D1') and tag == 'PBH':
                    pass
                else:
                    self.input_tags[name].append(name + '_' + tag)
                    # self.input_tags[name].append('GJOA_RISER_OIL_B_CHK')

        pressure_in_tags = ['CHK']
        # self.well_names=chk_names

        self.input_tags['PRESSURE_INPUT']=[]
        for key in self.well_names:
            for tag in tags:
                self.input_tags['PRESSURE_INPUT'].append(key + '_' + tag)
        for key in ['C1','C3', 'C4','B1','B3']:
                self.input_tags['PRESSURE_INPUT'].append(key + '_' + '1_PBH')
        self.input_tags['RISER_B_CHK_INPUT']=['GJOA_RISER_OIL_B_CHK']
        # self.input_tags['A1']=['C1_CHK','C1_PWH','C1_PDC','C1_PBH']
        # self.input_tags['B1']=['B1_CHK']
        # self.input_tags['C1']=['C1_CHK']
        print(self.input_tags)


        self.output_tags = {


            'C1_OUT': ['C1_QGAS'],
            'C2_OUT': ['C2_QGAS'],
            'C3_OUT': ['C3_QGAS'],
            'C4_OUT': ['C4_QGAS'],
            'D1_OUT': ['D1_QGAS'],
            'B3_OUT': ['B3_QGAS'],
            'B1_OUT': ['B1_QGAS'],

            'GJOA_TOTAL': ['GJOA_OIL_QGAS']
        }

        for key in self.well_names:
            self.output_tags[key+'_PWH_pred']=[key+'_PWH']
            self.output_tags[key + '_PDC_pred'] = [key + '_PDC']


        self.loss_weights = {

            'C1_OUT': 0.0,
            'C2_OUT': 0.0,
            'C3_OUT': 0.0,
            'C4_OUT': 0.0,
            'D1_OUT': 0.0,
            'B3_OUT': 0.0,
            'B1_OUT': 0.0,

            'C1_PWH_pred': 0.0,
            'C2_PWH_pred': 0.0,
            'C3_PWH_pred': 0.0,
            'C4_PWH_pred': 0.0,
            'D1_PWH_pred': 0.0,
            'B3_PWH_pred': 0.0,
            'B1_PWH_pred': 0.0,

            'C1_PDC_pred': 0.0,
            'C2_PDC_pred': 0.0,
            'C3_PDC_pred': 0.0,
            'C4_PDC_pred': 0.0,
            'D1_PDC_pred': 0.0,
            'B3_PDC_pred': 0.0,
            'B1_PDC_pred': 0.0,




            'GJOA_TOTAL': 1.0
        }
        print('HEREEE')

        print(self.output_tags)
        super().__init__()
    def initialize_model(self):
        print('Initializing %s' % (self.model_name))

        aux_inputs = []
        inputs = []
        rate_outputs = []
        outputs = []
        # merged_outputs=[]

        n_depth = self.n_depth
        n_width = self.n_width
        l2w = self.l2weight

        pres_inputs,pres_outputs,pres_output_layers=self.generate_pressure_model()

        all_inputs=pres_inputs
        all_outputs=pres_outputs
        for key in self.well_names:
            n_input = len(self.input_tags[key])
            aux_input, input,merged_output = self.generate_input_module(n_input=n_input,pres_output_layers=pres_output_layers[key],name=key)



            all_inputs.append(input)
            all_inputs.append(aux_input)
            rate_outputs.append(merged_output)
            all_outputs.append(merged_output)


        merged_input = merge(rate_outputs, mode='sum', name='GJOA_TOTAL')

        all_outputs.append(merged_input)

        self.model = Model(input=all_inputs, output=all_outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=self.loss_weights)

        self.model.load_weights(self.pressure_weights_path,by_name=True)


    def generate_input_module(self,n_input,pres_output_layers,name):
        sub_input = Input(shape=(n_input,), dtype='float32', name=name)
        #pres_output_layers.append(sub_input)
        all_inputs=merge([pres_output_layers,sub_input],mode='concat')

        output_layer = Dense(self.n_width, activation='relu', W_regularizer=l2(self.l2weight), init=INIT,
                            bias=True)(all_inputs)
        #output_layer=Dropout(0.01)(output_layer)
        output_layer = Dense(self.n_width, activation='relu', W_regularizer=l2(self.l2weight), init=INIT,
                             bias=True)(output_layer)



        output_layer = Dense(1,init=INIT,activation=self.output_layer_activation,W_regularizer=l2(self.l2weight),bias=True)(output_layer)
        #aux_input, merged_output = add_thresholded_output(output_layer, n_input, name)
        aux_input = Input(shape=(1,), dtype='float32', name='OnOff_' + name)
        merged_output=merge([aux_input, output_layer], mode='mul',name=name+'_OUT')



        return aux_input, sub_input,merged_output

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

    def generate_pressure_model(self):

        all_chk_input = Input(shape=(len(self.input_tags['PRESSURE_INPUT']),), dtype='float32', name='PRESSURE_INPUT')
        riser_chk_input = Input(shape=(len(self.input_tags['RISER_B_CHK_INPUT']),), dtype='float32',
                                name='RISER_B_CHK_INPUT')

        all_and_riser_chk_input = merge([all_chk_input, riser_chk_input], mode='concat', name='RISER_MERGE')

        output_layers = {}
        outputs = []
        inputs = [all_chk_input, riser_chk_input]

        # sub_model_all = Dense(20, W_regularizer=l2(self.l2weight), activation='relu',name='pres_sub_all')(all_chk_input)
        # sub_model_all = Dense(20, W_regularizer=l2(self.l2weight), activation='relu',name='pres_sub_all2')(sub_model_all)


        for key in self.well_names:
            sub_model_PWH = self.generate_sub_pressure_model(all_chk_input, name=key + '_PWH')
            sub_model_PDC = self.generate_sub_pressure_model(all_and_riser_chk_input, name=key + '_PDC')
            # sub_model_temp = Dense(20, W_regularizer=l2(self.l2weight), activation='relu')(sub_model_temp)


            # sub_model_temp = Dropout(0.05)(sub_model1)
            PWH_out = Dense(1,
                            W_regularizer=l2(self.l2weight), activation='relu', name=key + '_PWH_out',trainable=False)(sub_model_PWH)

            PDC_out = Dense(1,
                            W_regularizer=l2(self.l2weight), activation='relu', name=key + '_PDC_out',trainable=False)(sub_model_PDC)

            aux_input_PDC = Input(shape=(1,), dtype='float32', name='OnOff_PDC_' + key)
            aux_input_PWH = Input(shape=(1,), dtype='float32', name='OnOff_PWH_' + key)

            PWH_out = merge([PWH_out, aux_input_PWH], mode='mul', name=key + '_PWH_pred')
            PDC_out = merge([PDC_out, aux_input_PDC], mode='mul', name=key + '_PDC_pred')

            if key in ['C2','D1']:
                PRESSURE_OUT = merge([PWH_out, PDC_out], mode='concat')
            else:
                aux_input_PBH = Input(shape=(1,), dtype='float32', name=key+'_PBH_IN')
                PRESSURE_OUT = merge([PWH_out, PDC_out,aux_input_PBH], mode='concat')
                inputs.append(aux_input_PBH)

            # sub_model_out = Dense(1, W_regularizer=l2(self.l2weight), activation=self.out_act)(sub_model_out)

            output_layers[key] = PRESSURE_OUT
            outputs.append(PWH_out)
            outputs.append(PDC_out)

            inputs.append(aux_input_PDC)
            inputs.append(aux_input_PWH)



        return inputs,outputs,output_layers

    def generate_sub_pressure_model(self,input_layer,name,l2w=0,depth=2):
        if l2w==0:
            l2w=self.l2weight
        i=0
        sub_model = Dense(50, W_regularizer=l2(l2w), activation='relu',name=name+'_'+str(i),trainable=False)(input_layer)

        for i in range(1,depth):
            # sub_model_temp=Dropout(0.01)(sub_model_temp)
            sub_model = Dense(50, W_regularizer=l2(l2w), activation='relu',name=name+'_'+str(i),trainable=False)(sub_model)

        return sub_model