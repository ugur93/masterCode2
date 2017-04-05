
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

        self.output_layer_activation = 'relu'
        # Training config
        optimizer = 'adam'  # SGD(momentum=0.9,nesterov=True)
        loss = 'mae'
        self.activation='relu'
        nb_epoch = 1000
        batch_size = 64
        verbose = 0


        n_depth = 2
        n_width = 80
        l2w = 0.0003

        self.n_depth_pdc=2
        self.n_depth_pwh=2
        self.n_depth_pbh=2

        self.n_width_pdc=100
        self.n_width_pwh=100
        self.n_width_pbh=100
        self.add_thresholded_output=True




        self.input_tags = {}
        self.well_names = ['C1', 'C2', 'C3', 'C4', 'B1', 'B3', 'D1']  #
        tags = ['CHK']#, 'PBH', 'PWH', 'PDC']
        for name in self.well_names:
            if name not in ['C2','D1']:
                self.input_tags[name+'_PBH_IN']=[name+'_PBH']
            self.input_tags[name+'_CHK'] = []
            for tag in tags:
                if (name == 'C2' or name == 'D1') and tag == 'PBH':
                    pass
                else:
                    self.input_tags[name+'_CHK'].append(name + '_' + tag)

        self.input_tags['PRESSURE_INPUT']=[]
        for key in self.well_names:
            for tag in tags:
                self.input_tags['PRESSURE_INPUT'].append(key + '_' + tag)
        for key in ['C1','C3', 'C4','B1','B3']:
                self.input_tags['PRESSURE_INPUT'].append(key + '_' + '1_PBH')
        self.input_tags['RISER_B_CHK_INPUT']=['GJOA_RISER_OIL_B_CHK']



        self.output_tags = {


            'C1_out': ['C1_QGAS'],
            'C2_out': ['C2_QGAS'],
            'C3_out': ['C3_QGAS'],
            'C4_out': ['C4_QGAS'],
            'D1_out': ['D1_QGAS'],
            'B3_out': ['B3_QGAS'],
            'B1_out': ['B1_QGAS'],

            'GJOA_TOTAL': ['GJOA_OIL_QGAS']
        }

        for key in self.well_names:
            self.output_tags[key+'_PWH_out']=[key+'_PWH']
            self.output_tags[key + '_PDC_out'] = [key + '_PDC']


        self.loss_weights = {

            'C1_out': 0.0,
            'C2_out': 0.0,
            'C3_out': 0.0,
            'C4_out': 0.0,
            'D1_out': 0.0,
            'B3_out': 0.0,
            'B1_out': 0.0,

            'C1_PWH_out': 0.0,
            'C2_PWH_out': 0.0,
            'C3_PWH_out': 0.0,
            'C4_PWH_out': 0.0,
            'D1_PWH_out': 0.0,
            'B3_PWH_out': 0.0,
            'B1_PWH_out': 0.0,

            'C1_PDC_out': 0.0,
            'C2_PDC_out': 0.0,
            'C3_PDC_out': 0.0,
            'C4_PDC_out': 0.0,
            'D1_PDC_out': 0.0,
            'B3_PDC_out': 0.0,
            'B1_PDC_out': 0.0,




            'GJOA_TOTAL': 1.0
        }


        print(self.output_tags)
        super().__init__(n_width=n_width, n_depth=n_depth, l2_weight=l2w, dp_rate=0, seed=3014,
                         optimizer=optimizer, loss=loss, nb_epoch=nb_epoch, batch_size=batch_size)

    def initialize_model(self):
        print('Initializing %s' % (self.model_name))


        rate_outputs = []

        pres_inputs,pres_outputs,pres_output_layers=self.generate_pressure_model()

        all_inputs=pres_inputs
        all_outputs=pres_outputs
        for key in self.well_names:
            n_input = len(self.input_tags[key+'_CHK'])
            aux_input, input,merged_output,aux_input_PBH = self.generate_input_module(n_input=n_input,pres_output_layers=pres_output_layers[key],name=key)



            all_inputs.append(input)
            all_inputs.append(aux_input)
            if aux_input_PBH is not None:
                all_inputs.append(aux_input_PBH)
            rate_outputs.append(merged_output)
            all_outputs.append(merged_output)


        merged_input = Add( name='GJOA_TOTAL')(rate_outputs)

        all_outputs=all_outputs+[merged_input]

        self.model = Model(input=all_inputs, output=all_outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=self.loss_weights)

        #.model.load_weights(self.pressure_weights_path,by_name=True)


    def generate_input_module(self,n_input,pres_output_layers,name):
        chk_input = Input(shape=(n_input,), dtype='float32', name=name+'_CHK')

        PWH_out=pres_output_layers[0]
        PDC_out=pres_output_layers[1]
        aux_input_PBH=None
        if name in ['C2', 'D1']:
            all_inputs = Concatenate(name=name)([chk_input, PWH_out, PDC_out])
        else:

            aux_input_PBH = Input(shape=(1,), dtype='float32', name=name + '_PBH_IN')
            all_inputs = Concatenate(name=name)([chk_input, PWH_out, PDC_out,aux_input_PBH])
        #pres_output_layers.append(sub_input)



        output_layer = Dense(self.n_width, activation='relu', kernel_regularizer=l2(self.l2weight),
                            kernel_initializer=self.init, use_bias=True, name=name + '_0')(all_inputs)
        # temp_output = MaxoutDense(self.n_width,kernel_regularizer=l2(self.l2weight),kernel_initializer=self.init,use_bias=True)(input_layer)
        # temp_output=PReLU()(temp_output)
        for i in range(1, self.n_depth):
            if self.dp_rate > 0:
                output_layer = Dropout(self.dp_rate, name=name + '_dp_' + str(i))(output_layer)
            output_layer = Dense(self.n_width, activation='relu', kernel_regularizer=l2(self.l2weight),
                                kernel_initializer=self.init, use_bias=True, name=name + '_' + str(i))(output_layer)
            # temp_output = PReLU()(temp_output)

        output_layer = Dense(1, kernel_initializer=self.init, kernel_regularizer=l2(self.l2weight),
                             activation=self.output_layer_activation, use_bias=True, name=name + '_o')(output_layer)


        #aux_input, merged_output = add_thresholded_output(output_layer, n_input, name)
        aux_input = Input(shape=(1,), dtype='float32', name='OnOff_' + name)
        merged_output=Multiply(name=name+'_out')([aux_input, output_layer])



        return aux_input, chk_input,merged_output,aux_input_PBH

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

        all_and_riser_chk_input = Concatenate( name='RISER_MERGE')([all_chk_input, riser_chk_input])

        output_layers = {}
        outputs = []
        inputs = [all_chk_input, riser_chk_input]

        # sub_model_all = Dense(20, W_regularizer=l2(self.l2weight), activation='relu',name='pres_sub_all')(all_chk_input)
        # sub_model_all = Dense(20, W_regularizer=l2(self.l2weight), activation='relu',name='pres_sub_all2')(sub_model_all)


        for key in self.well_names:
            sub_model_PWH = self.generate_sub_pressure_model(all_chk_input, name=key + '_PWH',depth=self.n_depth_pwh,width=self.n_width_pwh)
            sub_model_PDC = self.generate_sub_pressure_model(all_and_riser_chk_input, name=key + '_PDC',depth=self.n_depth_pdc,width=self.n_width_pdc)
            # sub_model_temp = Dense(20, W_regularizer=l2(self.l2weight), activation='relu')(sub_model_temp)


            # sub_model_temp = Dropout(0.05)(sub_model1)
            PWH_out = Dense(1,
                            kernel_regularizer=l2(self.l2weight), activation='linear', name=key + '_PWH_out',trainable=False)(sub_model_PWH)

            PDC_out = Dense(1,
                            kernel_regularizer=l2(self.l2weight), activation='linear', name=key + '_PDC_out',trainable=False)(sub_model_PDC)



            # sub_model_out = Dense(1, W_regularizer=l2(self.l2weight), activation=self.out_act)(sub_model_out)

            output_layers[key] = [PWH_out,PDC_out]
            outputs.append(PWH_out)
            outputs.append(PDC_out)





        return inputs,outputs,output_layers

    def generate_sub_pressure_model(self,input_layer,name,depth=2,width=50):
        i=0
        sub_model = Dense(width, kernel_regularizer=l2(self.l2weight), activation='relu',name=name+'_'+str(i),trainable=False)(input_layer)
        for i in range(1,depth):
            # sub_model_temp=Dropout(0.01)(sub_model_temp)
            sub_model = Dense(width, kernel_regularizer=l2(self.l2weight), activation='relu',name=name+'_'+str(i),trainable=False)(sub_model)

        return sub_model