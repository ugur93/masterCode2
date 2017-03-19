
from .base import *
from .base_class import NN_BASE

import keras.backend as K

def sub(inputs):
    s = inputs[0]
    for i in range(1, len(inputs)):
        s -= inputs[i]
    return s
PRES='PBHs'
from theano.ifelse import ifelse
from theano import tensor as T
import theano
def zero_layer(x):
    a,b=T.scalars('a','b')
    c, y = T.matrices('v', 'y')

    z_lazy=ifelse(T.lt(a,b),c,y)

    f_lazyifelse= theano.function([a,b,c,y], z_lazy,
                               mode=theano.Mode(linker='vm'),allow_input_downcast=True)
    return f_lazyifelse(5,x,0,x)
    #+return x
class SSNET3_PRESSURE(NN_BASE):


    def __init__(self,Data):



        self.model_name='GJOA_OIL_WELLS_PBH_INPUT_PWH_PDC_MODEL_wONOFFn9'
        self.out_act='relu'
        # Training config
        self.optimizer ='adam'#SGD(momentum=0.9,nesterov=True)
        self.loss = 'mse'
        self.nb_epoch = 10000
        self.batch_size = 64
        self.verbose = 0
        #self.mno=mno
        #self.mnh=mnh
        self.p_dropout=1

        self.n_inputs=5
        self.n_outputs=1
        self.SCALE=100000
        # Input module config
        self.n_inception = 0 #(n_inception, n_depth inception)
        self.n_depth = 1
        self.n_width = 1
        self.l2weight = 0.0001
        self.add_thresholded_output=True



        self.input_name='E1'
        #chk_names=['F1','B2','D3','E1']

        chk_names=['C1', 'C2', 'C3', 'C4', 'B1', 'B3', 'D1']
        if PRES=='PBH':
            self.well_names=['C1','C3', 'C4','B1','B3']
        else:
            self.well_names = ['C1', 'C2', 'C3', 'C4', 'B3','B1','D1']
        tags=['CHK']
        #self.well_names=chk_names

        self.input_tags={'PRESSURE_INPUT':[]}
        for key in chk_names:
            for tag in tags:
                self.input_tags['PRESSURE_INPUT'].append(key+'_'+tag)
        for key in ['C1','C3', 'C4','B1','B3']:
                self.input_tags['PRESSURE_INPUT'].append(key + '_' + 'PBH')
        self.input_tags['RISER_B_CHK_INPUT']=['GJOA_RISER_OIL_B_CHK']
        self.n_inputs = len(self.input_tags['PRESSURE_INPUT'])
        self.n_outputs=1

        self.output_tags = {}

        for name in self.well_names:

            self.output_tags[name + '_PWH_pred'] = [name + '_' + 'PWH']
            self.output_tags[name + '_PDC_pred'] = [name + '_' + 'PDC']


        self.output_zero_thresholds = {}
        #self.chk_threshold_value=5
        self.initialize_zero_thresholds(Data)
        #self.initialize_chk_thresholds(Data, True)
        print(self.output_zero_thresholds)
        super().__init__()


    def initialize_model2(self):
        print('Initializing %s' % (self.model_name))

        chk_input = Input(shape=(len(self.input_tags['Main_input']),), dtype='float32', name='Main_input')

        outputs=[]
        inputs=[chk_input]

        sub_model1 = Dense(20, W_regularizer=l2(self.l2weight), activation='relu')(chk_input)


        for key in self.well_names:
            #sub_model_temp = Dense(20, W_regularizer=l2(self.l2weight), activation='relu')(chk_input)
            #sub_model_temp = Dropout(0.05)(sub_model1)
            sub_model_temp = Dense(len(self.output_tags[key + '_out']),
                                   W_regularizer=l2(self.l2weight), activation=self.out_act, name=key + '_oSut')(sub_model1)

            aux_input = Input(shape=(len(self.input_tags['aux_' + key]),), dtype='float32', name='OnOff_' + key)

            sub_model_out = merge([sub_model_temp, aux_input], mode='mul', name=key + '_out')
            #sub_model_out = Dense(1, W_regularizer=l2(self.l2weight), activation=self.out_act)(sub_model_out)

            outputs.append(sub_model_out)
            inputs.append(aux_input)

        #sub_model=Dense(20,W_regularizer=l2(self.l2weight),activation='relu')(sub_model)


        #sub_model=Dense(len(self.output_tags['Main_output']),W_regularizer=l2(self.l2weight),name='Main_output')(sub_model)

        self.model = Model(input=inputs, output=outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
    def initialize_model(self):
        print('Initializing %s' % (self.model_name))

        all_chk_input = Input(shape=(len(self.input_tags['PRESSURE_INPUT']),), dtype='float32', name='PRESSURE_INPUT')
        riser_chk_input = Input(shape=(len(self.input_tags['RISER_B_CHK_INPUT']),), dtype='float32', name='RISER_B_CHK_INPUT')

        all_and_riser_chk_input=merge([all_chk_input,riser_chk_input],mode='concat',name='RISER_MERGE')

        output_layers = {}
        outputs = []
        inputs = [all_chk_input,riser_chk_input]

        #sub_model_all = Dense(20, W_regularizer=l2(self.l2weight), activation='relu',name='pres_sub_all')(all_chk_input)
        #sub_model_all = Dense(20, W_regularizer=l2(self.l2weight), activation='relu',name='pres_sub_all2')(sub_model_all)


        for key in self.well_names:
            sub_model_PWH=self.generate_sub_model(all_chk_input,name=key+'_PWH')
            sub_model_PDC=self.generate_sub_model(all_and_riser_chk_input,name=key+'_PDC')
            #sub_model_temp = Dense(20, W_regularizer=l2(self.l2weight), activation='relu')(sub_model_temp)


            # sub_model_temp = Dropout(0.05)(sub_model1)
            PWH_out = Dense(1,
                            W_regularizer=l2(self.l2weight), activation='relu', name=key + '_PWH_out')(sub_model_PWH)

            PDC_out = Dense(1,
                            W_regularizer=l2(self.l2weight), activation='relu', name=key + '_PDC_out')(sub_model_PDC)

            aux_input_PDC = Input(shape=(1,), dtype='float32', name='OnOff_PDC_' + key)
            aux_input_PWH = Input(shape=(1,), dtype='float32', name='OnOff_PWH_' + key)

            PWH_out = merge([PWH_out, aux_input_PWH], mode='mul', name=key + '_PWH_pred')
            PDC_out = merge([PDC_out, aux_input_PDC], mode='mul', name=key + '_PDC_pred')


            #PRESSURE_OUT = merge([PWH_out, PDC_out], mode='concat')
            # sub_model_out = Dense(1, W_regularizer=l2(self.l2weight), activation=self.out_act)(sub_model_out)

            #output_layers[key] = PRESSURE_OUT
            outputs.append(PWH_out)
            outputs.append(PDC_out)

            inputs.append(aux_input_PDC)
            inputs.append(aux_input_PWH)

        self.model = Model(input=inputs, output=outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def update_model(self):
        self.nb_epoch=10000
        self.out_act='relu'
        #self.aux_inputs=[]
        #self.inputs=[]
        #self.merged_outputs=[]
        #self.outputs=[]

        old_model=self.model
        self.initialize_model()
        weights=old_model.get_weights()
        self.model.set_weights(weights)

    def generate_sub_model(self,input_layer,name,l2w=0,depth=2):
        if l2w==0:
            l2w=self.l2weight
        i=0
        sub_model = Dense(50, W_regularizer=l2(l2w), activation='relu',name=name+'_'+str(i))(input_layer)

        for i in range(1,depth):
            # sub_model_temp=Dropout(0.01)(sub_model_temp)
            sub_model = Dense(50,W_regularizer=l2(l2w), activation='relu',name=name+'_'+str(i))(sub_model)

        return sub_model