
from .base import *
from .base_class import NN_BASE

import keras.backend as K
K.set_image_dim_ordering('th')
#GOOD

#incept 3
# dept 1
# width 20
#l2 0.0001
#opt rmsprop


#FOR OIL DATASET
#2x30 Dense
#Maxnorm 4 for all
#nbepoch=100
#batch=64


#MSE: 5708
#Best params:{'n_depth': 3, 'l2w': 0.00029999999999999997, 'n_width': 100, 'seed': 3014}
#OIL Best params:{'seed': 3014, 'l2w': 0.000706122448979592, 'n_depth': 2, 'n_width': 90}
#GAS Best params:{'n_depth': 2, 'seed': 3014, 'l2w': 0.00030204081632653063, 'n_width': 60}

sas=5708
OUT = 'GASs'
class NCNET1_GJOA2(NN_BASE):



    def __init__(self,n_depth=2 ,n_width=200,l2w=0.0001,dp_rate=0,seed=3014):



        self.input_tags = {}

        self.well_names = ['C1', 'C2', 'C3', 'C4', 'B1', 'B3', 'D1']

        measurement_tags = ['CHK', 'PWH', 'PBH','PDC']
        for name in self.well_names:
            self.input_tags[name] = []
            for tag in measurement_tags:
                if (name == 'C2' or name == 'D1') and tag == 'PBH':
                    pass
                else:
                    self.input_tags[name].append(name + '_' + tag)

        if OUT == 'GAS':
            self.output_tags = OIL_WELLS_QGAS_OUTPUT_TAGS
        else:
            self.output_tags = OIL_WELLS_QOIL_OUTPUT_TAGS

        self.loss_weights = {
            'B1_out': 0.0,
            'B3_out': 0.0,
            'C2_out': 0.0,
            'C3_out': 0.0,
            'D1_out': 0.0,
            'C4_out': 0.0,
            'C1_out': 0.0,

            'GJOA_TOTAL': 1.0,
        }

        self.output_layer_activation = 'relu'

        # Training config
        optimizer = 'adam'
        loss ='mae'
        nb_epoch = 5000
        batch_size = 64
        self.activation='relu'

        self.model_name ='GJOA_OIL_WELLS_OIL_MODEL_FINAL'# 'GJOA_OIL2S_WELLS_{}_D{}_W{}_L2{}_DPR{}'.format(loss, n_depth, n_width, l2w,dp_rate)



        super().__init__(n_width=n_width,n_depth=n_depth,l2_weight=l2w,dp_rate=dp_rate,seed=seed,
                         optimizer=optimizer,loss=loss,nb_epoch=nb_epoch,batch_size=batch_size)






    def initialize_model(self):
        print('Initializing %s' % (self.model_name))

        aux_inputs=[]
        inputs=[]
        merged_outputs=[]
        outputs=[]

        for key in self.well_names:
            n_input=len(self.input_tags[key])
            aux_input,input,merged_out,out=self.generate_input_module(name=key,n_input=n_input)

            aux_inputs.append(aux_input)
            inputs.append(input)
            merged_outputs.append(merged_out)


        merged_input = Add(name='GJOA_TOTAL')(merged_outputs)

        all_outputs=merged_outputs+[merged_input]
        all_inputs = inputs

        if self.add_thresholded_output:
            all_inputs+=aux_inputs


        self.model = Model(inputs=all_inputs, outputs=all_outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=self.loss_weights)

    def initialize_model2(self):
        print('Initializing %s' % (self.model_name))

        aux_inputs=[]
        inputs=[]
        merged_outputs=[]
        outputs=[]
        #merged_outputs=[]

        for key in self.well_names:
            n_input=len(self.input_tags[key])
            aux_input_mod1, aux_input_mod2, input_layer, merged_output_mod1, merged_output_mod2, main_out=self.generate_input_module_gas(name=key,n_input=n_input)

            aux_inputs.append(aux_input_mod1)
            aux_inputs.append(aux_input_mod2)
            inputs.append(input_layer)
            merged_outputs.append(main_out)
            #merged_outputs.append(merged_output_mod2)
            #outputs.append(main_out)


        merged_input = Add(name='GJOA_TOTAL')(merged_outputs)

        all_outputs=merged_outputs+[merged_input]
        all_inputs = inputs

        if self.add_thresholded_output:
            all_inputs+=aux_inputs


        self.model = Model(inputs=all_inputs, outputs=all_outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=self.loss_weights)



    def generate_input_module(self, name, n_input):


        input_layer = Input(shape=(n_input,), dtype='float32', name=name)

        temp_output = Dense(self.n_width,activation=self.activation,kernel_regularizer=l2(self.l2weight),kernel_initializer=self.init,use_bias=True,name=name+'_0')(input_layer)
        #temp_output = MaxoutDense(self.n_width,kernel_regularizer=l2(self.l2weight),kernel_initializer=self.init,use_bias=True)(input_layer)
        #temp_output=PReLU()(temp_output)
        for i in range(1,self.n_depth):
            if self.dp_rate>0:
                temp_output=Dropout(self.dp_rate,name=name+'_dp_'+str(i))(temp_output)
            temp_output = Dense(self.n_width, activation=self.activation,kernel_regularizer=l2(self.l2weight),kernel_initializer=self.init,use_bias=True,name=name+'_'+str(i))(temp_output)
            #temp_output = PReLU()(temp_output)

        output_layer = Dense(1, kernel_initializer=self.init,kernel_regularizer=l2(self.l2weight),activation=self.output_layer_activation, use_bias=True,name=name+'_o')(temp_output)

        aux_input, merged_output = add_thresholded_output(output_layer, n_input, name)

        return aux_input, input_layer, merged_output, output_layer

    def generate_input_module_gas(self, n_depth, n_width, l2_weight, name, n_input, thresholded_output, n_inception=0):
        K.set_image_dim_ordering('th')
        input_layer = Input(shape=(n_input,), dtype='float32', name=name)


        mod1 = Dense(20, activation='relu', kernel_regularizer=l2(self.l2weight), kernel_initializer=INIT,
                     use_bias=True,trainable=True)(input_layer)
        for i in range(1, self.n_depth):
            mod1 = Dense(20, activation='relu', kernel_regularizer=l2(self.l2weight), kernel_initializer=INIT,
                         use_bias=True,trainable=True)(mod1)

        mod1_out = Dense(1, kernel_initializer=INIT, kernel_regularizer=l2(self.l2weight),
                             activation=self.output_layer_activation,
                             use_bias=True)(mod1)

        aux_input_mod1 = Input(shape=(1,), dtype='float32', name='OnOff_1_' + name)
        merged_output_mod1=multiply([aux_input_mod1, mod1_out])



        mod2 = Dense(20, activation='relu', kernel_regularizer=l2(self.l2weight), kernel_initializer=INIT,
                     use_bias=True, trainable=True)(input_layer)

        for i in range(1, self.n_depth):
            mod2 = Dense(20, activation='relu', kernel_regularizer=l2(self.l2weight), kernel_initializer=INIT,
                              use_bias=True, trainable=True)(mod2)

        mod2_out = Dense(1, kernel_initializer=INIT, kernel_regularizer=l2(self.l2weight),
                         activation=self.output_layer_activation,
                         use_bias=True)(mod2)

        aux_input_mod2 = Input(shape=(1,), dtype='float32', name='OnOff_2_' + name)
        merged_output_mod2 = multiply([aux_input_mod2, mod2_out])
        main_out = add([merged_output_mod2, merged_output_mod1],name=name + '_out')






        return aux_input_mod1,aux_input_mod2, input_layer, merged_output_mod1,merged_output_mod2,main_out

    def update_model(self):
        self.nb_epoch=10000
        self.output_layer_activation='relu'
        self.aux_inputs=[]
        self.inputs=[]
        self.merged_outputs=[]
        self.outputs=[]

        old_model=self.model
        self.initialize_model()
        weights=old_model.get_weights()
        self.model.set_weights(weights)


class ENSEMBLE(NN_BASE):



    def __init__(self,PATHS):

        self.model_name='NCNET2_ENSEMBLE_LEARNING'

        self.PATHS=PATHS


        self.output_layer_activation='relu'

        # Training config
        optimizer = 'adam'
        loss = 'mse'
        nb_epoch = 1
        batch_size = 64
        verbose = 0
        self.reg_constraint=False

        #Model config


        #Input module config
        self.n_inception =0
        self.n_depth = 2
        self.n_depth_incept=3
        self.n_width_incept=50
        self.n_width = 5


        self.l2weight = 0.1
        self.models=[]


        self.make_same_model_for_all=True
        self.add_thresholded_output=True

        self.input_tags = {}

        self.well_names = ['C1','C2', 'C3', 'C4','B1','B3','D1']

        tags = ['CHK','PWH','PBH','PDC']

        for name in self.well_names:

            self.input_tags[name] = []
            for tag in tags:
                if (name=='C2' or name=='D1') and tag=='PBH':
                    pass
                else:
                    self.input_tags[name].append(name + '_' + tag)


        OUT='GAS'
        if OUT=='GAS':
            self.output_tags = OIL_WELLS_QGAS_OUTPUT_TAGS
        else:
            self.output_tags = OIL_WELLS_QOIL_OUTPUT_TAGS

        self.loss_weights = {
            'B1_out':  0.0,
            'B3_out':  0.0,
            'C2_out':  0.0,
            'C3_out':  0.0,
            'D1_out':  0.0,
            'C4_out':  0.0,
            'C1_out':  0.0,

            'GJOA_TOTAL': 1.0,


        }

        super().__init__(n_width=0, n_depth=1, l2_weight=2, seed=2,
                         optimizer=optimizer, loss=loss, nb_epoch=nb_epoch, batch_size=batch_size)

    def initialize_model(self):
        print('Initializing %s' % (self.model_name))
        from keras.models import load_model
        self.models=[]
        for path in self.PATHS:
            print('Initializing '+path)
            self.models.append(load_model(path))
        self.model=self.models[0]

    def predict(self, X):
        predictions=[]

        for model in self.models:
            prediction=self.get_prediction(X,model)
            predictions.append(prediction)

        pred_concat=pd.concat(predictions,axis=1)
        pred_out=pd.DataFrame()

        for col in self.output_tag_ordered_list:
            pred_out[col]=pred_concat[col].mean(axis=1)

        return pred_out

    def get_prediction(self, X,model):
        X_dict,_=self.preprocess_data(X)
        predicted_data=model.predict(X_dict)
        #print(predicted_data[0])
        N_output_modules=len(self.output_tags.keys())

        #print(predicted_data.shape)



        if N_output_modules>1:
            predicted_data_reshaped=np.asarray(predicted_data[0])
            #print(predicted_data_reshaped)
            for i in range(1,N_output_modules):
                temp_data=np.asarray(predicted_data[i])
                predicted_data_reshaped=np.hstack((predicted_data_reshaped,temp_data))
        else:
            predicted_data_reshaped=np.asarray(predicted_data)
        #print(predicted_data_reshaped.shape)
        return pd.DataFrame(data=predicted_data_reshaped,columns=self.output_tag_ordered_list)
