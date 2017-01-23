
from .base import *
import keras
class SSNET1(object):

    def __init__(self):

        #Config
        self.model_name = 'SSNET1'
        self.l2weight = 0.005
        self.optimizer='adam'
        self.loss='mse'
        self.nb_epoch=1500
        self.batch_size=1000
        self.verbose=0
        self.history = LossHistory()
        self.Earlystopping=EarlyStopping(monitor='val_loss', min_delta=0.001, patience=100, verbose=0, mode='min')
        #self.TB=keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
        self.callbacks=[self.history,EpochVerbose()]

        self.model = None
        self.outputs = None
        self.inputs=None
        self.initializeModel()
        plotModel(self.model,self.model_name)
    def initializeModel(self):
        print ('Initializing %s' % (self.model_name))
        aux_input1,Input1, merged_out1, Out1 = generateInputModule(n_depth=2, n_width=5, n_input=1,n_inception=3, l2_weight=self.l2weight, name='chk1')
        aux_input2,Input2,merged_out2,  Out2 = generateInputModule(n_depth=2, n_width=5, n_input=1,n_inception=3, l2_weight=self.l2weight, name='chk2')
        aux_input3,Input3,merged_out3,  Out3 = generateInputModule(n_depth=2, n_width=5, n_input=1,n_inception=3, l2_weight=self.l2weight,name='chk3')
        aux_input4,Input4,merged_out4, Out4 = generateInputModule(n_depth=2, n_width=5, n_input=1,n_inception=3, l2_weight=self.l2weight,name='chk4')

        merged_input = merge([merged_out1, merged_out2, merged_out3, merged_out4], mode='sum')
        final_model=Dense(5, activation='relu', W_regularizer=l2(self.l2weight),W_constraint=unitnorm())(merged_input)
        final_model = Dense(5, activation='relu', W_regularizer=l2(self.l2weight), W_constraint=unitnorm())(
            final_model)
        #final_model=addLayers(merged_input,n_depth=1,n_width=5,l2_weight=self.l2weight)
        main_output=Dense(1,name='Q',W_regularizer=l2(self.l2weight),W_constraint=unitnorm())(final_model)

        self.outputs = [main_output, Out1, Out2, Out3, Out4,merged_out1,merged_out2,merged_out3,merged_out4]
        self.inputs = [Input1, Input2, Input3, Input4, aux_input1, aux_input2, aux_input3, aux_input4]
        self.model = Model(input=self.inputs, output=[main_output])
        self.model.compile(optimizer=self.optimizer, loss=self.loss)



    def fit(self, input, output):
        print ('Training model %s, please wait' % (self.model_name))
        input_dict=dfToDict(input)
        output_dict=dfToDict(output)
        input_toggled=addToggledInput(input_dict)
        self.model.fit(input_toggled, output_dict, nb_epoch=self.nb_epoch, batch_size=self.batch_size, verbose=self.verbose,
                       callbacks=self.callbacks)

    def predict(self, input):
        input_dict=dfToDict(input)
        input_toggled = addToggledInput(input_dict)
        return self.model.predict(input_toggled)

    def predictWellOutput(self, input, well):
        temp_model = Model(input=self.inputs[well - 1], output=self.outputs[well])
        temp_model.compile('adam', 'mse')
        return temp_model.predict(input)

    def getHistory(self):
        return self.history.losses

    def printFitConfig(self):
        print('Fit config: \n epoch: {} \n batch size: {} \n verbose: {} \n callbacks: {}'.format(self.nb_epoch,
                                                                                                  self.batch_size,
                                                                                                  self.verbose,
                                                                                                  self.callbacks))
    def get_layer_weights(self,layer_name):
        return self.model.get_layer(layer_name).get_weights()

    def save_model_to_file(self,name,save_weights=True):
        PATH='/Users/UAC/GITFOLDERS/MasterThesisCode/Models/NeuralNetworks/SavedModels/'
        if save_weights:
            self.model.save(PATH+name+'.h5')
        else:
            json_model=self.model.to_json()
            print(json_model)

