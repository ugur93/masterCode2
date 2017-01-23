
from .base import *
from keras.models import load_model
class SSNET_EXTERNAL(object):

    def __init__(self,filename):

        #Config
        self.model_name = 'SSNET_EXTERNAL'
        self.nb_epoch=10
        self.batch_size=1000
        self.verbose=0
        self.history = LossHistory()
        self.Earlystopping=EarlyStopping(monitor='val_loss', min_delta=0.001, patience=100, verbose=0, mode='min')
        #self.TB=keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
        self.callbacks=[self.history,EpochVerbose()]

        self.model = None
        self.outputs = None
        self.inputs=None
        self.initializeModel(filename)
        plotModel(self.model,self.model_name)
    def initializeModel(self,filename):
        print ('Initializing %s' % (self.model_name))
        PATH='/Users/UAC/GITFOLDERS/MasterThesisCode/Models/NeuralNetworks/SavedModels/'
        self.model=load_model(PATH+filename+'.h5')
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

