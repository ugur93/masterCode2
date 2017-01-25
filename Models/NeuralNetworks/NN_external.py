
from .base import *
from keras.models import load_model
from .base_class import NN_BASE


class SSNET_EXTERNAL(NN_BASE):

    def __init__(self,filename):

        name = 'SSNET_EXTERNAL'

        self.filename=filename
        self.PATH = '/Users/UAC/GITFOLDERS/MasterThesisCode/Models/NeuralNetworks/SavedModels/'

        # Training config
        optimizer = 'nadam'
        loss = 'mse'
        nb_epoch = 10000
        batch_size = 1000
        verbose = 0

        train_params = {'optimizer': optimizer, 'loss': loss, 'nb_epoch': nb_epoch, 'batch_size': batch_size,
                        'verbose': verbose}

        super().__init__(name, train_params)

    def initialize_model(self):
        print ('Initializing %s' % (self.model_name))
        self.model=load_model(self.PATH+self.filename+'.h5')


