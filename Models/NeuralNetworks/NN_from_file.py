
from .base import *
from .base_class import NN_BASE

import keras.backend as K
from keras.models import load_model
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

def abs(x):
    return K.abs(x)
class NCNET_FILE(NN_BASE):



    def __init__(self,model_path):

        self.model_name='NCNET_GAS_PRETRAINED_WITH_OLD_DATA'

        self.model_path=model_path


        self.output_layer_activation='linear'

        # Training config
        self.optimizer = 'adam'
        self.loss = 'mse'
        self.nb_epoch = 10000
        self.batch_size = 64
        self.verbose = 0

        # Input module config
        self.n_depth = 2
        self.n_width = 20
        self.l2weight = 0.0001
        self.add_thresholded_output = True

        #Model inputs/ouputs config
        self.input_tags = {}

        self.well_names =['F1','B2','D3','E1']

        tags = ['CHK','PDC','PWH','PBH']

        for name in self.well_names:

            self.input_tags[name] = []
            for tag in tags:
                if (name=='C2' or name=='D1') and tag=='PBH':
                    pass
                else:
                    self.input_tags[name].append(name + '_' + tag)

        self.output_tags = {
            'F1_out': ['F1_QGAS'],
            'B2_out': ['B2_QGAS'],
            'D3_out': ['D3_QGAS'],
            'E1_out': ['E1_QGAS'],
            'GJOA_QGAS': ['GJOA_QGAS']
        }
        self.loss_weights = {
            'F1_out': 0.0,
            'B2_out': 0.0,
            'D3_out': 0.0,
            'E1_out': 0.0,
            'GJOA_QGAS': 1.0
        }


        super().__init__()

    def initialize_model(self):
        print('Initializing %s' % (self.model_name))

        self.model = load_model(self.model_path+'.h5')






