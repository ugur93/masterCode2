from keras.models import Sequential
from keras.layers import Dense,Activation,Merge,Input,merge
from keras.models import Model
#from keras.utils.visualize_util import plot
import numpy as nn
from matplotlib import pyplot as plt







class SSnet1:

    def __init__(self):
        print 'init'
        self.model=None

        self.generateModel()

        self.plotModel()

    def generateModel(self):


        #Well1 network
        well1Input=Input(shape=(4,),dtype='float32',name='input1')
        well1SN=Dense(50,activation='relu')(well1Input)
        well1SN = Dense(50, activation='relu')(well1SN)
        well1SN = Dense(50, activation='relu')(well1SN)
        well1SN = Dense(50, activation='relu')(well1SN)
        well1SN = Dense(50, activation='relu')(well1SN)
        well1Out = Dense(1,name='Well1_output')(well1SN)

        #Well2 network
        well2Input = Input(shape=(4,), dtype='float32', name='input2')
        well2SN = Dense(50,activation='relu')(well2Input)
        well2SN = Dense(50, activation='relu')(well2SN)
        well2SN = Dense(50, activation='relu')(well2SN)
        well2SN = Dense(50, activation='relu')(well2SN)
        well2SN = Dense(50, activation='relu')(well2SN)
        well2Out=Dense(1,name='Well2_output')(well2SN)


        #Main model
        merged_model=merge([well1Out,well2Out],mode='sum')
        merged_model=Dense(50,activation='relu')(merged_model)
        merged_model = Dense(50, activation='relu')(merged_model)
        merged_model = Dense(50, activation='relu')(merged_model)
        main_ouput=Dense(1,name='main_output')(merged_model)


        self.model=Model(input=[well1Input,well2Input],output=[main_ouput])
        self.model.compile(optimizer='adam',loss='mse')



    def fit(self,input,output):
        self.model.fit(input,output,nb_epoch=100,batch_size=32,verbose=True)
    def predict(self,input):
        return self.model.predict(input)
    def plotModel(self):
        print self.model.summary()
        #plot(self.model, to_file='model.png',show_shapes=True)

class SSnet2:
    def __init__(self):
        print 'init'
        self.model=None

        self.generateModel()

        #self.plotModel()

    def generateModel(self):


        #Well1 network
        input1=Input(shape=(8,),dtype='float32',name='input1')
        x=Dense(50,activation='relu')(input1)
        x = Dense(50, activation='relu')(x)
        x = Dense(50, activation='relu')(x)
        x = Dense(50, activation='relu')(x)
        out1 = Dense(1,name='main_output')(x)


        self.model=Model(input=[input1],output=[out1])
        self.model.compile(optimizer='adam',loss='mse')



    def fit(self,input,output):
        print input.keys()[0]
        self.model.fit(input.values()[1],output,nb_epoch=100,batch_size=32,verbose=True)
    def predict(self,input):
        return self.model.predict(input)
    def plotModel(self):
        print 'here'
        plot(self.model, to_file='model.png',show_shapes=True)
class SSnet3:

    def __init__(self):

        self.model_name='SSNET3'
        print 'Initializing SSNET3'


        self.model=None

        self.generateModel()

        self.plotModel()

    def generateModel(self):


        #Well1 network
        well1Input=Input(shape=(4,),dtype='float32',name='input1')
        well1Out1=Dense(50,activation='relu')(well1Input)
        well1Out2=Dense(50,activation='relu')(well1Input)
        well1Out3 = Dense(50, activation='relu')(well1Input)
        well1Merge = merge([well1Out1, well1Out2,well1Out3], mode='concat')

        #Well2 network
        well2Input = Input(shape=(4,), dtype='float32', name='input2')
        well2Out1 = Dense(50, activation='relu')(well2Input)
        well2Out2 = Dense(50, activation='relu')(well2Input)
        well2Out3 = Dense(50, activation='relu')(well2Input)
        well2Merge = merge([well2Out1, well2Out2, well2Out3], mode='concat')


        #Main model
        merged_model=merge([well1Merge,well2Merge],mode='concat')
        merged_model1=Dense(50,activation='relu')(merged_model)
        merged_model2 = Dense(50, activation='relu')(merged_model)
        merged_model3 = Dense(50, activation='relu')(merged_model)
        final_model = merge([merged_model1, merged_model2,merged_model3], mode='concat')
        final_model=Dense(50, activation='relu')(final_model)
        main_output=Dense(1,name='main_output')(final_model)


        self.model=Model(input=[well1Input,well2Input],output=[main_output])
        self.model.compile(optimizer='adam',loss='mse')



    def fit(self,input,output):
        print 'Training model %s, please wait'%(self.model_name)
        self.model.fit(input,output,nb_epoch=100,batch_size=32,verbose=False)
    def predict(self,input):
        return self.model.predict(input)
    def plotModel(self):
        print self.model.summary()
        #plot(self.model, to_file='model.png',show_shapes=True)