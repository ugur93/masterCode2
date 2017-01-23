from keras.models import Sequential
from keras.layers import Dense,Activation,Merge,Input,merge
from keras.models import Model
from keras.utils.visualize_util import plot
from sklearn.svm import SVR
from keras.regularizers import l2

from keras.callbacks import Callback, EarlyStopping
import numpy as np
from matplotlib import pyplot as plt







class SSnet1:

    def __init__(self):

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
        print(self.model.summary())
        #plot(self.model, to_file='model.png',show_shapes=True)

class SSnet2:
    def __init__(self):

        self.model=None

        self.generateModel()

        #self.plotModel()

    def generateModel(self):


        #Well1 network
        input1=Input(shape=(4,),dtype='float32',name='input1')
        x=Dense(5,activation='relu')(input1)
        x = Dense(5, activation='relu')(x)
        #x = Dense(50, activation='relu')(x)
        #x = Dense(50, activation='relu')(x)
        out1 = Dense(1,name='main_output')(x)


        self.model=Model(input=[input1],output=[out1])
        self.model.compile(optimizer='adam',loss='mse')



    def fit(self,input,output):
        self.model.fit(input,output,nb_epoch=5000,batch_size=1000,verbose=False)
    def predict(self,input):
        return self.model.predict(input)
    def plotModel(self):
        print('here')
        plot(self.model, to_file='model.png',show_shapes=True)
class SSnet3:

    def __init__(self):

        self.model_name='SSNET3'
        print ('Initializing SSNET3')


        self.model=None

        self.generateModel()

        self.plotModel()

    def generateModel(self):

        #Well1 network
        well1Input = Input(shape=(4,), dtype='float32', name='input1')
        well1Output=self.getInceptionModule(well1Input)

        #Well2 network
        well2Input = Input(shape=(4,), dtype='float32', name='input2')
        well2Output= self.getInceptionModule(well2Input)

        well3Input = Input(shape=(4,), dtype='float32', name='input3')
        well3Output = self.getInceptionModule(well3Input)

        well4Input = Input(shape=(4,), dtype='float32', name='input4')
        well4Output = self.getInceptionModule(well4Input)

        #Main model
        merged_model=merge([well1Output,well2Output],mode='concat')
        final_model=self.getInceptionModule(merged_model)
        final_model=self.addLayers(final_model,50,3)

        main_output=Dense(1,name='main_output')(final_model)

        #Compile the model
        self.model=Model(input=[well1Input,well2Input,well3Input,well4Input],output=[main_output])
        self.model.compile(optimizer='adam',loss='mse')

    def getInceptionModule(self,input):
        out1 = Dense(20, activation='relu')(input)
        out2 = Dense(20, activation='relu')(input)
        out3 = Dense(20, activation='relu')(input)
        output_merged = merge([out1, out2, out3], mode='concat')

        return output_merged
    def addLayers(self,model,width,depth):
        for i in range(depth):
            model=Dense(width,activation='relu')(model)
        return model
    def fit(self,input,output):
        print ('Training model %s, please wait'%(self.model_name))
        self.model.fit(input,output,nb_epoch=100,batch_size=32,verbose=False)
    def predict(self,input):
        return self.model.predict(input)
    def plotModel(self):
        #print self.model.summary()
        plot(self.model, to_file='model.png',show_shapes=True)


class SVM1:
    def __init__(self):

        self.model_name='SVM1'
        print ('Initializing %s'%(self.model_name))


        self.model=None

        self.C=500
        self.epsilon=0.1
        self.gamma='auto'

        self.generateModel()


    def generateModel(self):
        self.model=SVR(C=self.C,epsilon=self.epsilon,gamma=self.gamma)

    def fit(self,input,output):
        X=input['input1']
        Y=output['main_output']
        self.model.fit(X,Y)

    def predict(self,input):
        X=input['input1']
        return self.model.predict(X)



class SSnet4:

    def __init__(self):

        self.model_name='SSNET4'
        print ('Initializing %s'%(self.model_name))

        self.history=LossHistory()
        self.l2weight=0.01
        self.model=None
        self.outputs=None
        self.generateModel()
        self.plotModel()

    def generateModel(self):

        #Well1 network
        Input1,Out1=self.createInceptionModule(n_depth=1,n_width=5,n_input=1,name='chk1')
        Input2, Out2 = self.createInceptionModule(n_depth=1, n_width=5, n_input=1, name='chk2')
        Input3, Out3 = self.createInceptionModule(n_depth=1, n_width=5, n_input=1, name='chk3')
        Input4, Out4 = self.createInceptionModule(n_depth=1, n_width=5, n_input=1, name='chk4')

        #Main model
        merged_model=merge([Out1,Out2,Out3,Out4],mode='sum')
        #final_model=Dense(5,activation='relu',W_regularizer=l2(self.l2weight),trainable=True)(merged_model)
        #final_model = Dense(5, activation='relu', W_regularizer=l2(self.l2weight), trainable=True)(final_model)

        main_output=Dense(1,name='Q',trainable=False)(merged_model)

        self.outputs = [main_output,Out1, Out2, Out3, Out4]
        self.inputs  = [Input1,Input2,Input3,Input4]

        #Compile the model
        self.model=Model(input=[Input1,Input2,Input3,Input4],output=[main_output])
        self.model.compile(optimizer='adam',loss='mse')

    def createInceptionModule(self,n_depth,n_width,n_input,name):
        input = Input(shape=(n_input,), dtype='float32', name=name)
        out1 = Dense(n_width, activation='relu',W_regularizer=l2(self.l2weight))(input)
        out1=Dense(n_width, activation='relu',W_regularizer=l2(self.l2weight))(out1)
        out2 = Dense(n_width, activation='relu',W_regularizer=l2(self.l2weight))(input)
        out2 = Dense(n_width, activation='relu', W_regularizer=l2(self.l2weight))(out2)
        out3 = Dense(n_width, activation='relu',W_regularizer=l2(self.l2weight))(input)
        out3 = Dense(n_width, activation='relu', W_regularizer=l2(self.l2weight))(out3)
        output_merged = merge([out1, out2, out3], mode='concat')

        output=Dense(1)(output_merged)

        return input,output
    def addLayers(self,model,width,depth):
        for i in range(depth):
            model=Dense(width,activation='relu')(model)
        return model
    def createInputModule(self,n_depth,n_width,n_input,name):
        input = Input(shape=(n_input,), dtype='float32', name=name)
        output = Dense(n_width, activation='relu', W_regularizer=l2(self.l2weight))(input)
        for i in range(1,n_depth):
            output=Dense(n_width,activation='relu',W_regularizer=l2(self.l2weight))(output)
        output = Dense(1)(output)
        return input,output

    #def createInput
    def fit(self,input,output):
        nb_epoch=3000
        batch_size=1000
        verbose=False
        callbacks=[self.history]
        print ('Training model %s, please wait'%(self.model_name))
        print ('Fit config: \n epoch: {} \n batch size: {} \n verbose: {} \n callbacks: {}'.format(nb_epoch,batch_size,verbose,callbacks))
        self.model.fit(input,output,nb_epoch=nb_epoch,batch_size=batch_size,verbose=verbose,callbacks=[self.history])
    def predict(self,input):

        return self.model.predict(input)
    def predictWellOutput(self,input,well):
        temp_model=Model(input=self.inputs[well-1],output=self.outputs[well])
        temp_model.compile('adam','mse')
        #plot(temp_model, to_file='model.png', show_shapes=True)
        return temp_model.predict(input)

    def plotModel(self):
        #print self.model.summary()
        plot(self.model, to_file='model.png',show_shapes=True)
    def getHistory(self):
        return self.history.losses



class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
