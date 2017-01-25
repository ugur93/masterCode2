
from .base import *

class NN_BASE:

    def __init__(self,name,train_params):

        #Config
        self.model_name = name
        self.optimizer=train_params['optimizer']
        self.loss=train_params['loss']
        self.nb_epoch=train_params['nb_epoch']
        self.batch_size=train_params['batch_size']
        self.verbose=train_params['verbose']

        self.threshold=5
        self.history = LossHistory()

        self.Earlystopping=EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=1000, verbose=1, mode='min')
        #self.TB=keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
        self.callbacks=[self.history,EpochVerbose()]

        #Model Params:
        self.model = None
        self.outputs = []
        self.merged_outputs=[]
        self.inputs=[]
        self.aux_inputs=[]


        self.chk_thresholds={}

        self.input_names=['F1','B2','D3','E1']
        self.generate_input_tags()
        self.initialize_model()
        plotModel(self.model,self.model_name)

    def initialize_model(self):
        pass

    def preprocess_data(self,X,Y=[]):
        X_dict = df2dict(X,self.input_tags,'X')
        if len(Y)>0:
            Y_dict = df2dict(Y,self.input_tags,'Y')
            Y_dict = addDummyOutput(X_dict, Y_dict)
        else:
            Y_dict=[]
        X_dict = addToggledInput(X_dict, self.chk_thresholds)
        return X_dict,Y_dict

    def fit(self, X, Y,X_val=[],Y_val=[]):
        print ('Training model %s, please wait' % (self.model_name))
        print('Training data sample-size: '+str(len(X)))

        X_dict,Y_dict=self.preprocess_data(X,Y)
        #print(X_dict)
        #self.debug(X_dict,Y_dict)
        #Early stopping
        validation_data=None
        if len(X_val)>0 and len(Y_val)>0:
            X_val_dict,Y_val_dict=self.preprocess_data(X_val,Y_val)
            validation_data=(X_val_dict,Y_val_dict)
            self.callbacks.append(self.Earlystopping)
        self.model.fit(X_dict, Y_dict, nb_epoch=self.nb_epoch, batch_size=self.batch_size, verbose=self.verbose,
                       callbacks=self.callbacks,shuffle=False,validation_data=validation_data)

    def predict(self, X):
        X_dict,_=self.preprocess_data(X)
        #X_dict = addToggledInput(X_dict,self.chk_thresholds)
        return self.model.predict(X_dict)[0]

    def predict_well_output(self, X, tag):
        X_dict,_=self.preprocess_data(X)
        #X_dict = addToggledInput(X_dict,self.chk_thresholds)
        return self.model.predict(X_dict)[self.input_tags[tag][0]]

    def get_history(self):
        return self.history.losses


    def get_layer_weights(self,layer_name):
        return self.model.get_layer(layer_name).get_weights()

    def save_model_to_file(self,name,save_weights=True):
        PATH='/Users/UAC/GITFOLDERS/MasterThesisCode/Models/NeuralNetworks/SavedModels/'
        if save_weights:
            self.model.save(PATH+name+'.h5')
        else:
            json_model=self.model.to_json()
            print(json_model)

        #Save config:
        f=open(PATH+name+'_config','w')
        f.write(self.get_config())
        f.close()

        plotModel(self.model,name)

    def get_config(self):
        s = '-------------------------------------------------\n'
        s+='Input module Config: \n'
        s+=' n_depth: {} \n n_width: {} \n n_inception: {} \n l2_weight: {} \n Threshold: {} \n'.format(self.IM_n_depth,self.IM_n_width,self.IM_n_inception,self.l2weight,self.add_thresholded_output)
        s+='-------------------------------------------------\n'
        s+='Fit config: \n epoch: {} \n batch size: {} \n verbose: {} \n callbacks: {} \n optimizer: {} \n'.format(self.nb_epoch,
                                                                                                  self.batch_size,
                                                                                                  self.verbose,
                                                                                                     self.callbacks,self.optimizer)
        return s

    def initialize_thresholds(self,data,scaled=True):

        col_length=len(data.X_transformed.columns)
        if scaled:
            thresh_transformed=data.transform_using_scaler([[self.threshold for i in range(col_length)]],'X')
        else:
            thresh_transformed=[[self.threshold for i in range(col_length)]]
        for key,tag_tuple in self.input_tags.items():
            chk_index=data.X_transformed.columns.get_loc(tag_tuple[1])
            tag_name=tag_tuple[1].split('_')[0]
            self.chk_thresholds[tag_name]=thresh_transformed[0][chk_index]


    def debug(self,X_toggled,Y):
        for i,key in zip(range(1,5),self.chk_thresholds):
            plt.subplot(2,2,i)
            plt.plot(X_toggled['aux_'+key],color='red')
            plt.plot(X_toggled[key],color='blue')
        plt.figure()
        plt.plot(Y['GJOA_QGAS'])
        plt.show()

    def generate_input_tags(self):

        self.n_inputs=len(self.input_tags)
        temp_input_tags={}
        for i in range(len(self.input_names)+len(self.input_tags)-1):
            temp_input_tags[self.input_names[i]]=(i+1,)
            for tag_end in self.input_tags:
                temp_input_tags[self.input_names[i]]+=(self.input_names[i]+'_'+tag_end,)
        self.input_tags=temp_input_tags
        print(temp_input_tags)

