
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

        self.Earlystopping=EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=500, verbose=1, mode='min')
        #self.TB=keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
        self.callbacks=[self.history,EpochVerbose(),self.Earlystopping]

        #Model Params:
        self.model = None
        self.outputs = []
        self.merged_outputs=[]
        self.inputs=[]
        self.aux_inputs=[]


        self.chk_thresholds={}
        self.output_index,self.n_outputs = output_tags_to_index(self.output_tags)


        self.initialize_model()
        plotModel(self.model,self.model_name)

    def initialize_model(self):
        pass

    def preprocess_data(self,X,Y=[],Y_Q=[]):
        X_dict = df2dict(X,self.input_tags,self.output_tags,'X')
        X_dict = addToggledInput(X_dict, self.chk_thresholds)
        if len(Y)>0:
            Y_dict = df2dict(Y,self.input_tags,self.output_tags,'Y')
        else:
            Y_dict=[]
        return X_dict,Y_dict

    def fit(self, X, Y,X_val,Y_val):

        print(self.get_config())
        print ('Training model %s, please wait' % (self.model_name))
        print('Training data sample-size: '+str(len(X)))

        X_dict,Y_dict=self.preprocess_data(X,Y)
        X_val_dict,Y_val_dict=self.preprocess_data(X_val,Y_val)

        #self.debug(X_dict,Y_dict,True)


        self.model.fit(X_dict, Y_dict, nb_epoch=self.nb_epoch, batch_size=self.batch_size, verbose=self.verbose,
                       callbacks=self.callbacks,shuffle=True,validation_data=(X_val_dict,Y_val_dict))

    def predict(self, X,tag=False):
        X_dict,_=self.preprocess_data(X)
        predicted_data=self.model.predict(X_dict)

        data_tuple=()
        #print(predicted_data.shape)
        for i in range(self.n_outputs):
            data_tuple+=(predicted_data[i],)
        predicted_data_reshaped=np.hstack(data_tuple)
        if self.model_name=='SSNET3' or self.n_outputs==1:
            predicted_data_reshaped=np.array(predicted_data)


        if tag==False:
            return predicted_data_reshaped
        else:
            return predicted_data_reshaped[:,self.output_index[tag]]

    def get_history(self):
        return self.history.losses

    def get_layer_weights(self,layer_name):
        return self.model.get_layer(layer_name).get_weights()

    def save_model_to_file(self,name,scores,save_weights=True):
        PATH='/Users/UAC/GITFOLDERS/MasterThesisCode/Models/NeuralNetworks/SavedModels/'
        if save_weights:
            self.model.save(PATH+name+'.h5')
        else:
            json_model=self.model.to_json()
            print(json_model)

        #Save config:
        f=open(PATH+name+'_config','w')
        f.write(self.get_config())
        f.write('\n')
        f.write(scores)
        f.close()

        plotModel(self.model,name)

    def get_config(self):
        s = '-------------------------------------------------\n'
        s+='Input module Config: \n'
        s+=' n_depth: {} \n n_width: {} \n n_inception: {} \n l2_weight: {} \n Threshold: {} \n'.format(self.n_depth,self.n_width,self.n_inception,self.l2weight,self.add_thresholded_output)
        s+='-------------------------------------------------\n'
        s+='Fit config: \n epoch: {} \n batch size: {} \n verbose: {} \n callbacks: {} \n optimizer: {} \n'.format(self.nb_epoch,
                                                                                                  self.batch_size,
                                                                                                  self.verbose,
                                                                                                     self.callbacks,self.optimizer)
        s += '-------------------------------------------------\n'
        s+='Input tags: \n {} \n'.format(self.input_tags)
        s+='Output tags: \n {} \n'.format(self.output_tags)
        return s

    def initialize_chk_thresholds(self,data,scaled=True):
        col_length=len(data.X_transformed.columns)
        if scaled:
            thresh_transformed=data.transform([[self.threshold for i in range(col_length)]],'X')
        else:
            thresh_transformed=[[self.threshold for i in range(col_length)]]
        for key,tag_list in self.input_tags.items():
            tag=find_tag_that_ends_with(tag_list,'CHK')
            if tag:
                chk_index=data.X_transformed.columns.get_loc(tag)
                self.chk_thresholds[key]=thresh_transformed[0][chk_index]



    def debug(self,X_toggled,Y,plot=True):
        print('### DEBUG ###')
        print('CHK thresholds: ')
        print(self.chk_thresholds)
        print('-----------------------------------')
        print('OUTPUT INDEX')
        print(self.output_index)
        print('-----------------------------------')
        print('INPUT TAGS')
        print(self.input_tags)
        print('-----------------------------------')
        print('OUTPUT TAGS')
        print(self.output_tags)
        print('-----------------------------------')
        print('output_tags_to_list function')
        print(tags_to_list(self.output_tags))
        print('-----------------------------------')
        print('### --- ###')
        if plot:
            for i,key in zip(range(1,5),self.chk_thresholds):
                plt.subplot(2,2,i)
                plt.plot(X_toggled['aux_'+key],color='red')
                plt.plot(X_toggled[key],color='blue')
            plt.figure()
            plt.plot(Y['GJOA_QGAS'])
            plt.show()

    def evaluate(self,X_train,X_test,Y_train,Y_test):

        cols=tags_to_list(self.output_tags)
        cols2='GJOA_QGAS'
        score_test_MSE = metrics.mean_squared_error(Y_test[cols], self.predict(X_test), multioutput='raw_values')
        score_train_MSE = metrics.mean_squared_error(Y_train[cols], self.predict(X_train), multioutput='raw_values')
        score_test_r2 = metrics.r2_score(Y_test[cols], self.predict(X_test), multioutput='raw_values')
        score_train_r2 = metrics.r2_score(Y_train[cols], self.predict(X_train), multioutput='raw_values')

        return score_train_MSE,score_test_MSE,score_train_r2,score_test_r2

    def visualize_plot(self,X_train,X_test,Y_train,Y_test,output_cols=[]):

        if len(output_cols)==0:
            output_cols=tags_to_list(self.output_tags)

        for output_tag in output_cols:
            plt.figure()
            plt.plot(Y_train.index, Y_train[output_tag], color='blue', label=output_tag+'_true - train')
            plt.plot(Y_train.index, self.predict(X_train, output_tag), color='black', label=output_tag+'_pred - train')

            plt.plot(Y_test.index, Y_test[output_tag], color='red', label=output_tag+'_true - test')
            plt.plot(Y_test.index, self.predict(X_test, output_tag), color='green', label=output_tag+'_pred - test')

            # plt.legend(['Y_true - train','Y_pred - train','Y_true - test', 'Y_pred - test'],pos)
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                       ncol=2, mode="expand", borderaxespad=0., fontsize=20)


    def visualize_scatter(self,X_train,X_test,Y_train,Y_test,input_cols=[],output_cols=[]):
        if len(input_cols)==0:
            input_cols=tags_to_list(self.input_tags)
        if len(output_cols)==0:
            output_cols=tags_to_list(self.output_tags)

        len_input_cols=self.n_inputs
        #len_input_cols=len(input_cols)
        #len_output_cols=len(output_cols)

        if len_input_cols<=4:
            sp_y=2
            sp_x=int(len_input_cols/sp_y+0.5)
        else:
            sp_y = int(len_input_cols/2)
            sp_x = int(len_input_cols/sp_y+0.5)

        print(sp_y,sp_x,len_input_cols)
        i=1
        sp_x=2
        sp_y=2
        for output_tag in output_cols:
            #plt.figure()
            #i=1
            for input_tag in input_cols:
                if input_tag.split('_')[0]==output_tag.split('_')[0] and output_tag!='GJOA_QGAS':
                    plt.subplot(sp_y, sp_x, i)
                    i+=1
                    plt.scatter(X_train[input_tag], Y_train[output_tag], color='blue', label=output_tag+'_true - train')
                    plt.scatter(X_train[input_tag], self.predict(X_train, output_tag), color='black', label=output_tag+'_pred - train')

                    plt.scatter(X_test[input_tag], Y_test[output_tag], color='red', label=output_tag+'_true - test')
                    plt.scatter(X_test[input_tag], self.predict(X_test, output_tag), color='green', label=output_tag+'_pred - test')

                    plt.xlabel(input_tag)
                    plt.ylabel(output_tag)
                    plt.title(input_tag +' vs '+output_tag)

                    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                               ncol=2, mode="expand", borderaxespad=0., fontsize=15)

