
from .base import *

class NN_BASE:

    def __init__(self):

        #Config
        self.chk_threshold_value=5

        self.history = LossHistory()
        self.Earlystopping=EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=500, verbose=1, mode='min')
        self.callbacks=[self.history,EpochVerbose(),self.Earlystopping]

        #Model Params:
        self.model = None
        self.outputs = []
        self.merged_outputs=[]
        self.inputs=[]
        self.aux_inputs=[]
        self.chk_thresholds={}

        self.n_outputs=len(tags_to_list(self.output_tags))
        self.initialize_model()

        self.output_index,self.output_tag_ordered_list,_ = output_tags_to_index(self.output_tags,self.model.get_config()['output_layers'])

        ordered_list, layer_names=layer_to_ordered_tag_list(self.input_tags,self.model.get_config()['input_layers'])
        #print(ordered_list)
        #print(layer_names)
        plotModel(self.model,self.model_name)

    def initialize_model(self):
        pass

    def preprocess_data(self,X,Y=[]):
        X_dict = df2dict(X,self.input_tags,self.output_tags,'X')
        if self.add_thresholded_output:
            X_dict = add_OnOff_state_input(X,X_dict, self.chk_thresholds)
        if len(Y)>0:
            Y_dict = df2dict(Y,self.input_tags,self.output_tags,'Y')
            #Y_dict['MAIN_OUT'] = Y_dict['MAIN_OUT'].reshape(Y_dict['MAIN_OUT'].shape[0],
            #                                                    Y_dict['MAIN_OUT'].shape[1])
        else:
            Y_dict=[]
        for key in X_dict.keys():
            #print(X_dict[key].shape,key)
            if key.split('_')[0]!='OnOff':
                X_dict[key]=X_dict[key].reshape(X_dict[key].shape[0],1,X_dict[key].shape[1])
        #X_dict['Main_input']=X_dict['Main_input'].reshape(X_dict['Main_input'].shape[0],1,X_dict['Main_input'].shape[1])

        return X_dict,Y_dict

    def fit(self, X, Y,X_val,Y_val):

        print(self.get_config())
        print ('Training model %s, please wait' % (self.model_name))
        print('Training data sample-size: '+str(len(X)))

        X_dict,Y_dict=self.preprocess_data(X,Y)
        X_val_dict,Y_val_dict=self.preprocess_data(X_val,Y_val)

        #self.debug(X_dict,Y_dict,True)

        self.model.fit(X_dict, Y_dict, nb_epoch=self.nb_epoch, batch_size=self.batch_size, verbose=self.verbose,
                       callbacks=self.callbacks,shuffle=False,validation_data=(X_val_dict,Y_val_dict))

        #print(self.model.get_weights())

    def predict(self, X):
        X_dict,_=self.preprocess_data(X)
        predicted_data=self.model.predict(X_dict)
        N_output_modules=len(self.output_tags.keys())

        if N_output_modules>1:
            predicted_data_reshaped=np.asarray(predicted_data[0])
            for i in range(1,N_output_modules):
                temp_data=np.asarray(predicted_data[i])
                predicted_data_reshaped=np.hstack((predicted_data_reshaped,temp_data))
        else:
            predicted_data_reshaped=np.asarray(predicted_data)

        return pd.DataFrame(data=predicted_data_reshaped,columns=self.output_tag_ordered_list)

    def get_history(self):
        return self.history.losses

    def get_layer_weights(self,layer_name):
        return self.model.get_layer(layer_name).get_weights()

    def save_model_to_file(self,name,scores,save_weights=True):
        PATH='./Models/NeuralNetworks/SavedModels/'
        if save_weights:
            self.model.save(PATH+name+'.h5')
        else:
            json_model=self.model.to_json()
            print(json_model)



        #plotModel(self.model,name)
    def save_model_config(self,scores):
        PATH='./Models/NeuralNetworks/ConfigFiles2/'
        # Save config:

        f_file_number=open(PATH+'file_number.txt','r+')
        file_data=f_file_number.read().splitlines()
        print(file_data)
        i,s=find_next_file_number(self.model_name,file_data)
        f_file_number.seek(0)
        f_file_number.write(s)
        f_file_number.truncate()
        f_file_number.close()

        f = open(PATH + self.model_name + '_config_'+str(i), 'w')
        f.write(self.get_config())
        f.write('\n')
        f.write(scores)
        f.close()
    def get_config(self):

        def tags_to_bulleted_list(s,tags):
            for key in tags:
                s+=key+': '+str(tags[key])+'\n'
            return s

        s='########### LAYER CONFIG ###########'
        sub_network_names, sub_network_config = self.get_layer_config()
        s+=layer_config_to_string(sub_network_names, sub_network_config)
        s+='\n\n'
        s+= '----------------------------------- \n'
        s+= '##### CHK thresholds ##### \n'
        s += '----------------------------------- \n'
        s=tags_to_bulleted_list(s,self.chk_thresholds)
        s+= '----------------------------------- \n'
        s+= '##### OUTPUT INDEX ##### \n'
        s += '----------------------------------- \n'
        s=tags_to_bulleted_list(s,self.output_index)
        s+= '-------------------------------------------------\n'
        s+= '##### Input-module config ##### \n'
        s+= '-------------------------------------------------\n'
        s+= '- n_depth: {} \n- n_width: {} \n- n_inception: {} \n- l2_weight: {} \n- OnOff_state: {} \n- Initialization: {} \n'.format(self.n_depth,self.n_width,self.n_inception,self.l2weight,self.add_thresholded_output,INIT)
        s+= '-------------------------------------------------\n'
        s+= '##### Fit config ##### \n'
        s+= '------------------------------------------------- \n'
        s+= '- epoch: {} \n- batch size: {} \n- verbose: {} \n- callbacks: {} \n- optimizer: {} \n'.format(self.nb_epoch,
                                                                                                  self.batch_size,
                                                                                                  self.verbose,
                                                                                                     self.callbacks,self.optimizer)
        s += '-------------------------------------------------\n'
        s+='##### Input tags ##### \n'
        s += '-------------------------------------------------\n'
        s=tags_to_bulleted_list(s,self.input_tags)
        s += '-------------------------------------------------\n'
        s+='##### Output tags ##### \n '
        s += '-------------------------------------------------\n'
        s=tags_to_bulleted_list(s,self.output_tags)
        s += '-------------------------------------------------\n'
        s += '-------------------------------------------------\n'
        return s

    def initialize_chk_thresholds(self,data,scaled=True):

        chk_cols=[]
        for tag in self.well_names:
            chk_cols.append(tag+'_'+'CHK')

        print(self.chk_threshold_value * np.ones((len(chk_cols),)))
        print(chk_cols)
        chk_threshold_data=pd.DataFrame(data=self.chk_threshold_value*np.ones((1,len(chk_cols))),columns=chk_cols)
        print(chk_threshold_data.shape)
        thresh_transformed=data.transform(chk_threshold_data)
        print(thresh_transformed)
        for col in thresh_transformed.columns:
            key=col.split('_')[0]
            self.chk_thresholds[key]=thresh_transformed[col][0]





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
            for i,key in zip(range(1,8),self.chk_thresholds):
                plt.subplot(3,3,i)
                plt.plot(X_toggled['OnOff_'+key],color='red')
                plt.plot(X_toggled[key],color='blue')
                plt.title(key)
            #plt.figure()
            #plt.plot(Y['GJOA_TOTAL_QOIL'])
            plt.show()

    def get_chk_threshold(self):
        return self.chk_thresh_val

    def evaluate(self,data,X_train,X_test,Y_train,Y_test):

        cols=self.output_tag_ordered_list
        print(cols)
        score_test_MSE = metrics.mean_squared_error(data.inverse_transform(Y_test)[cols], data.inverse_transform(self.predict(X_test)), multioutput='raw_values')
        score_train_MSE = metrics.mean_squared_error(data.inverse_transform(Y_train)[cols], data.inverse_transform(self.predict(X_train)), multioutput='raw_values')
        score_test_r2 = metrics.r2_score(Y_test[cols], self.predict(X_test), multioutput='raw_values')
        score_train_r2 = metrics.r2_score(Y_train[cols], self.predict(X_train), multioutput='raw_values')

        return score_train_MSE,score_test_MSE,score_train_r2,score_test_r2,cols

    def evaluate_zeros(self,X_train,X_test,Y_train,Y_test):
        cols = self.output_tag_ordered_list
        score_test_MSE=[]
        score_train_MSE=[]
        score_test_r2=[]
        score_train_r2=[]
        for col in cols:
            well=col.split('_')[0]
            ind_train = X_train[well + '_CHK'] > 0.05
            ind_test = X_test[well + '_CHK'] > 0.05
            score_test_MSE.append(metrics.mean_squared_error(self.SCALE * Y_test[col][ind_test], self.SCALE * self.predict(X_test[ind_test],col),
                                                        multioutput='raw_values'))
            score_train_MSE.append(metrics.mean_squared_error(self.SCALE * Y_train[col][ind_train], self.SCALE * self.predict(X_train[ind_train],col),
                                                         multioutput='raw_values'))
            score_test_r2.append(metrics.r2_score(Y_test[col][ind_test], self.predict(X_test[ind_test],col), multioutput='raw_values'))
            score_train_r2.append(metrics.r2_score(Y_train[col][ind_train], self.predict(X_train[ind_train],col), multioutput='raw_values'))
        return score_train_MSE, score_test_MSE, score_train_r2, score_test_r2

    def get_layer_config(self):

        def find_layer_name_index(lst,layer_name):
            indexes=[]
            for sub in lst:
                for name in sub:
                    if name==layer_name:
                        indexes.append(lst.index(sub))
            return indexes

        layers=self.model.get_config()['layers']
        n_layers=len(layers)
        sub_network_config=[]
        sub_network_names=[]

        for input_layer in self.model.get_config()['input_layers']:
            sub_network_names.append([input_layer[0]])
            sub_network_config.append([input_layer[0]])

        for i in range(1,n_layers):

            for name_list in layers[i]['inbound_nodes']:
                for inbounds in name_list:
                    name_inbound=inbounds[0]
                    name=layers[i]['name']
                    #print(i,name,name_inbound)
                    indexes=find_layer_name_index(sub_network_names,name_inbound)

                    for k in indexes:
                        sub_network_names[k].append(name)
                        sub_network_config[k].append(layers[i]['config'])


        return sub_network_names,sub_network_config