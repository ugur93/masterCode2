
from .base import *

class NN_BASE:

    def __init__(self,n_depth,n_width,l2_weight,seed,optimizer,loss,nb_epoch,batch_size,dp_rate=0):


        #Variables:
        self.output_zero_thresholds={}
        self.chk_thresh_val=None
        self.chk_thresholds = {}



        #Model Config
        self.chk_threshold_value=5
        self.do_shuffle=True
        self.n_depth=n_depth
        self.n_width=n_width
        self.l2weight=l2_weight
        self.init=glorot_normal(seed)
        self.add_thresholded_output = True

        self.optimizer=optimizer
        self.loss=loss
        self.nb_epoch=nb_epoch
        self.batch_size=batch_size
        self.verbose=0
        self.seed=seed
        self.dp_rate=dp_rate



        #Fit callbacks config
        self.history = History()
        self.Earlystopping=CustomEarlyStopping(monitor='val_loss', min_delta=0.0001, patience=500, verbose=1, mode='min')
        self.callbacks=[self.history,EpochVerbose(),self.Earlystopping]

        #Model Params:
        self.model = None

        self.n_outputs=len(tags_to_list(self.output_tags))


        self.initialize_model()

        self.output_index,self.output_tag_ordered_list,_ = output_tags_to_index(self.output_tags,self.model.get_config()['output_layers'])

        #ordered_list, layer_names=layer_to_ordered_tag_list(self.input_tags,self.model.get_config()['input_layers'])


        plotModel(self.model,self.model_name)
    def get_params(self,deep=False):
        return {'n_depth': self.n_depth, 'n_width': self.n_width,
                     'l2w': self.l2weight, 'seed': self.seed}

    def set_params(self):
        return
        #self.n_depth=params['n_depth']
        #self.n_width=params['n_width']
        #self.l2weight=params['l2w']
        #self.seed=params['seed']
        #self.initialize_model()
    def initialize_model(self):
        pass

    def preprocess_data(self,X,Y=[]):
        X_dict = df2dict(X,self.input_tags,self.output_tags,'X')
        if self.add_thresholded_output:
            X_dict = add_OnOff_state_input(X,X_dict, self.chk_thresholds)
            #X_dict=add_output_threshold_input(X,X_dict,self.output_zero_thresholds)
       # print(X_dict.keys())
        if len(Y)>0:
            Y_dict = df2dict(Y,self.input_tags,self.output_tags,'Y')
        else:
            Y_dict=[]

        return X_dict,Y_dict

    def fit(self, X, Y,X_val,Y_val):


        print ('Training model %s, please wait' % (self.model_name))
        print('Training data sample-size: '+str(len(X)))

        X_dict,Y_dict=self.preprocess_data(X,Y)
        X_val_dict,Y_val_dict=self.preprocess_data(X_val,Y_val)

        #self.debug(X_dict,Y_dict,True)

        self.model.fit(X_dict, Y_dict, epochs=self.nb_epoch, batch_size=self.batch_size, verbose=self.verbose,
                       callbacks=self.callbacks,shuffle=self.do_shuffle,validation_data=(X_val_dict,Y_val_dict))

        #print(self.model.get_weights())

    def predict(self, X):
        X_dict,_=self.preprocess_data(X)
        predicted_data=self.model.predict(X_dict)
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



    def get_history(self):
        return self.history.losses

    def get_weights(self):
        return self.model.get_weights()
    def set_weights(self,init_weights):
        self.model.set_weights(init_weights)

    def get_layer_weights(self,layer_name):
        return self.model.get_layer(layer_name).get_weights()

    def save_model_to_file(self,name):
        PATH='./Models/NeuralNetworks/SavedModels2/'

        self.model.save(PATH+'hdf5_files/'+name+'.h5')

        json_model=self.model.to_json()
        f = open(PATH + 'Architecture/'+name, 'w')
        f.write(json_model)
        f.close()

        self.model.save_weights(PATH+'Weights/'+name+'.h5')
        #print(json_model)



        #plotModel(self.model,name)

    def save_model_config(self,scores):
        PATH='./Models/NeuralNetworks/ConfigFiles2/'
        # Save config:

        f_file_number=open(PATH+'file_number.txt','r+')
        file_data=f_file_number.read().splitlines()
        #print(file_data)
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
        s+= '- n_depth: {} \n- n_width: {}\n- dp_rate: {} \n- l2_weight: {} \n' \
            '- OnOff_state: {} \n- Initialization: {} \n- Seed: {}\n'.format(self.n_depth,self.n_width,self.dp_rate,self.l2weight,self.add_thresholded_output,self.init,self.seed)
        s+= '-------------------------------------------------\n'
        s+= '##### Fit config ##### \n'
        s+= '------------------------------------------------- \n'
        s+= '- epoch: {} \n- batch size: {} \n- verbose: {} \n- callbacks: {} \n- optimizer: {} \n- Loss: {}\n- Shuffle {} \n'.format(self.nb_epoch,
                                                                                                  self.batch_size,
                                                                                                  self.verbose,
                                                                                                     self.callbacks,self.optimizer,self.loss,self.do_shuffle)
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

        chk_threshold=pd.DataFrame(data=self.chk_threshold_value*np.ones((1,len(chk_cols))),columns=chk_cols)
        thresh_transformed=data.transform(chk_threshold,'X')

        for col in thresh_transformed.columns:
            key=col.split('_')[0]
            self.chk_thresholds[key]=thresh_transformed[col][0]


    def initialize_zero_thresholds(self,data):

        cols=tags_to_list(self.output_tags)
        print(cols)
        thresh_data=pd.DataFrame(data=np.zeros((1,len(cols))),columns=cols)
        temp_output_zero_thresholds=data.transform(thresh_data,'Y')

        for col in thresh_data.columns:
            key = col.split('_')[0]
            self.output_zero_thresholds[key]=temp_output_zero_thresholds[col][0]
        print(self.output_zero_thresholds)




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

    def evaluate(self,data,X_train,X_test,Y_train,Y_test,as_dict=False):

        cols=self.output_tag_ordered_list
        Y_test_pred=data.inverse_transform(self.predict(X_test),'Y').set_index(Y_test.index)
        Y_train_pred=data.inverse_transform(self.predict(X_train),'Y').set_index(Y_train.index)


        score_test_MSE = metrics.mean_squared_error(data.inverse_transform(Y_test,'Y')[cols], Y_test_pred, multioutput='raw_values')
        score_train_MSE = metrics.mean_squared_error(data.inverse_transform(Y_train,'Y')[cols],Y_train_pred , multioutput='raw_values')
        score_test_r2 = metrics.r2_score(data.inverse_transform(Y_test,'Y')[cols], Y_test_pred[cols], multioutput='raw_values')
        score_train_r2 = metrics.r2_score(data.inverse_transform(Y_train,'Y')[cols],Y_train_pred[cols], multioutput='raw_values')
        if as_dict:
            score_train_MSE = pd.DataFrame(data=score_train_MSE, index=cols).T
            score_test_MSE = pd.DataFrame(data=score_test_MSE, index=cols).T

            score_train_r2 = pd.DataFrame(data=score_train_r2, index=cols).T
            score_test_r2 = pd.DataFrame(data=score_test_r2, index=cols).T

            score_test_MSE = metrics.mean_squared_error(data.inverse_transform(Y_test, 'Y')[cols],
                                                        data.inverse_transform(self.predict(X_test), 'Y'),
                                                        multioutput='raw_values')
            score_train_MSE = metrics.mean_squared_error(data.inverse_transform(Y_train, 'Y')[cols],
                                                         data.inverse_transform(self.predict(X_train), 'Y'),
                                                         multioutput='raw_values')

            return {'MSE_train':score_train_MSE,'MSE_test':score_test_MSE,'R2_train':score_train_r2,'R2_test':score_test_r2}
        else:
            return score_train_MSE,score_test_MSE,score_train_r2,score_test_r2,cols

    def evaluate_single(self,data,X,Y):

        cols=self.output_tag_ordered_list
        score_MSE = metrics.mean_squared_error(data.inverse_transform(Y,'Y')[cols], data.inverse_transform(self.predict(X),'Y'), multioutput='raw_values')
        score_r2 = metrics.r2_score(Y[cols], self.predict(X), multioutput='raw_values')

        return score_MSE,score_r2,cols
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