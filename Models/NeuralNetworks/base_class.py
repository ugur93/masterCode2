
from .base import *

class NN_BASE:

    def __init__(self):

        #Config
        self.chk_threshold_value=2
        self.history = LossHistory()

        self.Earlystopping=EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=500, verbose=1, mode='min')
        #self.TB=keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
        self.callbacks=[self.history,EpochVerbose()]#,self.Earlystopping]

        #Model Params:
        self.model = None
        self.outputs = []
        self.merged_outputs=[]
        self.inputs=[]
        self.aux_inputs=[]


        self.chk_thresholds={}

        self.initialize_model()

        self.output_index,self.output_tag_ordered_list,self.n_outputs = output_tags_to_index(self.output_tags,self.model.get_config()['output_layers'])
        print(self.output_tag_ordered_list)

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
        #print(self.model.summary())
        print ('Training model %s, please wait' % (self.model_name))
        print('Training data sample-size: '+str(len(X)))

        X_dict,Y_dict=self.preprocess_data(X,Y)
        X_val_dict,Y_val_dict=self.preprocess_data(X_val,Y_val)

       # exit()

        #self.debug(X_dict,Y_dict,True)


        self.model.fit(X_dict, Y_dict, nb_epoch=self.nb_epoch, batch_size=self.batch_size, verbose=self.verbose,
                       callbacks=self.callbacks,shuffle=True,validation_data=(X_val_dict,Y_val_dict))

    def predict(self, X,tag=False):
        X_dict,_=self.preprocess_data(X)
        predicted_data=self.model.predict(X_dict)

        predicted_data_reshaped=np.asarray(predicted_data)
        if self.n_outputs>1:
            predicted_data_reshaped=predicted_data_reshaped.T[0,:,:]

        if tag==False:
            return predicted_data_reshaped
        else:
            return predicted_data_reshaped[:,self.output_index[tag]]

    def get_history(self):
        return self.history.losses

    def get_layer_weights(self,layer_name):
        return self.model.get_layer(layer_name).get_weights()

    def save_model_to_file(self,name,scores,save_weights=True):
        #PATH='/Users/UAC/GITFOLDERS/MasterThesisCode/Models/NeuralNetworks/SavedModels/'
        PATH='C:/users/ugurac/Documents/GITFOLDERS/MasterThesisCode/Models/NeuralNetworks/model_figures'
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
        s='CHK thresholds: \n'
        s+=str(self.chk_thresholds)+'\n'
        s+='----------------------------------- \n'
        s+='OUTPUT INDEX \n'
        s+=str(self.output_index)+'\n'
        s+= '-------------------------------------------------\n'
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
        self.SCALE=data.Y_SCALE
        if scaled:
            thresh_transformed=data.transform([[self.chk_threshold_value for i in range(col_length)]],'X')
        else:
            thresh_transformed=[[self.chk_threshold_value for i in range(col_length)]]
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
            for i,key in zip(range(1,8),self.chk_thresholds):
                plt.subplot(3,3,i)
                plt.plot(X_toggled['aux_'+key],color='red')
                plt.plot(X_toggled[key],color='blue')
                plt.title(key)
            #plt.figure()
            #plt.plot(Y['GJOA_TOTAL_QOIL'])
            plt.show()

    def evaluate(self,X_train,X_test,Y_train,Y_test):

        cols=self.output_tag_ordered_list
        print(cols)
        score_test_MSE = metrics.mean_squared_error(self.SCALE*Y_test[cols], self.SCALE*self.predict(X_test), multioutput='raw_values')
        score_train_MSE = metrics.mean_squared_error(self.SCALE*Y_train[cols], self.SCALE*self.predict(X_train), multioutput='raw_values')
        score_test_r2 = metrics.r2_score(Y_test[cols], self.predict(X_test), multioutput='raw_values')
        score_train_r2 = metrics.r2_score(Y_train[cols], self.predict(X_train), multioutput='raw_values')

        return score_train_MSE,score_test_MSE,score_train_r2,score_test_r2




    def visualize(self,X_train,X_test,Y_train,Y_test,output_cols=[],input_cols=[]):


        #self.plot_scatter_input_output(X_train, X_test, Y_train, Y_test, input_cols=input_cols,output_cols=output_cols)
        #self.plot_residuals(X_train, X_test, Y_train, Y_test, output_cols)
        self.plot_true_and_predicted(X_train, X_test, Y_train, Y_test, output_cols)
        #self.plot_true_and_predicted_with_input(X_train, X_test, Y_train, Y_test, output_cols=output_cols)
        plt.show()


    def plot_residuals(self,X_train,X_test,Y_train,Y_test,output_cols=[]):

        if len(output_cols)==0:
            output_cols=self.output_tag_ordered_list

        sp_y=int(len(output_cols)/2+0.5)
        sp_x=int(len(output_cols)/sp_y+0.5)
        print(sp_y,sp_x,len(output_cols))

        i = 1
        plt.figure()
        for output_tag in output_cols:
            if output_tag == 'GJOA_QGAS':
                plt.figure()
            else:
                plt.subplot(sp_y, sp_x, i)
                i += 1

            plt.scatter(Y_train.index,
                        self.SCALE * Y_train[output_tag] - self.SCALE * self.predict(X_train, output_tag),
                        color='black',
                        label=output_tag + '_train_(true-pred)')

            plt.scatter(Y_test.index, self.SCALE * Y_test[output_tag] - self.SCALE * self.predict(X_test, output_tag),
                        color='green',
                        label=output_tag + '__test_(true-pred)')
            plt.title(output_tag + '-' + 'Residuals')
            # plt.legend(['Y_true - train','Y_pred - train','Y_true - test', 'Y_pred - test'],pos)
            # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            #           ncol=2, mode="expand", borderaxespad=0., fontsize=10)

    def plot_true_and_predicted(self,X_train,X_test,Y_train,Y_test,output_cols=[]):

        if len(output_cols)==0:
            output_cols=self.output_tag_ordered_list

        sp_y=int(len(output_cols)/2+0.5)
        sp_x=int(len(output_cols)/sp_y+0.5)
        print(sp_y,sp_x,len(output_cols))

        i=1
        plt.figure()
        for output_tag in output_cols:
            if output_tag=='GJOA_QGAS':
                plt.figure()
            else:
                plt.subplot(sp_y,sp_x,i)
                i+=1
            plt.scatter(Y_train.index, self.SCALE*Y_train[output_tag], color='blue', label=output_tag+'_true - train')
            plt.scatter(Y_train.index, self.SCALE*self.predict(X_train, output_tag), color='black', label=output_tag+'_pred - train')

            plt.scatter(Y_test.index, self.SCALE*Y_test[output_tag], color='red', label=output_tag+'_true - test')
            plt.scatter(Y_test.index, self.SCALE*self.predict(X_test, output_tag), color='green', label=output_tag+'_pred - test')
            plt.title(output_tag)
            # plt.legend(['Y_true - train','Y_pred - train','Y_true - test', 'Y_pred - test'],pos)
            #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            #           ncol=2, mode="expand", borderaxespad=0., fontsize=10)

    def plot_scatter_input_output(self,X_train,X_test,Y_train,Y_test,input_cols=[],output_cols=[]):
        if len(input_cols)==0:
            input_cols=tags_to_list(self.input_tags)
        if len(output_cols)==0:
            output_cols=self.output_tag_ordered_list

        i=1
        sp_x=count_n_well_inputs(input_cols)
        sp_y=1
        print(sp_y, sp_x)
        for output_tag in output_cols:
            plt.figure()
            i=1
            for input_tag in input_cols:
                if input_tag.split('_')[0]==output_tag.split('_')[0] and output_tag!='GJOA_QGAS':
                    plt.subplot(sp_y, sp_x, i)
                    i+=1
                    plt.scatter(X_train[input_tag], self.SCALE*Y_train[output_tag], color='blue', label=output_tag+'_true - train')
                    plt.scatter(X_train[input_tag], self.SCALE*self.predict(X_train, output_tag), color='black', label=output_tag+'_pred - train')

                    plt.scatter(X_test[input_tag], self.SCALE*Y_test[output_tag], color='red', label=output_tag+'_true - test')
                    plt.scatter(X_test[input_tag], self.SCALE*self.predict(X_test, output_tag), color='green', label=output_tag+'_pred - test')

                    plt.xlabel(input_tag)
                    plt.ylabel(output_tag)
                    plt.title(input_tag +' vs '+output_tag)

                    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                               ncol=2, mode="expand", borderaxespad=0., fontsize=15)


    def plot_true_and_predicted_with_input(self,X_train,X_test,Y_train,Y_test,output_cols=[]):

        if len(output_cols)==0:
            output_cols=tags_to_list(self.output_tags)
        input_cols=tags_to_list(self.input_tags)

        sp_y=count_n_well_inputs(input_cols)+1
        sp_x=1
        print(sp_y,sp_x,len(input_cols))

        i=1

        for output_tag in output_cols:
            plt.figure()
            i=1
            tag = output_tag
            plt.subplot(sp_y, sp_x, i)
            plt.scatter(Y_train.index, self.SCALE * Y_train[tag], color='blue', label=tag + '_true - train')
            plt.scatter(Y_train.index, self.SCALE * self.predict(X_train, tag), color='black',
                        label=tag + '_pred - train')
            plt.scatter(Y_test.index, self.SCALE * Y_test[tag], color='red', label=tag + '_true - test')
            plt.scatter(Y_test.index, self.SCALE * self.predict(X_test, tag), color='green', label=tag + '_pred - test')
            plt.title(tag)
            plt.ylabel(tag)
            plt.xlabel('time')
            for input_tag in input_cols:
                if input_tag.split('_')[0]==output_tag.split('_')[0]:
                    i += 1
                    plt.subplot(sp_y, sp_x, i)
                    tag=input_tag
                    plt.scatter(Y_train.index, self.SCALE*X_train[tag], color='black', label=tag+'_true - train')
                    plt.scatter(Y_test.index, self.SCALE*X_test[tag], color='blue', label=tag+'_true - test')
                    plt.title(tag)
                    plt.ylabel(tag)
                    plt.xlabel('time')
                    # plt.legend(['Y_true - train','Y_pred - train','Y_true - test', 'Y_pred - test'],pos)
                    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                    #           ncol=2, mode="expand", borderaxespad=0., fontsize=10)

    def visualize_plot_zeros(self,X_train,X_test,Y_train,Y_test,output_cols=[]):

        if len(output_cols)==0:
            output_cols=tags_to_list(self.output_tags)

        sp_y=2
        sp_x=4
        i=1
        plt.figure()
        for output_tag in output_cols:
            if output_tag=='GJOA_QGAS':
                plt.figure()
            else:
                plt.subplot(sp_y,sp_x,i)
                i+=1
            well=output_tag.split('_')[0]
            ind_train=X_train[well+'_CHK']>0.05
            ind_test = X_test[well + '_CHK'] > 0.05
            print(ind_train[ind_train].shape,Y_train.shape,Y_train[output_tag][ind_train].shape)
            plt.scatter(Y_train.index[ind_train], self.SCALE*Y_train[output_tag][ind_train], color='blue', label=output_tag+'_true - train')
            plt.scatter(Y_train.index[ind_train], self.SCALE*self.predict(X_train, output_tag)[ind_train], color='black', label=output_tag+'_pred - train')

            plt.scatter(Y_test.index[ind_test], self.SCALE*Y_test[output_tag][ind_test], color='red', label=output_tag+'_true - test')
            plt.scatter(Y_test.index[ind_test], self.SCALE*self.predict(X_test, output_tag)[ind_test], color='green', label=output_tag+'_pred - test')
            plt.title(output_tag)
            # plt.legend(['Y_true - train','Y_pred - train','Y_true - test', 'Y_pred - test'],pos)
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                       ncol=2, mode="expand", borderaxespad=0., fontsize=10)
        i = 1
        plt.figure()
        for output_tag in output_cols:
            if output_tag == 'GJOA_QGAS':
                plt.figure()
            else:
                plt.subplot(sp_y, sp_x, i)
                i += 1
            well = output_tag.split('_')[0]
            ind_train = X_train[well + '_CHK'] > 0.05
            ind_test = X_test[well + '_CHK'] > 0.05
            plt.scatter(Y_train.index[ind_train], (self.SCALE*Y_train[output_tag]-self.SCALE*self.predict(X_train, output_tag))[ind_train], color='black',
                        label=output_tag + '_train_(true-pred)')

            plt.scatter(Y_test.index[ind_test], (self.SCALE*Y_test[output_tag]-self.SCALE*self.predict(X_test, output_tag))[ind_test], color='green',
                        label=output_tag + '__test_(true-pred)')
            plt.title(output_tag)
            # plt.legend(['Y_true - train','Y_pred - train','Y_true - test', 'Y_pred - test'],pos)
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                       ncol=2, mode="expand", borderaxespad=0., fontsize=10)

    def visualize_scatter_zeros(self,X_train,X_test,Y_train,Y_test,input_cols=[],output_cols=[]):
        if len(input_cols)==0:
            input_cols=tags_to_list(self.input_tags)
        if len(output_cols)==0:
            output_cols=tags_to_list(self.output_tags)

        len_input_cols=len(input_cols)#self.n_inputs
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
        sp_y=4

        for output_tag in output_cols:
            #plt.figure()
            #i=1
            well = output_tag.split('_')[0]
            ind_train = X_train[well + '_CHK'] > 0.05
            ind_test = X_test[well + '_CHK'] > 0.05
            for input_tag in input_cols:
                if input_tag.split('_')[0]==output_tag.split('_')[0] and output_tag!='GJOA_QGAS':
                    plt.subplot(sp_y, sp_x, i)
                    i+=1
                    plt.scatter(X_train[input_tag][ind_train], self.SCALE*Y_train[output_tag][ind_train], color='blue', label=output_tag+'_true - train')
                    plt.scatter(X_train[input_tag][ind_train], self.SCALE*self.predict(X_train, output_tag)[ind_train], color='black', label=output_tag+'_pred - train')

                    plt.scatter(X_test[input_tag][ind_test], self.SCALE*Y_test[output_tag][ind_test], color='red', label=output_tag+'_true - test')
                    plt.scatter(X_test[input_tag][ind_test], self.SCALE*self.predict(X_test, output_tag)[ind_test], color='green', label=output_tag+'_pred - test')

                    plt.xlabel(input_tag)
                    plt.ylabel(output_tag)
                    plt.title(input_tag +' vs '+output_tag)

                    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                               ncol=2, mode="expand", borderaxespad=0., fontsize=15)