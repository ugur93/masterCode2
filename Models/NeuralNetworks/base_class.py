
from .base import *

class NN_BASE:

    def __init__(self):

        #Config
        self.chk_threshold_value=5
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



        self.n_outputs=len(tags_to_list(self.output_tags))
        self.initialize_model()


        self.output_index,self.output_tag_ordered_list,_ = output_tags_to_index(self.output_tags,self.model.get_config()['output_layers'])

        print(self.output_tag_ordered_list)
        print(self.model.get_config()['output_layers'])

        plotModel(self.model,self.model_name)



    def initialize_model(self):
        pass

    def preprocess_data(self,X,Y=[],Y_Q=[]):
        X_dict = df2dict(X,self.input_tags,self.output_tags,'X')
        if self.add_thresholded_output:
            X_dict = addToggledInput(X,X_dict, self.chk_thresholds)
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

        print(X_dict.keys())
        print(Y_dict.keys())

       # exit()

        #self.debug(X_dict,Y_dict,True)


        self.model.fit(X_dict, Y_dict, nb_epoch=self.nb_epoch, batch_size=self.batch_size, verbose=self.verbose,
                       callbacks=self.callbacks,shuffle=False,validation_data=(X_val_dict,Y_val_dict))

    def predict(self, X,tag=False):
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

        predicted_data_reshaped=pd.DataFrame(data=predicted_data_reshaped,columns=self.output_tag_ordered_list)
        if tag==False:
            return predicted_data_reshaped
        else:
            return predicted_data_reshaped[tag]#[:,self.output_index[tag]]


    def get_history(self):
        return self.history.losses

    def get_layer_weights(self,layer_name):
        return self.model.get_layer(layer_name).get_weights()

    def save_model_to_file(self,name,scores,save_weights=True):
        #PATH='/Users/UAC/GITFOLDERS/MasterThesisCode/Models/NeuralNetworks/SavedModels/'
        #PATH='C:/users/ugurac/Documents/GITFOLDERS/MasterThesisCode/Models/NeuralNetworks/model_figures'
        PATH='./Models/NeuralNetworks/model_figures'
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
        #self.SCALE=data.Y_SCALE
        if scaled:
            thresh_transformed=[self.chk_threshold_value/data.get_scale(type='CHK') for i in range(col_length)]#data.transform([[self.chk_threshold_value for i in range(col_length)]],'X')
        else:
            thresh_transformed=[[self.chk_threshold_value for i in range(col_length)]]
        for key,tag_list in self.input_tags.items():
            #print(thresh_transformed)
            tag=find_tag_that_ends_with(tag_list,'CHK')
            #tag=key+'_CHK'
            if tag:
                chk_index=data.X_transformed.columns.get_loc(tag)
                self.chk_thresholds[key]=thresh_transformed[chk_index]




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
    def visualize(self,data,X_train,X_test,Y_train,Y_test,output_cols=[],input_cols=[]):

        self.X_SCALE=100
        #self.plot_scatter_input_output(data,X_train, X_test, Y_train, Y_test, input_cols=input_cols,output_cols=output_cols)
        #self.plot_scatter_chk_well(data,X_train, X_test, Y_train, Y_test, input_cols=input_cols, output_cols=output_cols)
        self.plot_residuals(data,X_train, X_test, Y_train, Y_test, output_cols)
        self.plot_true_and_predicted(data,X_train, X_test, Y_train, Y_test, output_cols)
        #self.plot_true_and_predicted_with_input(data,X_train, X_test, Y_train, Y_test, output_cols=output_cols)
        plt.show()

    def plot_residuals(self,data,X_train,X_test,Y_train,Y_test,output_cols=[],remove_zeros=False):

        if len(output_cols)==0:
            output_cols=self.output_tag_ordered_list
        N_PLOTS = len(output_cols)
        sp_y=int(N_PLOTS/2+0.5)
        if sp_y==0:
            sp_y=1
        sp_x=int(N_PLOTS/sp_y+0.5)
        if sp_x==0:
            sp_x=1
        print(sp_y,sp_x,len(output_cols))

        i = 1
        plt.figure()
        for output_tag in output_cols:
            if output_tag == 'GJOA_QGAS' or output_tag=='GJOA_TOTAL_QOIL_SUM' or output_tag=='GJOA_TOTAL_QOIL':
                plt.figure()
            else:
                plt.subplot(sp_x, sp_y, i)
                i += 1
            #print(self.predict(X_test, output_tag))
            plt.scatter(Y_train.index,
                        data.inverse_transform(Y_train)[output_tag] -data.inverse_transform(self.predict(X_train))[output_tag].values,
                        color='black',
                        label='train')

            plt.scatter(Y_test.index, data.inverse_transform(Y_test)[output_tag] - data.inverse_transform(self.predict(X_test))[output_tag].values,
                        color='green',
                        label='test')
            plt.title(output_tag + '-' + 'Residuals (true-pred)')
            # plt.legend(['Y_true - train','Y_pred - train','Y_true - test', 'Y_pred - test'],pos)
            plt.legend(bbox_to_anchor=(0., 1., 1.01, .0), loc=3,
                       ncol=2, mode="expand", borderaxespad=0.2)
            # plt.legend()
            plt.subplots_adjust(wspace=0.08, hspace=.45, top=0.95, bottom=0.06, left=0.04, right=0.99)
            #plt.tight_layout()

    def plot_true_and_predicted(self,data,X_train,X_test,Y_train,Y_test,output_cols=[]):

        if len(output_cols)==0:
            output_cols=self.output_tag_ordered_list


        N_PLOTS=len(output_cols)
        sp_y=int(N_PLOTS/2+0.5)
        if sp_y==0:
            sp_y=1
        sp_x=int(N_PLOTS/sp_y+0.5)
        if sp_x==0:
            sp_x=1
        print(sp_y,sp_x,len(output_cols))

        i=1
        plt.figure()
        for output_tag in output_cols:
            if output_tag=='GJOA_QGAS' or output_tag=='GJOA_TOTAL_QOIL_SUM' or output_tag=='GJOA_TOTAL_QOIL':
                plt.figure()
            else:
                plt.subplot(sp_x,sp_y,i)
                i+=1
            plt.scatter(Y_train.index, data.inverse_transform(Y_train)[output_tag], color='blue', label='true - train')
            plt.scatter(Y_train.index, data.inverse_transform(self.predict(X_train))[output_tag], color='black', label='pred - train')

            plt.scatter(Y_test.index, data.inverse_transform(Y_test)[output_tag], color='red', label='true - test')
            plt.scatter(Y_test.index, data.inverse_transform(self.predict(X_test))[output_tag], color='green', label='pred - test')
            plt.title(output_tag)
            # plt.legend(['Y_true - train','Y_pred - train','Y_true - test', 'Y_pred - test'],pos)
            plt.legend(bbox_to_anchor=(0., 1., 1.01, .0), loc=3,
                       ncol=2, mode="expand", borderaxespad=0.2)
            # plt.legend()
            plt.subplots_adjust(wspace=0.08, hspace=.2, top=0.94, bottom=0.06, left=0.04, right=0.99)
            #plt.tight_layout()

    def plot_scatter_input_output(self,data,X_train,X_test,Y_train,Y_test,input_cols=[],output_cols=[]):
        if len(input_cols)==0:
            input_cols=tags_to_list(self.input_tags)
        if len(output_cols)==0:
            output_cols=self.output_tag_ordered_list

        i=1
        N_plots=count_n_well_inputs(input_cols)
        sp_y=int(N_plots/2+0.5)
        if sp_y==0:
            sp_y=1
        sp_x=int(N_plots/sp_y+0.5)
        if sp_x==0:
            sp_x=1
        print(sp_y, sp_x)
        for output_tag in output_cols:
            plt.figure()
            i=1
            for input_tag in input_cols:
                if input_tag.split('_')[0]==output_tag.split('_')[0] and output_tag!='GJOA_QGAS'  and output_tag!='GJOA_TOTAL_QOIL_SUM' and output_tag!='GJOA_TOTAL_QOIL':
                    plt.subplot(sp_y, sp_x, i)
                    i+=1
                    plt.scatter(data.inverse_transform(X_train)[input_tag],
                                data.inverse_transform(Y_train)[output_tag], color='blue', label=output_tag+'_true - train')
                    plt.scatter(data.inverse_transform(X_train)[input_tag],
                                data.inverse_transform(self.predict(X_train))[output_tag], color='black', label=output_tag+'_pred - train')

                    plt.scatter(data.inverse_transform(X_test)[input_tag],
                                data.inverse_transform(Y_test)[output_tag], color='red', label=output_tag+'_true - test')
                    plt.scatter(data.inverse_transform(X_test)[input_tag],
                                data.inverse_transform(self.predict(X_test))[output_tag], color='green', label=output_tag+'_pred - test')

                    plt.xlabel(input_tag)
                    plt.ylabel(output_tag)
                    plt.title(input_tag +' vs '+output_tag)

                    plt.legend(bbox_to_anchor=(0., 1., 1.01, .0), loc=3,
                               ncol=2, mode="expand", borderaxespad=0.2)
                    # plt.legend()
                    plt.subplots_adjust(wspace=0.08, hspace=.45, top=0.95, bottom=0.06, left=0.04, right=0.99)
                    #plt.tight_layout()
    def plot_scatter_chk_well(self,data,X_train,X_test,Y_train,Y_test,input_cols=[],output_cols=[]):
        if len(input_cols)==0:
            input_cols=tags_to_list(self.input_tags)
        if len(output_cols)==0:
            output_cols=self.output_tag_ordered_list

        i=1
        N_plots=len(output_cols)#count_n_well_inputs(input_cols)
        sp_y=int(N_plots/2+0.5)
        if sp_y==0:
            sp_y=1
        sp_x=int(N_plots/sp_y+0.5)
        if sp_x==0:
            sp_x=1
        print(sp_y, sp_x,N_plots)
        plt.figure()
        i = 1
        for output_tag in output_cols:
            for input_tag in input_cols:
                if input_tag.split('_')[0]==output_tag.split('_')[0] and output_tag!='GJOA_QGAS' and input_tag.split('_')[1]=='CHK':
                    plt.subplot(sp_x, sp_y, i)
                    i+=1
                    plt.scatter(data.inverse_transform(X_train)[input_tag], data.inverse_transform(Y_train)[output_tag], color='blue', label='true - train')
                    plt.scatter(data.inverse_transform(X_train)[input_tag], data.inverse_transform(self.predict(X_train))[output_tag], color='black', label='pred - train')

                    plt.scatter(data.inverse_transform(X_test)[input_tag], data.inverse_transform(Y_test)[output_tag], color='red', label='true - test')
                    plt.scatter(data.inverse_transform(X_test)[input_tag], data.inverse_transform(self.predict(X_test))[output_tag], color='green', label='pred - test')

                    plt.xlabel(input_tag)
                    plt.ylabel(output_tag)
                    plt.title(input_tag +' vs '+output_tag)

                    plt.legend(bbox_to_anchor=(0., 1., 1.01, .0), loc=3,
                               ncol=2, mode="expand", borderaxespad=0.2)
                    #plt.legend()
                    plt.subplots_adjust(wspace=0.08,hspace=.45,top=0.95,bottom=0.06,left=0.04,right=0.99)
                    #plt.tight_layout()


    def plot_true_and_predicted_with_input(self,data,X_train,X_test,Y_train,Y_test,output_cols=[]):

        if len(output_cols)==0:
            output_cols=tags_to_list(self.output_tags)
        input_cols=tags_to_list(self.input_tags)

        sp_y=count_n_well_inputs(input_cols)+1
        sp_x=1
        print(sp_y,sp_x,len(input_cols))

        i=1

        for output_tag in output_cols:
            #plt.figure()
            i=1
            tag = output_tag
            #ax.subplot(sp_y, sp_x, i)
            _, axes = plt.subplots(sp_y, 1, sharex=True)
            ax=axes[0]
            ax.scatter(Y_train.index, self.SCALE * Y_train[tag], color='blue', label=tag + '_true - train')
            ax.scatter(Y_train.index, self.SCALE * self.predict(X_train, tag), color='black',
                        label=tag + '_pred - train')
            ax.scatter(Y_test.index, self.SCALE * Y_test[tag], color='red', label=tag + '_true - test')
            ax.scatter(Y_test.index, self.SCALE * self.predict(X_test, tag), color='green', label=tag + '_pred - test')
            ax.set_title(tag)
            ax.set_ylabel(tag)
            ax.set_xlabel('time')
            i=1
            for input_tag in input_cols:
                if input_tag.split('_')[0]==output_tag.split('_')[0]:
                    #plt.subplot(sp_y, sp_x, i)
                    tag=input_tag
                    ax=axes[i]
                    i+=1
                    ax.scatter(Y_train.index, self.SCALE*X_train[tag], color='black', label=tag+'_true - train')
                    ax.scatter(Y_test.index, self.SCALE*X_test[tag], color='blue', label=tag+'_true - test')
                    ax.set_title(tag)
                    ax.set_ylabel(tag)
                    ax.set_xlabel('time')
                    # plt.legend(['Y_true - train','Y_pred - train','Y_true - test', 'Y_pred - test'],pos)
                    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                    #           ncol=2, mode="expand", borderaxespad=0., fontsize=10)

    def plot_true_and_predicted_zeros(self,X_train,X_test,Y_train,Y_test,output_cols=[]):

        if len(output_cols)==0:
            output_cols=self.output_tag_ordered_list


        N_PLOTS=len(output_cols)
        sp_y=int(N_PLOTS/2+0.5)
        if sp_y==0:
            sp_y=1
        sp_x=int(N_PLOTS/sp_y+0.5)
        if sp_x==0:
            sp_x=1
        print(sp_y,sp_x,len(output_cols))

        i=1
        plt.figure()
        for output_tag in output_cols:
            if output_tag=='GJOA_QGAS' or output_tag=='GJOA_TOTAL_QOIL_SUM' or output_tag=='GJOA_TOTAL_QOIL':
                plt.figure()
            else:
                plt.subplot(sp_x,sp_y,i)
                i+=1
            well = output_tag.split('_')[0]
            ind_train = X_train[well + '_CHK'] > 0.05
            ind_test = X_test[well + '_CHK'] > 0.05
            plt.scatter(Y_train.index[ind_train], self.SCALE*Y_train[output_tag][ind_train], color='blue', label='true - train')
            plt.scatter(Y_train.index[ind_train], self.SCALE*self.predict(X_train[ind_train], output_tag), color='black', label='pred - train')

            plt.scatter(Y_test.index[ind_test], self.SCALE*Y_test[output_tag][ind_test], color='red', label='true - val')
            plt.scatter(Y_test.index[ind_test], self.SCALE*self.predict(X_test[ind_test], output_tag), color='green', label='pred - val')
            plt.title(output_tag)
            # plt.legend(['Y_true - train','Y_pred - train','Y_true - test', 'Y_pred - test'],pos)
            plt.legend(bbox_to_anchor=(0., 1.03, 1., .112), loc=3,
                       ncol=2, mode="expand", borderaxespad=0., fontsize=10)
    def plot_residuals_zeros(self,X_train,X_test,Y_train,Y_test,output_cols=[],remove_zeros=False):

        if len(output_cols)==0:
            output_cols=self.output_tag_ordered_list

        sp_y=int((len(output_cols)-1)/2+0.5)
        if sp_y==0:
            sp_y=1
        sp_x=int((len(output_cols))/sp_y+0.5)
        if sp_x==0:
            sp_x=1
        print(sp_y,sp_x,len(output_cols))

        i = 1
        plt.figure()
        for output_tag in output_cols:
            if output_tag == 'GJOA_QGAS' or output_tag=='GJOA_TOTAL_QOIL_SUM' or output_tag=='GJOA_TOTAL_QOIL':
                plt.figure()
            else:
                plt.subplot(sp_x, sp_y, i)
                i += 1
            well = output_tag.split('_')[0]
            ind_train = X_train[well + '_CHK'] > 0.05
            ind_test = X_test[well + '_CHK'] > 0.05
            plt.scatter(Y_train.index[ind_train],
                        self.SCALE *( Y_train[output_tag] -self.predict(X_train, output_tag))[ind_train],
                        color='black',
                        label='train')

            plt.scatter(Y_test.index[ind_test], self.SCALE *( Y_test[output_tag] -  self.predict(X_test, output_tag))[ind_test],
                        color='green',
                        label='val')
            plt.title(output_tag + '-' + 'Residuals (true-pred)')
            # plt.legend(['Y_true - train','Y_pred - train','Y_true - test', 'Y_pred - test'],pos)
            plt.legend(bbox_to_anchor=(0., 1.04, 1., .102), loc=9,
                      ncol=2, mode="expand", borderaxespad=0., fontsize=10)

    def plot_scatter_input_output_zeros(self, X_train, X_test, Y_train, Y_test, input_cols=[], output_cols=[]):
        if len(input_cols) == 0:
            input_cols = tags_to_list(self.input_tags)
        if len(output_cols) == 0:
            output_cols = self.output_tag_ordered_list

        i = 1
        N_plots = count_n_well_inputs(input_cols)
        sp_y = int(N_plots / 2 + 0.5)
        if sp_y == 0:
            sp_y = 1
        sp_x = int(N_plots / sp_y + 0.5)
        if sp_x == 0:
            sp_x = 1
        print(sp_y, sp_x)
        for output_tag in output_cols:
            plt.figure()
            i = 1
            for input_tag in input_cols:
                if input_tag.split('_')[0] == output_tag.split('_')[
                    0] and output_tag != 'GJOA_QGAS' and output_tag != 'GJOA_TOTAL_QOIL_SUM' and output_tag != 'GJOA_TOTAL_QOIL':
                    plt.subplot(sp_y, sp_x, i)
                    well = output_tag.split('_')[0]
                    ind_train = X_train[well + '_CHK'] > 0.05
                    ind_test = X_test[well + '_CHK'] > 0.05
                    i += 1
                    plt.scatter(X_train[input_tag][ind_train], self.SCALE * Y_train[output_tag][ind_train], color='blue',
                                label=output_tag + '_true - train')
                    plt.scatter(X_train[input_tag][ind_train], self.SCALE * self.predict(X_train, output_tag)[ind_train],
                                color='black', label=output_tag + '_pred - train')

                    plt.scatter(X_test[input_tag][ind_test], self.SCALE * Y_test[output_tag][ind_test], color='red',
                                label=output_tag + '_true - test')
                    plt.scatter(X_test[input_tag][ind_test], self.SCALE * self.predict(X_test, output_tag)[ind_test], color='green',
                                label=output_tag + '_pred - test')

                    plt.xlabel(input_tag)
                    plt.ylabel(output_tag)
                    plt.title(input_tag + ' vs ' + output_tag)

                    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                               ncol=2, mode="expand", borderaxespad=0., fontsize=10)
