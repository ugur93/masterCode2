
import Models.NeuralNetworks.NET1 as NN1
import Models.NeuralNetworks.NN_external as NNE
from Models.NeuralNetworks import NET2_PRESSURE,NN_from_file,NET3,NCNET_CHKPRES,NET_MISC,NCNET1_GJOA2,NCNET_VANILLA_GJOA2,CNN_test,test_model,NCNET4_combined
import time


from .base import *
from .Visualize import *

#params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
    #          'learning_rate': 0.01, 'loss': 'ls'}
    #model=ensemble.GradientBoostingRegressor(**params)
    #model = SVR(C=1000, gamma=0.001,epsilon=0.0001)
MODEL_SAVEFILE_NAME='SSNET2_PRETRAINING_2'




DATA_TYPE='GASs'
def validate(DataOIL,DataGAS):

    if DATA_TYPE=='GAS':
        Data=DataGAS
    else:
        Data=DataOIL
    #bagging_test(Data)
    validate_train_test_split(Data)
    #ensemble_learning_bagging(Data)
    #grid_search3(Data)
    #validateRepeat(Data)
    #validateCV(Data)

def validate_train_test_split(Data):
    X = Data.X_transformed
    Y = Data.Y_transformed


    #subsample(X,Y)

    #X_old,Y_old,X_new,Y_new=split_data(X,Y,split_size=0.3)

    #print(X_old.index)
    #print(X_new.index)
    #X=X[200:-1]
    #Y=Y[200:-1]


    #X_train, Y_train, X_val, Y_val, X_test, Y_test=get_train_test_val_data(X,Y,test_size=0.0,val_size=0.05)

    #print(len(X_train))
    #X_new=X_test
    #Y_new=Y_test
    #X_val=X_train
    #Y_val=Y_train

    #print(X_train.index)
    #print(X_new.index)
    #X_train=X_old
    #Y_train=Y_old
    #X_val=X_train
    #Y_val=Y_train
    PATH='Models/NeuralNetworks/SavedModels2/Weights/GJOA_OIL_WELLS_mae_D2_W20_L20.001_DPR0.h5'
    #PATH = 'Models/NeuralNetworks/SavedModels2/hdf5_files/NCNET_GAS_PRETRAINED_WITH_OLD_DATA'3
    #GJOA QGAS
    PATHS=[]
    for i in range(15):
        if i not in [16]:
            PATHS.append('Models/NeuralNetworks/SavedModels2/weights/ENSEMBLE_LEARNING_GAS_'+str(i)+'.h5')
    print(PATHS)
    #exit()
    #pressure_weights=
    if DATA_TYPE=='GAS':
        PATH = 'Models/NeuralNetworks/SavedModels2/Weights/'
        X_train, Y_train, X_val, Y_val, X_test, Y_test = get_train_test_val_data(X, Y, test_size=0.1, val_size=0.2)
        #model=NCNET_CHKPRES.PRESSURE_PDC(Data)
        #model=NCNET_CHKPRES.PRESSURE_PWH(Data)
        model = NET2_PRESSURE.SSNET2()

        #
        #model.load_weights_from_file(PATH)
        #model = NNE.SSNET_EXTERNAL(MODEL_SAVEFILE_NAME)
        #model = NN1.SSNET1()
        #model=NN_from_file.NCNET_FILE(PATH)

        #model.initialize_zero_thresholds(Data)
        model.initialize_chk_thresholds(Data, True)
        # model.initialize_zero_thresholds(Data)
        start = time.time()
        print(model.get_config())
        #model.model.load_weights(PATH + 'GJOA_GAS_WELLS_QGAS_FINAL.h5', by_name=True)
        # print(model.model.get_config())
        # model.fit(X_train,Y_train,X_val,Y_val)

        # Fit with old data
        #model.update_model()
        model.fit(X_train, Y_train, X_val, Y_val)
        #X_train, Y_train, X_val, Y_val, X_test, Y_test = get_train_test_val_data(X, Y, test_size=0.0, val_size=0.2)
    else:
        #GJOA_QOIL
        #pass
        PATH = 'Models/NeuralNetworks/SavedModels2/Weights/'
        X_train, Y_train, X_val, Y_val, X_test, Y_test = get_train_test_val_data(X, Y, test_size=0.1, val_size=0.2)
        model=NCNET1_GJOA2.NCNET1_GJOA2()
        #model=NCNET1_GJOA2.ENSEMBLE(PATHS)
        #model.model.load_weights(PATH+'GJOA_OIL_WELLS_GAS_MODEL22.h5')
        #model = NCNET1_GJOA2.ENSEMBLE(PATHS)
        #model=NCNET_VANILLA_GJOA2.NCNET_VANILLA()
        #model=CNN_test.CNN_GJOAOIL()
        #model = NCNET_CHKPRES.PRESSURE_PBH()
        #
        #model = NCNET_CHKPRES.PRESSURE(tag='PWH')
        #model.model.load_weights(PATH,by_name=True)
        #model = test_model.Test_model()
        #model=NCNET4_combined.NET4_W_PRESSURE2(PATH)

        #model=NCNET4_combined.NET4_W_PRESSURE3()


        #model.model.load_weights(PATH + 'GJOA_OIL_WELLS_PBH.h5', by_name=True)
        #model.model.load_weights(PATH + 'GJOA_OIL_WELLS_PDC.h5', by_name=True)
        #model.model.load_weights(PATH+'GJOA_OIL_WELLS_PWH.h5',by_name=True)
        #model.model.load_weights(PATH + 'GJOA_OIL_WELLS_GAS_HUBER_MODEL_FINAL.h5', by_name=True)


        # model.initialize_zero_thresholds(Data)
        model.initialize_chk_thresholds(Data, True)
        # model.initialize_zero_thresholds(Data)
        start = time.time()
        print(model.get_config())
        # print(model.model.get_config())
        #model.fit(X_train,Y_train,X_val,Y_val)

        # Fit with old data
        #model.update_model()
        model.fit(X_train[1:-1], Y_train[1:-1], X_val, Y_val)
    end = time.time()
    print('Fitted with time: {}'.format(end - start))
    scores, scores_latex = evaluate_model2(model, Data, X_train[1:-1], X_val, Y_train[1:-1], Y_val)
    print(scores)
    model.save_model_to_file(model.model_name)
   # exit()
    #model.save_model_config(scores)

    #get_choke_diff_deviation(model, Data, X_train, Y_train)
    if False:
        #import seaborn
        tag = 'PWH'
        name='C1'
        Y_p=model.predict(X).set_index(X.index)
        #Y_p=Y_p.add(X_val['C1_shifted_PDC'],axis=0)
        print(X_val[name+'_shifted_CHK'].head(10))
        print(X_val[name+'_CHK'].head(10))
        print(X_val[name+'_delta_CHK'].head(10))
        print(Y_p.head(10))
        fig,axes=plt.subplots(3,1,sharex=True)
        axes=axes.flatten()
        #+Data.inverse_transform(X, 'X')['C1_shifted_PWH']
        axes[0].grid()
        axes[1].grid()
        axes[2].grid()

        axes[0].plot(Data.inverse_transform(Y_p,'Y')[name+'_'+tag],'*',color='red',label=name+'_'+tag+'_predicted')
        axes[0].plot(Data.inverse_transform(Y,'Y')[name+'_'+tag],'.',color='blue',label=name+'_'+tag)
        axes[0].legend()
        axes[1].plot(Data.inverse_transform(X,'X')[name+'_CHK'],label=name+'_CHK')
        axes[1].plot(Data.inverse_transform(X,'X')[name+'_shifted_CHK']*-1,color='red',label=name+'_shifted_CHK')

        for key in ['C1', 'C2', 'C3', 'C4', 'B3','B1','D1']:
            axes[1].plot(Data.inverse_transform(X,'X')[key+'_delta_CHK'],label=key+'_delta_CHK')
        axes[1].plot(Data.inverse_transform(X, 'X')['GJOA_RISER_delta_CHK'], label='GJOA_RISER_delta_CHK')
        axes[1].legend()

        axes[2].plot(Data.inverse_transform(X,'X')[name+'_shifted_'+tag],color='red',label=name+'_prev_'+tag)
        axes[2].plot(Data.inverse_transform(X,'X')[name+'_'+tag], color='blue',label=name+'_now_'+tag)
        axes[2].plot(Data.inverse_transform(Y,'Y')[name+'_delta_'+tag], color='green', label=name+'_delta_'+tag)
        plt.legend()
        if False:
            fig,axes=plt.subplots(1,1)
            axes.scatter(Data.inverse_transform(X,'X')[name+'_delta_'+'CHK'],Data.inverse_transform(Y_p,'Y')[name+'_delta_'+tag],color='red',label='True')
            axes.scatter(Data.inverse_transform(X, 'X')[name + '_delta_' + 'CHK'],
                         Data.inverse_transform(Y, 'Y')[name + '_delta_' + tag],label='predicted',color='blue')
            axes.set_xlabel(name+'_delta_'+'CHK')
            axes.set_ylabel(name+'_delta_'+tag)

    else:
        visualize(model, Data, X_train, X_val, Y_train, Y_val, output_cols=[], input_cols=[])

    plt.show()



def validateRepeat(Data):
    X, Y, X_train, Y_train, X_val, Y_val, X_test, Y_test = get_train_test_val_data(Data, test_size=0.0, val_size=0.1)

    N_REPEAT=10


    mse_train_list=[]
    mse_test_list=[]
    r2_train_list=[]
    r2_test_list=[]


    for i in range(N_REPEAT):
        print('Training status: \n N_REPEAT= {}\n STATUS: {}\n '.format(N_REPEAT,i))
        #model = NCNET_CHKPRES.SSNET3_PRESSURE()
        #model = NN1.SSNET1()
        model = NCNET1_GJOA2.NCNET1_GJOA2()
        #model = NCNET_VANILLA_GJOA2.NCNET_VANILLA()
        #model=NET2_PRESSURE.SSNET2()
        model.initialize_chk_thresholds(Data, True)
        conf=model.get_config()
        model_name=model.model_name
        model.fit(X_train, Y_train, X_val, Y_val)
        model.update_model()
        model.fit(X_train, Y_train, X_val, Y_val)
        score_train_MSE, score_test_MSE, score_train_r2, score_test_r2,cols = model.evaluate(Data,X_train, X_val, Y_train,Y_val)
        del model
        #ind=cols.index()
        #print(cols)
        mse_train_list.append(score_train_MSE)
        mse_test_list.append(score_test_MSE)
        r2_train_list.append(score_train_r2)
        r2_test_list.append(score_test_r2)

    MSE_TRAIN=np.mean(mse_train_list,axis=0)
    MSE_TEST=np.mean(mse_test_list,axis=0)
    R2_TEST=np.mean(r2_test_list,axis=0)
    R2_TRAIN=np.mean(r2_train_list,axis=0)

    s=print_scores(Data, Y_train, Y_val, MSE_TRAIN, MSE_TEST, R2_TRAIN, R2_TEST, cols)


    save_to_file(model_name + '_NREPEAT' + str(N_REPEAT)+'_results_QOIL_CHK_7_THIS',conf+s+'\n NREPEAT: '+str(N_REPEAT))



def validateCV(Data,params=None,save=True):
    X = Data.X_transformed
    Y = Data.Y_transformed


    X, Y, X_val, Y_val, X_test, Y_test = get_train_test_val_data(X, Y, test_size=0.1, val_size=0)

    X.drop([0], inplace=True)
    Y.drop([0], inplace=True)


    kfold=model_selection.KFold(n_splits=5,random_state=False)

    scores_rmse_train=None
    scores_r2_test = None
    scores_rmse_test = None
    scores_r2_train = None



    #zprint scores
    #print(chkInputs.index)
    i=1

    filename='GJOA_OIL_WELLS_OIL_1'


    for train_index,test_index in kfold.split(X.index):

        #model=NCNET1_GJOA2.NCNET1_GJOA2()
        model = NCNET_CHKPRES.PRESSURE(**params, tag='PWH')
        model.initialize_chk_thresholds(Data, True)

        X_train=X.iloc[train_index]
        X_val=X.iloc[test_index]
        Y_train=Y.iloc[train_index]
        Y_val=Y.iloc[test_index]
        #model.fit(X_train,Y_train,X_val,Y_val)

        # Fit with old data
        #model.update_model()
        model.fit(X_train, Y_train,X_val,Y_val)
        scores= evaluate_model(model,Data, X_train, X_val, Y_train, Y_val)
        print(scores)

        if scores_rmse_train is None:
            scores_rmse_train=scores['RMSE_train'].to_frame().T
            scores_rmse_test = scores['RMSE_test'].to_frame().T
            scores_r2_train = scores['R2_train'].to_frame().T
            scores_r2_test = scores['R2_test'].to_frame().T
        else:
            scores_rmse_train=scores_rmse_train.append(scores['RMSE_train'].to_frame().T)
            scores_rmse_test=scores_rmse_test.append(scores['RMSE_test'].to_frame().T)
            scores_r2_train=scores_r2_train.append(scores['R2_train'].to_frame().T)
            scores_r2_test=scores_r2_test.append(scores['R2_test'].to_frame().T)

        print(scores_rmse_test)
        del model
        i+=1

    scores_rmse_train.set_index(pd.Index(range(0,len(scores_rmse_train))))
    scores_rmse_test.set_index(pd.Index(range(0, len(scores_rmse_test))))
    scores_r2_train.set_index(pd.Index(range(0, len(scores_r2_train))))
    scores_r2_test.set_index(pd.Index(range(0, len(scores_r2_test))))

    RMSE_TRAIN = np.mean(scores_rmse_train)
    RMSE_TEST = np.mean(scores_rmse_test)
    R2_TRAIN = np.mean(scores_r2_train)
    R2_TEST = np.mean(scores_r2_test)



    print(RMSE_TRAIN)
    print(RMSE_TEST)
    print(scores_rmse_train)
    print(scores_rmse_test)

    if save:
        s='RMSE_TRAIN: \n{}\n'.format(RMSE_TRAIN)
        s += 'RMSE_TEST: \n{}\n'.format(RMSE_TEST)
        s += 'RMSE_TRAIN_SCORES: \n{}\n'.format(scores_rmse_train)
        s += 'RMSE_TEST_SCORES: \n{}\n'.format(scores_rmse_test)

        PATH = 'Models/NeuralNetworks/CV_results/' + filename
        f = open(PATH, 'w')
        f.write(s)
        f.close()

    return {'RMSE_train':RMSE_TRAIN,'RMSE_test':RMSE_TEST,'R2_train':R2_TRAIN,'R2_test':R2_TEST}


def validateCV2(Data,cv=10):
    X = Data.X_transformed
    Y = Data.Y_transformed



    kfold=model_selection.KFold(n_splits=10,random_state=False)

    scores_rmse_train=np.empty(shape=(0,))
    scores_r2_test = np.empty(shape=(0,))
    scores_rmse_test = np.empty(shape=(0,))
    scores_r2_train = np.empty(shape=(0,))

    rmse_train_list = None
    rmse_test_list = None

    #zprint scores
    #print(chkInputs.index)
    i=1




    for train_index,test_index in kfold.split(X.index):
        #print("TRAIN:", train_index, "TEST:", test_index)
        #print(chkInputs[test_index])

        #print('Train index: {}'.format(train_index))
        #print('Val index: {}'.format(test_index))
        #model = NN1.SSNET1()
        #model = NCNET_CHKPRES.SSNET3_PRESSURE()
        model=NCNET1_GJOA2.NCNET1_GJOA2()
        model.initialize_chk_thresholds(Data, True)
        X_train=X.iloc[train_index]
        X_val=X.iloc[test_index]
        Y_train=Y.iloc[train_index]
        Y_val=Y.iloc[test_index]
        model.fit(X_train, Y_train,X_val,Y_val)
        score_train_MSE, score_test_MSE, score_train_r2, score_test_r2, cols = model.evaluate(Data, X_train, X_val, Y_train, Y_val)
        #score_train_MSE=pd.DataFrame(data=score_train_MSE,columns=cols)
        #score_test_MSE = pd.DataFrame(data=score_test_MSE, columns=cols)

        if rmse_train_list is None:
            rmse_train_list=pd.DataFrame(data=[],columns=cols,index=[0])
            rmse_test_list = pd.DataFrame(data=[], columns=cols, index=[0])
        rmse_train_list.loc[-1]=np.sqrt(score_train_MSE)
        rmse_train_list.index+=1
        rmse_test_list.loc[-1]=np.sqrt(score_test_MSE)
        rmse_test_list.index+=1
        print(rmse_train_list)


        del model
        i+=1
    RMSE_TRAIN = np.mean(rmse_train_list)
    RMSE_TEST = np.mean(rmse_test_list)


    print(RMSE_TRAIN)

    print(RMSE_TEST)





def plotTrainingHistory(model):
    history = model.get_history()
    plt.figure()
    plt.plot(history)
    plt.title('Loss history')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.show()



def generate_grid(search_params):
    from itertools import product
    items=sorted(search_params.items())
    keys, values=zip(*items)
    params=[]
    for v in product(*values):
        params.append(dict(zip(keys,v)))
    return params
def grid_search3(Data):
    X = Data.X_transformed
    Y = Data.Y_transformed

    cum_thresh=15

    X_train, Y_train, X_val, Y_val, X_test, Y_test = get_train_test_val_data(X, Y, test_size=0.1, val_size=0.2)

    search_params={'n_depth':[2],'n_width':[50,60,70,80,90,100],
                   'l2w':np.arange(0.00005,0.0005,0.00001),'seed':[3014]}

    #search_params = {'n_depth': [2], 'n_width': [50],
    #                 'l2w':[0.0001], 'seed': [3014]}

    grid_params=generate_grid(search_params)

    len_grid=len(grid_params)

    print('Size of search space: {}'.format(len_grid))


    best_results=None
    best_cost=1e100
    best_params={}
    #filename='GRID_SEARCH_OIL_WELLS_OIL2'
    filename = 'GRID_SEARCH_OIL_WELLS_PWH_2'
    ii=1
    pd.options.display.float_format = '{:.2f}'.format
    #col_eval=['GJOA_OIL_SUM_QGAS']
    col_eval=['GJOA_TOTAL_SUM_QOIL']
    for params in grid_params:

        print('Training with params: {}, filename: {} '.format(params, filename))
        print('\n\n\n')
        print('On n_grid: {} of {}'.format(ii, len_grid))
        ii += 1
        params['seed']=int(params['seed'])
        scores=validateCV(Data,params,save=False)

        #current_cost =scores['RMSE_test'][col_eval].values
        current_cost = np.sum(scores['RMSE_test'])
        #print(np.sqrt(scores['MSE_test']))
        #print(current_cost)
        if current_cost<best_cost:
            best_cost=current_cost
            best_params=params
            best_results=scores


        print('THIS COST: {}, BEST COST: {}'.format(current_cost, best_cost))
        print('Best params:{} \n'.format(best_params))
        print('BEST SCORES: ')
        print(best_results)



    s='Best results: \n'
    s+='Best params:{} \n'.format(best_params)
    s+='Best COST: {} \n'.format(best_cost)
    s+='BEST SCORES: \n {}'.format(best_results)

    PATH = 'Models/NeuralNetworks/'+filename
    f = open(PATH, 'w')
    f.write(s)
    f.close()
def grid_search2(Data):
    X = Data.X_transformed
    Y = Data.Y_transformed

    cum_thresh=15

    X_train, Y_train, X_val, Y_val, X_test, Y_test = get_train_test_val_data(X, Y, test_size=0.1, val_size=0.2)

    search_params={'n_depth':[2],'n_width':[30,40,50,60,80,90,100],
                   'l2w':np.arange(0.00005,0.001,0.0001),'seed':[3014]}

    #search_params = {'n_depth': [2], 'n_width': [50],
    #                 'l2w':[0.0001], 'seed': [3014]}

    grid_params=generate_grid(search_params)

    len_grid=len(grid_params)

    print('Size of search space: {}'.format(len_grid))


    best_results=None
    best_cost=1e100
    best_params={}
    #filename='GRID_SEARCH_OIL_WELLS_OIL2'
    filename = 'GRID_SEARCH_OIL_WELLS_PBH'
    ii=1
    pd.options.display.float_format = '{:.2f}'.format
    col_eval=['GJOA_TOTAL_SUM_QOIL']
    for params in grid_params:

        print('Training with params: {}, filename: {} '.format(params, filename))
        params['seed']=int(params['seed'])
        #model = NCNET1_GJOA2.NCNET1_GJOA2(**params)
        model=NCNET_CHKPRES.PRESSURE(**params,tag='PBH')
        #model = NCNET_CHKPRES.PRESSURE_PDC(**params)
        print('\n\n\n')
        print('On n_grid: {} of {}'.format(ii,len_grid))
        ii+=1
        model.initialize_chk_thresholds(Data, True)
        model.fit(X_train, Y_train, X_val, Y_val)
        scores= evaluate_model(model,Data, X_train, X_val, Y_train, Y_val)
        #print(scores)
        #current_cost =scores['RMSE_test'][col_eval].values
        current_cost = np.sum(scores['RMSE_test'])
        #print(np.sqrt(scores['MSE_test']))
        #print(current_cost)
        if current_cost<best_cost:
            best_cost=current_cost
            best_params=params
            best_results=scores
        del model

        print('THIS COST: {}, BEST COST: {}'.format(current_cost, best_cost))
        print('Best params:{} \n'.format(best_params))
        print('BEST SCORES: ')
        print(best_results)



    s='Best results: \n'
    s+='Best params:{} \n'.format(best_params)
    s+='Best COST: {} \n'.format(best_cost)
    s+='BEST SCORES: \n {}'.format(best_results)

    PATH = 'Models/NeuralNetworks/'+filename
    f = open(PATH, 'w')
    f.write(s)
    f.close()

def ensemble_learning(Data):
    X = Data.X_transformed
    Y = Data.Y_transformed

    cum_thresh = 15

    X_train, Y_train, X_val, Y_val, X_test, Y_test = get_train_test_val_data(X, Y, test_size=0.1, val_size=0.2)

    search_params = {'n_depth': [2], 'n_width': [100],
                     'l2w': [0.0003], 'seed': np.random.randint(100,10000,15)}
    grid_params = generate_grid(search_params)

    len_grid = len(grid_params)

    print('Size of search space: {}'.format(len_grid))


    name='ENSEMBLE_LEARNING_3_GAS_'
    PATH='Models/NeuralNetworks/SavedModels2/hdf5_files/'
    PATHS=[]
    i=0
    for params in grid_params:
        params['seed'] = int(params['seed'])
        model = NCNET1_GJOA2.NCNET1_GJOA2(**params)
        model.model_name=name+str(i)
        PATHS.append(PATH+model.model_name+'.h5')
        i+=1

        print('Training with params: {}'.format(params))
        model.initialize_chk_thresholds(Data, True)
        #model.fit(X_train, Y_train, X_val, Y_val)
        #model.update_model()
        model.fit(X_train, Y_train, X_val, Y_val)

        scores, scores_latex = evaluate_model(model, Data, X_train, X_val, Y_train, Y_val)
        model.save_model_to_file(model.model_name, scores)
        print(scores)


        del model

    print('[')
    for path in PATHS:
        print(path + ',')
    print(']')




def bagging_test(Data):
    X = Data.X_transformed
    Y = Data.Y_transformed

    cum_thresh = 15

    X_train, Y_train, X_val, Y_val, X_test, Y_test = get_train_test_val_data(X, Y, test_size=0.1, val_size=0.2)
    search_params = {'n_depth': [4], 'n_width': [50],
                     'l2w': [0.000001], 'seed': np.random.randint(1, 10000, 10)}

    params={'n_depth': 2, 'n_width': 50,
                     'l2w': 0.0001, 'seed': 9035}
    model = NCNET1_GJOA2.NCNET1_GJOA2(**params)

    #tags=model.output_tag_ordered_list

    mod=ensemble.BaggingRegressor(base_estimator=model,n_estimators=2)
    print(X_train.shape)
    mod.fit(X_train,Y_train['GJOA_OIL_QGAS'])

    print(mod.predict(X_val))

def ensemble_learning_bagging(Data):
    X = Data.X_transformed
    Y = Data.Y_transformed
    GS_SIZE=15

    print('Grid Search size: {}'.format(GS_SIZE))
    params = {'n_depth': 2, 'n_width': 100,
              'l2w': 0.0002, 'seed': 3014}

    name='ENSEMBLE_LEARNING_GAS_'
    PATH='Models/NeuralNetworks/SavedModels2/hdf5_files/'
    PATHS=[]
    i=1
    for i in range(GS_SIZE):
        if i>=12:
            print('Training with params: {}'.format(params))

            model = NCNET1_GJOA2.NCNET1_GJOA2(**params)

            model.model_name=name+str(i)

            PATHS.append(PATH+model.model_name+'.h5')

            X_train, Y_train, X_val, Y_val, X_test, Y_test = get_train_test_val_data(X, Y, test_size=0.1, val_size=0.2)
            X_train,Y_train=subsample(X_train,Y_train)


            model.initialize_chk_thresholds(Data, True)
            model.fit(X_train, Y_train, X_val, Y_val)

            scores, scores_latex = evaluate_model2(model, Data, X_train, X_val, Y_train, Y_val)
            model.save_model_to_file(model.model_name)
            print(scores)
            del model
        else:
            X_train, Y_train, X_val, Y_val, X_test, Y_test = get_train_test_val_data(X, Y, test_size=0.1, val_size=0.2)
            X_train, Y_train = subsample(X_train, Y_train)



    print('[')
    for path in PATHS:
        print(path+',')
    print(']')