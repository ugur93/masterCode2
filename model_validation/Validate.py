
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

    validate_train_test_split(Data)
    #ensemble_learning(Data)
    #grid_search2(Data)
    #validateRepeat(Data)
    #validateCV(Data)

def validate_train_test_split(Data):
    X = Data.X_transformed
    Y = Data.Y_transformed

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
    PATHS=['Models/NeuralNetworks/SavedModels2/hdf5_files/GJOA_OIL2_WELLS_mae_D2_W50_L28e-05_DPR0.h5',
           'Models/NeuralNetworks/SavedModels2/hdf5_files/GJOA_OIL2_WELLS_mae_D2_W50_L20.0001_DPR0.h5',
           'Models/NeuralNetworks/SavedModels2/hdf5_files/GJOA_OIL2_WELLS_mae_D2_W50_L25e-05_DPR0.h5'
            ]
    #pressure_weights=
    if DATA_TYPE=='GAS':
        PATH = 'Models/NeuralNetworks/SavedModels2/Weights/NCNET2_OIL_QGAS_ENSEMBLE_MODEL_MAE.h5'
        X_train, Y_train, X_val, Y_val, X_test, Y_test = get_train_test_val_data(X, Y, test_size=0.1, val_size=0.2)
        #model=NCNET_CHKPRES.SSNET3_PRESSURE(Data)
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
        # print(model.model.get_config())
        # model.fit(X_train,Y_train,X_val,Y_val)

        # Fit with old data
        #model.update_model()
        model.fit(X_train, Y_train, X_val, Y_val)
        #X_train, Y_train, X_val, Y_val, X_test, Y_test = get_train_test_val_data(X, Y, test_size=0.0, val_size=0.2)
    else:
        #GJOA_QOIL
        #pass
        PATH = 'Models/NeuralNetworks/SavedModels2/Weights/GJOA_OIL2_WELLS_mae_D2_W50_L20.0001_DPR0.h5'
        X_train, Y_train, X_val, Y_val, X_test, Y_test = get_train_test_val_data(X, Y, test_size=0.1, val_size=0.2)
        model=NCNET1_GJOA2.NCNET1_GJOA2()
        #model=NCNET1_GJOA2.ENSEMBLE(PATHS)
        #model.model.load_weights(PATH)
        #model = NCNET1_GJOA2.ENSEMBLE(PATHS)
        #model=NCNET_VANILLA_GJOA2.NCNET_VANILLA()
        #model=CNN_test.CNN_GJOAOIL()
        #model = NCNET_CHKPRES.SSNET3_PRESSURE(n_depth=2,n_width=50,l2w=0.0001)
        #model.model.load_weights(PATH,by_name=True)
        #model = test_model.Test_model()
        #model=NCNET4_combined.NET4_W_PRESSURE(PATH)

        # model.initialize_zero_thresholds(Data)
        model.initialize_chk_thresholds(Data, True)
        # model.initialize_zero_thresholds(Data)
        start = time.time()
        print(model.get_config())
        # print(model.model.get_config())
        #model.fit(X_train,Y_train,X_val,Y_val)

        # Fit with old data
        #model.update_model()
        model.fit(X_train, Y_train, X_val, Y_val)
    end = time.time()
    print('Fitted with time: {}'.format(end - start))


    #m

    #X_train, Y_train, X_val, Y_val, X_test, Y_test=get_train_test_val_data(X,Y,test_size=0.0,val_size=0.3)


    #Fit with new data
    if False:
        split_length = int(len(X_new) * (1 - 0.2))
        X_train, X_val = X_new[0:split_length], X_new[split_length:-1]
        Y_train, Y_val = Y_new[0:split_length], Y_new[split_length:-1]
        #X_train, Y_train, X_val, Y_val, X_test, Y_test = get_train_test_val_data(X_new, Y_new, test_size=0.0, val_size=0.4)

        model.fit(X_train, Y_train, X_val, Y_val)


    #X_train,Y_train,X_val,Y_val=X_train_new,Y_train_new,X_val_new,Y_val_new

    #print(model.model.get_weights())
    #model.fit(X_train[], Y_train, X_val, Y_val)



    #print(model.predict(X_train))




    #EVAL
    scores,scores_latex = evaluate_model(model,Data, X_train, X_val, Y_train, Y_val)
    print(scores)



    #model.save_model_config(scores_latex)
    #MODEL_SAVEFILE_NAME = 'NCNET2_OIL_QGAS_INCEPTION_LOCALLY_P_DENSE'
    model.save_model_to_file(model.model_name, scores)

    input_cols =[]


    output_cols=[]
    #plt.plot(model.get_history())
    #plt.title('History')
    visualize(model, Data, X_train, X_val, Y_train, Y_val, output_cols=output_cols, input_cols=input_cols)

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




def validateCV(Data,cv=10):
    X, Y, X_train, Y_train, X_val, Y_val, X_test, Y_test = get_train_test_val_data(Data, test_size=0.1, val_size=0.2)

    kfold=model_selection.KFold(n_splits=5,random_state=False)

    scores_rmse_train=np.empty(shape=(0,))
    scores_r2_test = np.empty(shape=(0,))
    scores_rmse_test = np.empty(shape=(0,))
    scores_r2_train = np.empty(shape=(0,))

    rmse_train_list = []
    rmse_test_list = []
    r2_train_list = []
    r2_test_list = []
    #zprint scores
    #print(chkInputs.index)
    i=1
    for train_index,test_index in kfold.split(X.index):
        #print("TRAIN:", train_index, "TEST:", test_index)
        #print(chkInputs[test_index])

        #print('Train index: {}'.format(train_index))
        #print('Val index: {}'.format(test_index))
        #model = NN1.SSNET1()
        model = NCNET_CHKPRES.SSNET3_PRESSURE()
        model.initialize_chk_thresholds(Data, True)
        X_train=X.iloc[train_index]
        X_val=X.iloc[test_index]
        Y_train=Y.iloc[train_index]
        Y_val=Y.iloc[test_index]
        model.fit(X_train, Y_train,X_val,Y_val)
        score_train_MSE, score_test_MSE, score_train_r2, score_test_r2 = model.evaluate(X_train, X_val, Y_train,
                                                                                        Y_val)

        rmse_train_list.append(np.sqrt(score_train_MSE[-1]))
        rmse_test_list.append(np.sqrt(score_test_MSE[-1]))
        r2_train_list.append(score_train_r2[-1])
        r2_test_list.append(score_test_r2[-1])
        print('Scores on fold: {} \n RMSE: {} \n R2: {}'.format(i,rmse_test_list[-1],r2_test_list[-1]))
        del model
        i+=1
    RMSE_TRAIN = np.mean(rmse_train_list)
    RMSE_TEST = np.mean(rmse_test_list)
    R2_TEST = np.mean(r2_test_list)
    R2_TRAIN = np.mean(r2_train_list)
    s_rmse = "Accuracy RMSE \n TRAIN: %0.2f (+/- %0.2f) \n TEST: %0.2f (+/- %0.2f)" % (
    RMSE_TRAIN, np.std(rmse_train_list) * 2, RMSE_TEST, np.std(rmse_test_list) * 2)
    s_r2 = "Accuracy R2 \n TRAIN: %0.2f (+/- %0.2f) \n TEST: %0.2f (+/- %0.2f)" % (
        R2_TRAIN, np.std(r2_train_list) * 2, R2_TEST, np.std(r2_test_list) * 2)

    print(s_rmse)
    print(s_r2)


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

def grid_search2(Data):
    X = Data.X_transformed
    Y = Data.Y_transformed

    cum_thresh=15

    X_train, Y_train, X_val, Y_val, X_test, Y_test = get_train_test_val_data(X, Y, test_size=0.1, val_size=0.2)

    search_params={'n_depth':[2],'n_width':[50],
                   'l2w':[0.0001],'seed':np.random.randint(1,10000,100)}

    grid_params=generate_grid(search_params)

    len_grid=len(grid_params)

    print('Size of search space: {}'.format(len_grid))

    best_sum_cumperf=-1e100
    best_r2_train=None
    best_rmse_train=None
    best_r2_test = None
    best_rmse_test = None
    best_params={}
    filename='GRID_SEARCH_GAS_SEED_HUBERLOSS'
    for params in grid_params:
        params['seed']=int(params['seed'])
        model = NCNET1_GJOA2.NCNET1_GJOA2(**params)
        print('Training with params: {}'.format(params))
        model.initialize_chk_thresholds(Data, True)
        model.fit(X_train,Y_train,X_val,Y_val)
        model.update_model()
        model.fit(X_train, Y_train, X_val, Y_val)
        score_train_MSE, score_test_MSE, score_train_r2, score_test_r2, cols = model.evaluate(Data, X_train, X_val, Y_train, Y_val)

        cum_perf=get_cumulative_deviation(model,Data,X_val,Y_val)
        #cum_perf_sum=cum_perf['GJOA_TOTAL_SUM_QOIL'][15]

        cum_perf_sum=count_number_of_samples_below_cum_devation(15, cum_perf)

        if cum_perf_sum>best_sum_cumperf:
            best_sum_cumperf=cum_perf_sum
            best_params=params
            best_rmse_test=np.sqrt(score_test_MSE)
            best_r2_test=score_test_r2
            best_rmse_train = np.sqrt(score_train_MSE)
            best_r2_train = score_train_r2
        del model


        print('Cols: {}'.format(cols))
        print('THIS SUM: {}, BEST SUM: {}'.format(cum_perf_sum, best_sum_cumperf))
        print('Best params:{} \n'.format(best_params))
        print('Best TEST RMSE: {} \n Best R2: {}'.format(best_rmse_test,best_r2_test))
        print('Best TRAIN RMSE: {} \n Best R2: {}'.format(best_rmse_train, best_r2_train))



    s='Best results: \n'
    s+='Cols: {}\n'.format(cols)
    s+='Best params:{} \n'.format(best_params)
    s+='Best SUM: {} \n'.format(best_sum_cumperf)
    s+='Best TEST\n RMSE: {} \n Best R2: {}\n'.format(best_rmse_test, best_r2_test)
    s+='Best TRAINn\n RMSE: {} \n Best R2: {}\n'.format(best_rmse_train, best_r2_train)
    PATH = 'Models/NeuralNetworks/'+filename
    f = open(PATH, 'w')
    f.write(s)
    f.close()


def grid_search(Data):
    X, Y, X_train, Y_train, X_val, Y_val, X_test, Y_test = get_train_test_val_data(Data, test_size=0.1, val_size=0.2)


    depth=[1,2]
    width=[20,50,100,150,200]
    mn_hidden=[1,2,3,4,5]
    mn_out=[1,2,3,4,5]
    n_depth=[2,3]
    n_width=[20,50,100]

    search_params={'depth':[1,2],'width':[20,50,100,150,200],'mn_hidden':[1,2,3,4],'mn_out':[1,2,3,4]}

    prev_sum=10000000000
    prev_max = 10000000000
    best_mnh=0
    best_mno=0
    best_depth=0
    best_width=0
    best_test_mse=None
    best_test_r2=None

    best_mnh_sum = 0
    best_mno_sum = 0
    best_depth_sum = 0
    best_width_sum=0
    best_test_mse_sum = None
    best_test_r2_sum = None
    for mnh in mn_hidden:
        for mno in mn_out:
            for depth in n_depth:
                for width in n_width:
                    #model = NCNET_CHKPRES.SSNET3_PRESSURE(Data,mno,mnh,depth)
                    model = NCNET1_GJOA2.NCNET1_GJOA2(mnh,mno,depth,width)
                    model.initialize_chk_thresholds(Data, True)
                    print('Training with: MNH: {}, MNO: {}, Depth {}, Width: {}'.format(mnh,mno,depth,width))
                    model.fit(X_train,Y_train,X_val,Y_val)
                    model.update_model()
                    model.fit(X_train, Y_train, X_val, Y_val)
                    score_train_MSE, score_test_MSE, score_train_r2, score_test_r2, cols = model.evaluate(Data, X_train, X_val, Y_train, Y_val)

                    curr_max=np.max(np.sqrt(score_test_MSE))
                    curr_sum=np.sum(np.sqrt(score_test_MSE))

                    print(np.sqrt(score_test_MSE))
                    print('THIS MAX: {}, SUM: {}'.format(curr_max,curr_sum))


                    if curr_max<prev_max:
                        prev_max=curr_max
                        best_mnh=mnh
                        best_mno=mno
                        best_width=width
                        best_depth=depth
                        best_test_mse=np.sqrt(score_test_MSE)
                        best_test_r2=score_test_r2
                    if curr_sum<prev_sum:
                        prev_sum=curr_sum
                        best_mnh_sum=mnh
                        best_mno_sum=mno
                        best_width_sum=width
                        best_depth_sum=depth
                        best_test_mse_sum=np.sqrt(score_test_MSE)
                        best_test_r2_sum=score_test_r2
                    del model
                    print('Current best: \n MNH: {}, \n MNO: {} \n Depth: {} \n Width: {}'.format(best_mnh, best_mno,best_depth,best_width))
                    print('Current best sum: \n MNH: {}, \n MNO: {}\n Depth: {} \n Width: {}'.format(best_mnh_sum, best_mno_sum,best_depth_sum,best_width_sum))
    print('Best results:')
    print(cols)
    print('Best max: \n MNH: {}, \n MNO: {} \n Depth: {} \n Width: {}'.format(best_mnh, best_mno,best_depth,best_width))
    print(best_test_mse)
    print(best_test_r2)

    print('Best sum: \n MNH: {}, \n MNO: {}\n Depth: {}\n Width: {}'.format(best_mnh_sum, best_mno_sum,best_depth_sum,best_width_sum))
    print(best_test_mse_sum)
    print(best_test_r2_sum)


def ensemble_learning(Data):
    X = Data.X_transformed
    Y = Data.Y_transformed

    cum_thresh = 15

    X_train, Y_train, X_val, Y_val, X_test, Y_test = get_train_test_val_data(X, Y, test_size=0.1, val_size=0.2)

    search_params = {'n_depth': [4], 'n_width': [50],
                     'l2w': [0.000001], 'seed': np.random.randint(1, 10000, 10)}
    grid_params = generate_grid(search_params)

    len_grid = len(grid_params)

    print('Size of search space: {}'.format(len_grid))


    name='ENSEMBLE_LEARNING_GAS_'
    PATH='Models/NeuralNetworks/SavedModels2/Weights/'
    PATHS=[]
    i=1
    for params in grid_params:
        params['seed'] = int(params['seed'])
        model = NCNET1_GJOA2.NCNET1_GJOA2(**params)
        model.model_name=name+str(i)
        PATHS.append(PATH+model.model_name+'.h5')
        i+=1

        print('Training with params: {}'.format(params))
        model.initialize_chk_thresholds(Data, True)
        model.fit(X_train, Y_train, X_val, Y_val)
        model.update_model()
        model.fit(X_train, Y_train, X_val, Y_val)

        scores, scores_latex = evaluate_model(model, Data, X_train, X_val, Y_train, Y_val)
        model.save_model_to_file(model.model_name, scores)


        del model

    print(PATHS)








