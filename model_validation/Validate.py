
import Models.NeuralNetworks.NET1 as NN1
import Models.NeuralNetworks.NN_external as NNE
from Models.NeuralNetworks import NET2_PRESSURE,NET3,NCNET_CHKPRES,NET_MISC,NCNET1_GJOA2,NCNET_VANILLA_GJOA2,CNN_test



from .base import *
from .Visualize import *

#params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
    #          'learning_rate': 0.01, 'loss': 'ls'}
    #model=ensemble.GradientBoostingRegressor(**params)
    #model = SVR(C=1000, gamma=0.001,epsilon=0.0001)
MODEL_SAVEFILE_NAME='SSNET2_PRETRAINING_2'
MODEL_SAVEFILE_NAME='NCNET1_2_WITHOUT_ONOFF'




def validate(DataOIL,DataGAS):
    Data=DataOIL

    validate_train_test_split(Data)
    #validateRepeat(Data)
    #validateCV(Data)

def validate_train_test_split(Data):

    X, Y, X_train, Y_train, X_val, Y_val, X_test, Y_test=get_train_test_val_data(Data,test_size=0.1,val_size=0.2)


    #GJOA QGAS
    #model=NCNET_CHKPRES.SSNET3_PRESSURE()
    #model = NET2_PRESSURE.SSNET2()
    #model = NNE.SSNET_EXTERNAL(MODEL_SAVEFILE_NAME)
    #model = NN1.SSNET1()

    #GJOA_QOIL
    model=NCNET1_GJOA2.NCNET1_GJOA2()
    #model=NCNET_VANILLA_GJOA2.NCNET_VANILLA()
    #model=CNN_test.CNN_GJOAOIL()



    model.initialize_chk_thresholds(Data, True)
    #print(model.model.get_config())
    model.fit(X_train,Y_train,X_val,Y_val)
    model.update_model()
    model.fit(X_train, Y_train, X_val, Y_val)
    #X, Y, X_train, Y_train, X_val, Y_val, X_test, Y_test = get_train_test_val_data(Data, test_size=0.1, val_size=0.2)
    #
    #model.fit(X_train, Y_train, X_val, Y_val)

    #model.update_model_2()
    #model.fit(X_train[0:155], Y_train[0:155], X_val, Y_val)

    #EVAL
    scores = evaluate_model(model,Data, X_train, X_val, Y_train, Y_val)
    print(scores)
    #model.save_model_to_file(MODEL_SAVEFILE_NAME, scores)

    input_cols =[]#['F1_CHK','B2_CHK','D3_CHK','E1_CHK']
    #output_cols =['C1_QOIL', 'C2_QOIL','C3_QOIL', 'C4_QOIL', 'B1_QOIL','B3_QOIL', 'D1_QOIL', 'GJOA_TOTAL_QOIL_SUM']
    output_cols=['C1_QGAS', 'C2_QGAS','C3_QGAS', 'C4_QGAS', 'B1_QGAS','B3_QGAS', 'D1_QGAS', 'GJOA_OIL_QGAS']
    #output_cols=['F1_PWH','F1_PDC','B2_PWH','B2_PDC','D3_PWH','D3_PDC','E1_PWH','E1_PDC']
    #output_cols= ['F1_QGAS','B2_QGAS','D3_QGAS','E1_QGAS','GJOA_QGAS']
    output_cols=[]
    visualize(model, Data, X_train, X_val, Y_train, Y_val, output_cols=output_cols, input_cols=input_cols)
    #plt.pause(0.5)
    #model.update_model()
    #model.fit(X_train, Y_train, X_val, Y_val)
    #scores = evaluate_model(model, Data, X_train, X_val, Y_train, Y_val)
    #print(scores)
    #visualize(model, Data, X_train, X_val, Y_train, Y_val, output_cols=output_cols, input_cols=input_cols)
    plt.show()



def validateRepeat(Data):
    X, Y, X_train, Y_train, X_val, Y_val, X_test, Y_test = get_train_test_val_data(Data, test_size=0.1, val_size=0.2)

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



