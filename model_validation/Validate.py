
import Models.NeuralNetworks.NET1 as NN1
import Models.NeuralNetworks.NN_external as NNE
from Models.NeuralNetworks import NET2_PRESSURE,NET3,NCNET_CHKPRES,NET_MISC,NCNET1_GJOA2,NCNET_VANILLA_GJOA2
import DataManager as DM
from sklearn import model_selection
from sklearn import metrics
from sklearn.svm import SVR
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn
from sklearn import ensemble

MODEL_SAVEFILE_NAME='SSNET2_PRETRAINING_2'
def train_test_split(X,Y,Y_Q,test_size):
    split_length=int(len(X)*(1-test_size))
    X_train,X_test=X[0:split_length],X[split_length-1:-1]
    Y_train,Y_test=Y[0:split_length],Y[split_length-1:-1]
    Y_Q_train, Y_Q_test = Y_Q[0:split_length], Y_Q[split_length - 1:-1]
    return X_train,X_test,Y_train,Y_test,Y_Q_train, Y_Q_test

def getTrainTestSplit(input,output,train_index,test_index):
    input_train, output_train = {'input1': input['input1'][train_index], 'input2': input['input2'][train_index]}, {
        'main_output': output['main_output'][train_index]}
    input_test, output_test = {'input1': input['input1'][test_index], 'input2': input['input2'][test_index]}, {
        'main_output': output['main_output'][test_index]}

    return input_train,output_train,input_test,output_test

def validate_train_test_split(Data):
    #Data.transform_Y_with_new_scale(100)
    X=Data.X_transformed#[500:-1]
    Y=Data.Y_transformed#[500:-1]
    Y_Q=Data.Y_Q_transformed
    #X, Y = remove_chk_zeros(X, Y, 'B2')
    X, X_test, Y, Y_test, _, _ = train_test_split(X, Y, Y, test_size=0.1)
    X_train, X_val, Y_train, Y_val, _, _ = train_test_split(X, Y, Y, test_size=0.2)




    #params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
    #          'learning_rate': 0.01, 'loss': 'ls'}
    #model=ensemble.GradientBoostingRegressor(**params)
    #model = SVR(C=1000, gamma=0.001,epsilon=0.0001)
    #model.fit(X_train,Y_train)
    #scores= print_scores(model, X_train, X_test, Y_train, Y_test)
    #lotTrueAndPredicted(model, X_train, X_test, Y_train, Y_test)
    #plotPressure(model, X_train, X_test, Y_train, Y_test)
    #exit()

    #model=NCNET_CHKPRES.SSNET3_PRESSURE()

    #model = NET2_PRESSURE.SSNET2()

    #model = NNE.SSNET_EXTERNAL(MODEL_SAVEFILE_NAME)
    MODEL_SAVEFILE_NAME='NCNET1_2_WITHOUT_ONOFF'
    #model = NN1.SSNET1()
    model=NCNET1_GJOA2.NCNET1_GJOA2()
    #model=NCNET_VANILLA_GJOA2.NCNET_VANILLA()
    #model=NET_MISC.NETTEST()

    #model=NET3.SS NET3()

    #model.update_model()

    model.initialize_chk_thresholds(Data, True)
    model.fit(X_train,Y_train,X_val,Y_val)

    #EVAL


    scores = print_scores(model, X_train, X_val, Y_train, Y_val)
    #exit()


    model.save_model_to_file(MODEL_SAVEFILE_NAME, scores)
    input_cols =[]#['F1_CHK','B2_CHK','D3_CHK','E1_CHK']
    output_cols =['C1_QOIL', 'C2_QOIL','C3_QOIL', 'C4_QOIL', 'B1_QOIL','B3_QOIL', 'D1_QOIL',  'GJOA_TOTAL_QOIL_SUM']#['F1_PWH','F1_PDC','B2_PWH','B2_PDC','D3_PWH','D3_PDC','E1_PWH','E1_PDC']#['F1_QGAS','B2_QGAS','D3_QGAS','E1_QGAS','GJOA_QGAS']
    model.visualize(X_train, X_val, Y_train, Y_val, input_cols=input_cols,output_cols=output_cols)


def validateRepeat(Data):
    X = Data.X_transformed  # [300:-1]
    Y = Data.Y_transformed  # [300:-1]

    X, X_test, Y, Y_test, _, _ = train_test_split(X, Y, Y, test_size=0.1)
    X_train, X_val, Y_train, Y_val, _, _ = train_test_split(X, Y, Y, test_size=0.2)

    N_REPEAT=10


    mse_train_list=[]
    mse_test_list=[]
    r2_train_list=[]
    r2_test_list=[]


    for i in range(N_REPEAT):
        #model = NCNET_CHKPRES.SSNET3_PRESSURE()
        #model = NN1.SSNET1()
        model = NCNET1_GJOA2.NCNET1_GJOA2()
        #model = NCNET_VANILLA_GJOA2.NCNET_VANILLA()
        #model=NET2_PRESSURE.SSNET2()
        model.initialize_chk_thresholds(Data, True)
        conf=model.get_config()
        model_name=model.model_name
        model.fit(X_train, Y_train, X_val, Y_val)
        score_train_MSE, score_test_MSE, score_train_r2, score_test_r2 = model.evaluate(X_train, X_val, Y_train,
                                                                                        Y_val)
        del model

        mse_train_list.append(score_train_MSE[-1])
        mse_test_list.append(score_test_MSE[-1])
        r2_train_list.append(score_train_r2[-1])
        r2_test_list.append(score_test_r2[-1])

    RMSE_TRAIN=np.mean(np.sqrt(mse_train_list))
    RMSE_TEST=np.mean(np.sqrt(mse_test_list))
    R2_TEST=np.mean(r2_test_list)
    R2_TRAIN=np.mean(r2_train_list)


    s_rmse = "Accuracy RMSE \n TRAIN: %0.2f (+/- %0.2f) \n VAL: %0.2f (+/- %0.2f)" % (RMSE_TRAIN, np.std(np.sqrt(mse_train_list)) * 2,RMSE_TEST, np.std(np.sqrt(mse_test_list)) * 2)
    s_r2 = "Accuracy R2 \n TRAIN: %0.2f (+/- %0.2f) \n VAL: %0.2f (+/- %0.2f)" % (
    R2_TRAIN, np.std(r2_train_list) * 2, R2_TEST, np.std(r2_test_list) * 2)

    print(s_rmse)
    print(s_r2)
    save_to_file(model_name + '_NREPEAT' + str(N_REPEAT)+'_results_QOIL_PDC_PWH_PBH_L6_chkthresh',conf+s_rmse+s_r2+'\n NREPEAT: '+str(N_REPEAT))




def validateCV(Data,cv=10):
    #input, output = DM.get_concrete_data()

    X=Data.X_transformed
    Y=Data.Y_transformed
    X, X_test, Y, Y_test,_,_ = train_test_split(X, Y,Y ,test_size=0.1)

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

def print_scores(model,X_train,X_test,Y_train,Y_test):
    score_train_MSE, score_test_MSE, score_train_r2, score_test_r2 = model.evaluate(X_train, X_test, Y_train, Y_test)

    #score_train_r2=metrics.r2_score(Y_train,model.predict(X_train))
    #score_test_r2 = metrics.r2_score(Y_test,model.predict(X_test))
    #score_train_MSE=metrics.mean_squared_error(Y_train,model.predict(X_train))
    #score_test_MSE = metrics.mean_squared_error(Y_test, model.predict(X_test))

    # scores='#### Scores #### \n RMSE train {0:0.2f} \n RMSE test: {1:0.2f} ' \
    #       '\n R2 train: {2:0.2f} \n R2 test: {3:0.2f} \n#### ------ ####'.format(np.sqrt(score_train),np.sqrt(score_test),score_train_r2,score_test_r2)
    scores = '#### Scores #### \n RMSE train {} \n RMSE test: {} ' \
             '\n R2 train: {} \n R2 test: {} \n#### ------ ####'.format(np.sqrt(score_train_MSE),
                                                                        np.sqrt(score_test_MSE), score_train_r2,
                                                                        score_test_r2)
    print(scores)
    return scores

def remove_chk_zeros(X,Y,well):

    #X_cols=[well+'_PDC',well+'_CHK']
    #Y_cols=[well+'_PWH']

    #Y=Y[Y_cols]
    #X=X[X_cols]

    ind=X[well+'_CHK']<0.05
    Y=Y[~ind]
    X=X[~ind]
    return X,Y
def save_to_file(filename,str):
    PATH = '/Users/UAC/GITFOLDERS/MasterThesisCode/Models/NeuralNetworks/'
    PATH='C:/users/ugurac/Documents/GITFOLDERS/MasterThesisCode/Models/NeuralNetworks/'
    f = open(PATH + filename + '_config', 'w')
    f.write(str)
    f.close()