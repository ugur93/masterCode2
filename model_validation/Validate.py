
import Models.NeuralNetworks.NET1 as NN1
import Models.NeuralNetworks.NN_external as NNE
from Models.NeuralNetworks import NET2_PRESSURE
import DataManager as DM
from sklearn import model_selection
from sklearn import metrics
from sklearn.svm import SVR
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn

EXTERNAL=False
def train_test_split(X,Y,test_size):
    split_length=int(len(X)*(1-test_size))
    X_train,X_test=X[0:split_length],X[split_length-1:-1]
    Y_train,Y_test=Y[0:split_length],Y[split_length-1:-1]
    return X_train,X_test,Y_train,Y_test

def getTrainTestSplit(input,output,train_index,test_index):
    input_train, output_train = {'input1': input['input1'][train_index], 'input2': input['input2'][train_index]}, {
        'main_output': output['main_output'][train_index]}
    input_test, output_test = {'input1': input['input1'][test_index], 'input2': input['input2'][test_index]}, {
        'main_output': output['main_output'][test_index]}

    return input_train,output_train,input_test,output_test

def validate_train_test_split(Data):
    X=Data.X_transformed
    Y=Data.Y_transformed



    #print(Data)

    X_train, X_test, Y_train, Y_test=train_test_split(X, Y,test_size=0.2)



    if EXTERNAL:
        model = NNE.SSNET_EXTERNAL('SSNET1_GJOA_1_GJOA')
        model.initialize_thresholds(Data, True)
        #model.fit(X_train, Y_train)
        #model.save_model_to_file('SSNET1_GJOA_1' + Data.name, save_weights=True)
    else:
        model=NN1.SSNET1()
        #model=NET2_PRESSURE.SSNET2()
        model.initialize_thresholds(Data,True)

        #print_weights(model)
        model.fit(X_train, Y_train)

        model.save_model_to_file('SSNET1_GJOA_double_inception_w10_depth2_newscaler2_pressure_' + Data.name, save_weights=True)
    #print_weights(model)

    #model = SVR(C=100, gamma=0.0005)
    #model.fit(X_train,Y_train)



    score_test=metrics.mean_squared_error(Y_test,model.predict(X_test))
    score_train = metrics.mean_squared_error(Y_train, model.predict(X_train))
    score_test_r2 = metrics.r2_score(Y_test, model.predict(X_test))
    score_train_r2 = metrics.r2_score(Y_train, model.predict(X_train))

    print('#### Scores ####')
    print("RMSE train: %0.2f" % (np.sqrt(score_train)))
    print("RMSE test: %0.2f" % (np.sqrt(score_test)))
    print("R2 train: %0.2f" % (score_train_r2))
    print("R2 test: %0.2f" % (score_test_r2))
    print('#### ------ ####')
    plotTrueAndPredicted(model, X_train, X_test, Y_train, Y_test)
    try:
        plotTrainingHistory(model)
    except(AttributeError):
        pass
    try:
        plotWellOutput(model,Data.X_transformed,Data.Y_Q_transformed)
    except(AttributeError):
        pass
    plt.show()
    #visualizeResults(model, chkInputs, wellOutputs, chk_train, chk_test, Q_train, Q_test)







def validateCV(Data,cv=10):
    #input, output = DM.get_concrete_data()

    X=Data.X
    Y=Data.Y
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.2)

    kfold=model_selection.TimeSeriesSplit(n_splits=10)

    scores=np.empty(shape=(0,))
    #zprint scores
    #print(chkInputs.index)
    for train_index,test_index in kfold.split(X.index):
        #print("TRAIN:", train_index, "TEST:", test_index)
        #print(chkInputs[test_index])
        model = NN1.SSNET1()
        X_train=X.iloc[train_index]
        X_val=X.iloc[test_index]
        Y_train=Y.iloc[train_index]
        Y_val=Y.iloc[test_index]
        model.fit(X_train, Y_train)
        scores=np.append(scores,np.sqrt(metrics.mean_squared_error(X_val,model.predict(Y_val))))
        #print(scores)
        del model

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print(scores)

def plotTrueAndPredicted(model,X_train, X_test, Y_train, Y_test):

    plt.figure()
    plt.plot(Y_train.index,Y_train,color='blue',label='Y_true - train')
    plt.plot(Y_train.index,model.predict(X_train),color='black',label='Y_pred - train')


    plt.plot(Y_test.index,Y_test,color='red',label='Y_true - test')
    plt.plot(Y_test.index,model.predict(X_test),color='green',label='Y_pred - test')

    #plt.legend(['Y_true - train','Y_pred - train','Y_true - test', 'Y_pred - test'],pos)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.,fontsize=20)




    plt.figure()

    for i,key in zip(range(4),X_train):
        plt.subplot(2,2,i+1)
        plt.scatter(X_train[key], Y_train, color='blue', label='Y_true - train')
        plt.scatter(X_train[key], model.predict(X_train), color='black', label='Y_pred - train')

        plt.scatter(X_test[key], Y_test, color='red', label='Y_true - test')
        plt.scatter(X_test[key], model.predict(X_test), color='green', label='Y_pred - test')

        # plt.legend(['Y_true - train','Y_pred - train','Y_true - test', 'Y_pred - test'],pos)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0., fontsize=20)
    #plt.show()

def plotTrainingHistory(model):
    history = model.get_history()
    plt.figure()
    plt.plot(history)
    plt.title('Loss history')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.show()
def plotWellOutput(model,X,X_Q):
    WELL_NAMES = ['F1', 'B2', 'D3', 'E1']
    plt.figure()
    for i,key in zip(range(len(WELL_NAMES)),X.columns):
        name=WELL_NAMES[i]+'_QGAS'
        X_Q_predicted=model.predict_well_output(X,WELL_NAMES[i])
        plt.subplot(2,2,i+1)
        plt.plot(X_Q[name], label=name + '_output_true',color='blue')
        plt.plot(X_Q_predicted,color='red',label=name+'_output_predicted')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0., fontsize=10)
        plt.title(key)
    plt.figure()
    i=1
    for i,well_name in zip(range(4),WELL_NAMES):
        key=well_name+'_CHK'
        name=well_name+'_QGAS'
        data=model.predict_well_output(X,well_name)
        plt.subplot(2,2,i+1)
        plt.scatter(X[key],X_Q[name], label=name + '_output_true',color='blue')
        plt.scatter(X[key],data,color='red',label=name+'_output_predicted')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0., fontsize=10)
        plt.title(key)
        i+=1


def print_weights(model):
    try:
        print('')
        print(model.get_layer_weights('dense_1'))
        print(model.get_layer_weights('dense_2'))
        print(model.get_layer_weights('dense_3'))
        print(model.get_layer_weights('dense_4'))
        print(model.get_layer_weights('E1_CHK'))
        print('')
    except(AttributeError):
        print('Error with layer names')

    try:
        print('')
        print('F1: '+str(model.get_layer_weights('F1_CHK_out')))
        print('B2: '+str(model.get_layer_weights('B2_CHK_out')))
        print('D3: '+str(model.get_layer_weights('D3_CHK_out')))
        print('E1: '+str(model.get_layer_weights('E1_CHK_out')))
        print(model.get_layer_weights('E1_CHK'))
        print('')
    except(AttributeError):
        print('Error with layer names')