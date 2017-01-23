
import Models.NeuralNetworks.NET1 as NN1
import Models.NeuralNetworks.NN_external as NNE
import DataManager as DM
from sklearn import model_selection
from sklearn import metrics
from sklearn.svm import SVR
import numpy as np
from matplotlib import pyplot as plt
import seaborn


def train_test_split(input,output,test_size):
    split_length=int(len(input)*(1-test_size))
    input_train,input_test=input[0:split_length],input[split_length-1:-1]
    output_train,output_test=output[0:split_length],output[split_length-1:-1]
    return input_train,input_test,output_train,output_test

def getTrainTestSplit(input,output,train_index,test_index):
    input_train, output_train = {'input1': input['input1'][train_index], 'input2': input['input2'][train_index]}, {
        'main_output': output['main_output'][train_index]}
    input_test, output_test = {'input1': input['input1'][test_index], 'input2': input['input2'][test_index]}, {
        'main_output': output['main_output'][test_index]}

    return input_train,output_train,input_test,output_test

def validate_train_test_split(chkInputs,wellOutputs,totalOutput):
    chk_train, chk_test, Q_train, Q_test=train_test_split(chkInputs, totalOutput,test_size=0.4)
    #model=SVR(C=50,gamma=0.0005)#
    model=NN1.SSNET1()
    #model=NNE.SSNET_EXTERNAL('SSNET1_test')
    #model.fit(chk_train,Q_train)
    model.save_model_to_file('SSNET1_test2',save_weights=False)



    score_test=metrics.mean_squared_error(Q_test,model.predict(chk_test))
    score_train = metrics.mean_squared_error(Q_train, model.predict(chk_train))
    score_test_r2 = metrics.r2_score(Q_test, model.predict(chk_test))
    score_train_r2 = metrics.r2_score(Q_train, model.predict(chk_train))
    print('#### Scores ####')
    print("RMSE train: %0.2f" % (np.sqrt(score_train)))
    print("RMSE test: %0.2f" % (np.sqrt(score_test)))
    print("R2 train: %0.2f" % (score_train_r2))
    print("R2 test: %0.2f" % (score_test_r2))
    print('#### ------ ####')
    plotTrueAndPredicted(model, chk_train, chk_test, Q_train, Q_test)
    #visualizeResults(model, chkInputs, wellOutputs, chk_train, chk_test, Q_train, Q_test)







def validateCV(chkInputs,wellOutputs,totalOutput,cv=10):
    #input, output = DM.get_concrete_data()

    kfold=model_selection.TimeSeriesSplit(n_splits=10)

    scores=np.empty(shape=(0,))
    #zprint scores
    #print(chkInputs.index)
    for train_index,test_index in kfold.split(chkInputs.index):
        #print("TRAIN:", train_index, "TEST:", test_index)
        #print(chkInputs[test_index])
        model = NN1.SSNET1()
        chk_train=chkInputs.iloc[train_index]
        chk_test=chkInputs.iloc[test_index]
        Q_train=totalOutput.iloc[train_index]
        Q_test=totalOutput.iloc[test_index]
        model.fit(chk_train, Q_train)
        scores=np.append(scores,np.sqrt(metrics.mean_squared_error(Q_test,model.predict(chk_test))))
        print(scores)
        del model

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print(scores)

def plotTrueAndPredicted(model,chk_train, chk_test, Q_train, Q_test):

    plt.figure()
    plt.plot(Q_train.index,Q_train,color='blue')
    plt.plot(Q_train.index,model.predict(chk_train),color='black')
    plt.plot(Q_test.index,Q_test,color='red')
    plt.plot(Q_test.index,model.predict(chk_test),color='green')
    plt.show()
def visualizeResults(model,chkInputs,wellOutputs,chk_train, chk_test, Q_train, Q_test):

    #weights=model.get_layer_weights('dense_29')
    #print(np.linalg.norm(weights[0],axis=0))
    #print(np.linalg.norm(weights[0], axis=1))
    history=model.getHistory()

    chk1_out=model.predictWellOutput(chkInputs['chk1'],1)
    plt.plot(chk1_out,color='red')
    plt.plot(wellOutputs['well_1'])
    plt.title('Well1')

    plt.figure()
    chk2_out = model.predictWellOutput(chkInputs['chk2'], 2)
    plt.plot(chk2_out, color='red')
    plt.plot(wellOutputs['well_2'])
    plt.title('Well2')

    plt.figure()
    chk3_out = model.predictWellOutput(chkInputs['chk3'], 3)
    plt.plot(chk3_out, color='red')
    plt.plot(wellOutputs['well_3'])
    plt.title('Well3')

    plt.figure()
    chk4_out = model.predictWellOutput(chkInputs['chk4'], 4)
    plt.plot(chk4_out, color='red')
    plt.plot(wellOutputs['well_4'])
    plt.title('Well4')
    #plt.show()



    plt.figure()
    plt.plot(Q_train.index,Q_train,color='blue')
    plt.plot(Q_train.index,model.predict(chk_train),color='black')
    plt.plot(Q_test.index,Q_test,color='red')
    plt.plot(Q_test.index,model.predict(chk_test),color='green')


    plt.figure()
    plt.plot(history)
    plt.title('Loss history')
    plt.show()