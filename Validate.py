
import Model
import DataManager as DM
from sklearn import model_selection
from sklearn import metrics
import numpy as np

def getTrainTestSplit(input,output,train_index,test_index):
    input_train, output_train = {'input1': input['input1'][train_index], 'input2': input['input2'][train_index]}, {
        'main_output': output['main_output'][train_index]}
    input_test, output_test = {'input1': input['input1'][test_index], 'input2': input['input2'][test_index]}, {
        'main_output': output['main_output'][test_index]}

    return input_train,output_train,input_test,output_test


def printCVScore(cv=10):
    input, output = DM.get_concrete_data()
    ffnn1 = Model.SSnet3()
    kfold=model_selection.KFold(n_splits=10)

    scores=np.empty(shape=(0,))
    #zprint scores
    for train_index,test_index in kfold.split(output['main_output']):
        input_train, output_train, input_test, output_test=getTrainTestSplit(input, output, train_index, test_index)
        ffnn1.fit(input_train, output_train)
        scores=np.append(scores,metrics.r2_score(output_test['main_output'],ffnn1.predict(input_test)))

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print scores