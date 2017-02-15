
from sklearn import model_selection
from sklearn import metrics
from sklearn.svm import SVR
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn
from sklearn import ensemble
import DataManager as DM

def train_test_split(X,Y,test_size):
    split_length=int(len(X)*(1-test_size))
    X_train,X_test=X[0:split_length],X[split_length-1:-1]
    Y_train,Y_test=Y[0:split_length],Y[split_length-1:-1]
    return X_train,X_test,Y_train,Y_test

def get_train_test_val_data(Data,test_size,val_size):
    X = Data.X_transformed
    Y = Data.Y_transformed

    # X, Y = remove_chk_zeros(X, Y, 'B2')
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=test_size)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val_size)

    return X,Y,X_train,Y_train,X_val,Y_val,X_test,Y_test

def print_scores(model,X_train,X_test,Y_train,Y_test):
    score_train_MSE, score_test_MSE, score_train_r2, score_test_r2 = model.evaluate(X_train, X_test, Y_train, Y_test)
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