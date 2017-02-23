
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

    #X=X.reshape(1,X.shape[0],X.shape[1])
    #Y = Y.reshape(1, Y.shape[0], Y.shape[1])
    # X, Y = remove_chk_zeros(X, Y, 'B2')
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=test_size)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val_size)

    return X,Y,X_train,Y_train,X_val,Y_val,X_test,Y_test



def evaluate_model(model,data,X_train,X_test,Y_train,Y_test):
    score_train_MSE, score_test_MSE, score_train_r2, score_test_r2, cols = model.evaluate(data, X_train, X_test,
                                                                                          Y_train, Y_test)

    return print_scores(data, Y_train, Y_test, score_train_MSE, score_test_MSE, score_train_r2, score_test_r2,cols)
def print_scores(data,Y_train,Y_test,score_train_MSE, score_test_MSE, score_train_r2, score_test_r2,cols):

    #Treig og daarlig kode, fiks paa dette!!!!!!!!!
    n_empty_space=30
    def print_empty_space(s,n_empty_space):
        for k in range(n_empty_space):
            s += ' '
        return s
    def scores_to_tabbed_string(s,scores_train,score_test,cols,Y=[]):
        for i,col in zip(range(len(cols)),cols):
            s_temp='{0}: {1:0.2f}'.format(col,scores_train[i])
            s_len=len(s_temp)
            s_temp=print_empty_space(s_temp,n_empty_space-s_len)
            s_temp +='{0}: {1:0.2f}'.format( col,score_test[i])
            if len(Y)>0:
                Y_MEAN=np.mean(Y[col][Y[col]>0])
                s_temp = print_empty_space(s_temp, 2*n_empty_space - len(s_temp))
                s_temp+='{0}: {1:0.2f}%'.format(col,score_test[i]/Y_MEAN*100)
                s_temp = print_empty_space(s_temp, 3 * n_empty_space - len(s_temp)+10)
                s_temp += '{0}: {1:0.2f}'.format(col, Y_MEAN)
            s_temp+='\n'
            s+=s_temp
        return s



    s='                 #### Scores #### \n'
    s+='RMSE TRAIN:'
    s=print_empty_space(s,n_empty_space-len('RMSE TRAIN:'))
    s+='RMSE VAL:'
    s = print_empty_space(s, n_empty_space - len('RMSE VAL:'))
    s+='Percentage error (VAL/MEAN)*100'
    s = print_empty_space(s, 10+n_empty_space - len('Percentage error (VAL/MEAN)*100'))
    s += 'MEAN'
    s+='\n'
    s+='------------------------------------------------------------------------------------------------------------------------\n'
    s=scores_to_tabbed_string(s,np.sqrt(score_train_MSE),np.sqrt(score_test_MSE),cols,data.inverse_transform(pd.concat([Y_train,Y_test],axis=0)))
    s += '-------------------------------------------------------\n'
    s += 'R2 TRAIN:'
    s = print_empty_space(s, n_empty_space-len('R2 TRAIN:'))
    s += 'R2 VAL: \n'
    s += '-------------------------------------------------------\n'
    s = scores_to_tabbed_string(s, score_train_r2,score_test_r2, cols)
    s += '-------------------------------------------------------\n'
    s+='#### ------ #### \n'
    return s


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