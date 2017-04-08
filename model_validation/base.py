
from sklearn import model_selection
from sklearn import metrics
from sklearn.svm import SVR
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn
from sklearn import ensemble
import DataManager as DM

from random import seed
from random import random
from random import randrange

def subsample(X,Y):
    X_new=pd.DataFrame()
    Y_new=pd.DataFrame()

    n_sample = len(X)
    for i in range(n_sample):
        index = np.random.randint(0,n_sample)
        X_new=X_new.append(X.iloc[index])
        Y_new=Y_new.append(Y.iloc[index])

    return X_new,Y_new

def get_sample_deviation(measured,predicted):
    diff=np.abs(measured-predicted)
    delta=1e-100

    #return np.abs(predicted/(measured+1e-10))*100
    return diff/(measured)*100
def get_sample_deviation_flow(measured,predicted):
    diff=measured-predicted
    print(np.mean(diff))
    #diff=diff-np.mean(diff)
    delta=1e-100

    #return np.abs(predicted/(measured+1e-10))*100
    #ind_zero=measured==0
    #print(np.sum(ind_zero))
    res=diff/(measured)*100
    #res[ind_zero]=0
    #res.loc[ind_zero] = 0

    return res
def startswith(col,tag):
    return col.split('_')[0]==tag
def remove_zero_measurements(X,Y,cols):

    for col in cols:
        if not startswith(col,'GJOA'):
            ind_zero_mes=Y[col]==0
            ind_not_zero_chk=X[col.split('_')[0] + '_CHK'] != 0
            X=X[~(ind_zero_mes&ind_not_zero_chk)]
            Y=Y[~(ind_zero_mes&ind_not_zero_chk)]
    return X,Y
def get_predicted_and_measured_df(model,data,X,Y):
    cols = model.output_tag_ordered_list

    measured = pd.DataFrame(data=data.inverse_transform(Y, 'Y'), columns=cols)

    predicted = data.inverse_transform(model.predict(X), 'Y')
    predicted = pd.DataFrame(data=predicted, columns=cols)
    predicted = predicted.set_index(Y.index)
    return measured,predicted

def get_cumulative_deviation(model,data,X,Y):

    cols = model.output_tag_ordered_list
    deviation_range = np.arange(0, 50, 1)

    measured, predicted=get_predicted_and_measured_df(model,data,X,Y)

    deviation_points = get_sample_deviation(measured, predicted)
    deviation_points.fillna(0, inplace=True)



    cumulative_deviation=pd.DataFrame(data=np.zeros((len(deviation_range),len(cols))),columns=cols)
    cumulative_deviation=cumulative_deviation.set_index(deviation_range)
    cumulative_deviation.index.name=None

    N = len(deviation_points)
    for col in cols:
        for percentage in deviation_range:
            cumulative_deviation[col][percentage]=np.sum(deviation_points[col]<=percentage)/N*100

    #print(cumulative_deviation)
    #for i in deviation_points.index.values:
    #    print(deviation_points['B1_QGAS'].loc[[i]])
    return cumulative_deviation


def get_absolute_deviation(model,data,X,Y):
    cols = model.output_tag_ordered_list
    #deviation_range = np.arange(0, 50, 1)

    measured, predicted = get_predicted_and_measured_df(model, data, X, Y)

    deviation_points = get_sample_deviation(measured, predicted)
    deviation_points.fillna(0, inplace=True)
    for key in cols:
        if key.split('_')[0] not in ['GJOA']:
            chk_col=key.split('_')[0]+'_CHK'
            ind_zero=data.inverse_transform(X,'X')[chk_col]<5
            deviation_points.loc[ind_zero, key] = 0

    return deviation_points

def count_number_of_samples_below_cum_devation(thresh,cumulative_error,tag):
    return cumulative_error.sum(axis=1)[tag][thresh]


def train_test_split(X,Y,test_size):
    split_length=int(len(X)*(1-test_size))
    X_train,X_test=X[0:split_length],X[split_length-1:-1]
    Y_train,Y_test=Y[0:split_length],Y[split_length-1:-1]
    return X_train,X_test,Y_train,Y_test

def get_train_test_val_data(X,Y,test_size,val_size):
    #X = Data.X_transformed
    #Y = Data.Y_transformed

    #X=X.reshape(1,X.shape[0],X.shape[1])
    #Y = Y.reshape(1, Y.shape[0], Y.shape[1])
    # X, Y = remove_chk_zeros(X, Y, 'B2')
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=test_size)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val_size)

    return X_train,Y_train,X_val,Y_val,X_test,Y_test
def split_data(X,Y,split_size):

    X_start, X_end, Y_start, Y_end = train_test_split(X, Y, test_size=split_size)


    return X_start,Y_start,X_end,Y_end




def evaluate_model(model,data,X_train,X_test,Y_train,Y_test):
    score_train_MSE, score_test_MSE, score_train_r2, score_test_r2, cols = model.evaluate(data, X_train, X_test,Y_train, Y_test)

    return print_scores(data, Y_train, Y_test, score_train_MSE, score_test_MSE, score_train_r2, score_test_r2,cols),scores_to_latex(data, Y_train, Y_test, score_train_MSE, score_test_MSE, score_train_r2, score_test_r2,cols)
def print_scores(data,Y_train,Y_test,score_train_MSE, score_test_MSE, score_train_r2, score_test_r2,cols):

    #Treig og daarlig kode, fiks paa dette!!!!!!!!!
    n_empty_space=30
    def print_empty_space(s,n):
        s+=' '*n
        return s
    def scores_to_tabbed_string(s,scores_train,score_test,cols,Y=[]):
        for i,col in zip(range(len(cols)),cols):
            s_temp='{0}: {1:0.2f}'.format(col,scores_train[i])
            s_len=len(s_temp)
            s_temp=print_empty_space(s_temp,n_empty_space-s_len)
            s_temp =''.join((s_temp,'{0}: {1:0.2f}'.format( col,score_test[i])))
            if len(Y)>0:
                Y_MEAN=np.mean(Y[col][Y[col]>0])
                s_temp = print_empty_space(s_temp, 2*n_empty_space - len(s_temp))
                s_temp=''.join((s_temp,'{0}: {1:0.2f}%'.format(col,score_test[i]/Y_MEAN*100)))
                s_temp = print_empty_space(s_temp, 3 * n_empty_space - len(s_temp)+10)
                s_temp =''.join((s_temp, '{0}: {1:0.2f}'.format(col, Y_MEAN)))
            s_temp=''.join((s_temp,'\n'))
            s=''.join((s,s_temp))
        return s



    s='                 #### Scores #### \n'
    s=''.join((s,'RMSE TRAIN:'))
    s=print_empty_space(s,n_empty_space-len('RMSE TRAIN:'))
    s=''.join((s,'RMSE VAL:'))
    s = print_empty_space(s, n_empty_space - len('RMSE VAL:'))
    s=''.join((s,'Percentage error (VAL/MEAN)*100'))
    s = print_empty_space(s, 10+n_empty_space - len('Percentage error (VAL/MEAN)*100'))
    s = ''.join((s,'MEAN'))
    s+=''.join((s,'\n'))
    s+='------------------------------------------------------------------------------------------------------------------------\n'
    s=scores_to_tabbed_string(s,np.sqrt(score_train_MSE),np.sqrt(score_test_MSE),cols,data.inverse_transform(Y_test,'Y'))
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

def scores_to_latex(data, Y_train, Y_test, score_train_MSE, score_test_MSE, score_train_r2, score_test_r2, cols):

        # Treig og daarlig kode, fiks paa dette!!!!!!!!!
        n_empty_space = 30

        def print_empty_space(s, n):
            s += ' ' * (n-1)
            s+='&'
            return s

        def scores_to_tabbed_string(s, scores_train, score_test, cols, Y=[]):
            for i, col in zip(range(len(cols)), cols):
                #col=col.replace('_','_')
                s_temp = '{0}& {1:0.2f}'.format(col.replace('_','\_'), scores_train[i])
                s_len = len(s_temp)
                s_temp = print_empty_space(s_temp, n_empty_space - s_len)
                s_temp = ''.join((s_temp, '{0:0.2f}'.format(score_test[i])))
                if len(Y) > 0:
                    Y_MEAN = np.mean(Y[col][Y[col] > 0])
                    s_temp = print_empty_space(s_temp, 2 * n_empty_space - len(s_temp))
                    s_temp = ''.join((s_temp, '{0:0.2f}\%'.format(score_test[i] / Y_MEAN * 100)))
                    s_temp = print_empty_space(s_temp, 3 * n_empty_space - len(s_temp) + 10)
                    s_temp = ''.join((s_temp, '{0:0.2f}'.format(Y_MEAN)))
                s_temp = ''.join((s_temp, '\\\ \n'))
                s = ''.join((s, s_temp))
            return s

        #s = '                 #### Scores #### \\\ \n'
        s='\hline \n'
        s = ''.join((s, 'Tag&RMSE TRAIN:'))

        s = print_empty_space(s, n_empty_space - len('RMSE TRAIN:'))
        s = ''.join((s, 'RMSE VAL:'))
        s = print_empty_space(s, n_empty_space - len('RMSE VAL:'))
        s = ''.join((s, 'Percentage error (VAL/MEAN)*100'))
        s = print_empty_space(s, 10 + n_empty_space - len('Percentage error (VAL/MEAN)*100'))
        s = ''.join((s, 'MEAN'))
        s+='\\\\'
        s += '\n \hline \n '
        #s = ''.join((s, '\\\ \n'))
        #s += '------------------------------------------------------------------------------------------------------------------------\\\ \n'
        s = scores_to_tabbed_string(s, np.sqrt(score_train_MSE), np.sqrt(score_test_MSE), cols,
                                    data.inverse_transform(pd.concat([Y_train, Y_test], axis=0),'Y'))
        #s += '-------------------------------------------------------\\\ \n'
        s += '\n \hline \n '
        s += 'Tag&R2 TRAIN:'
        s = print_empty_space(s, n_empty_space - len('R2 TRAIN:'))
        s += 'R2 VAL:&\\\ '
        s += '\n \hline \n '
        #s += '-------------------------------------------------------\\\ \n'
        s = scores_to_tabbed_string(s, score_train_r2, score_test_r2, cols)
        s += '\n \hline \n '
        #s += '-------------------------------------------------------\\\ \n'
        #s += '#### ------ #### \\\ \n'
        return s