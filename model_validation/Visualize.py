from matplotlib import pyplot as plt
import numpy as np
from .base import *
#from matplotlib import rc
#rc('text', usetex=True)
OUTPUT_COLS_ON_SINGLE_PLOT=['Total production','GJOA_QGAS','GJOA_TOTAL_QOIL','GJOA_TOTAL_QOIL_SUM','GJOA_OIL_QGASss']

N_PLOT_SUB=0
well_names = ['C1', 'C2', 'C3', 'C4', 'B1', 'B3', 'D1']
KEY_MAP={}
for i,key in zip(range(1,len(well_names)+1),well_names):
    KEY_MAP[key+'_QGAS']='Well O{}'.format(i)
    KEY_MAP[key + '_QOIL'] = 'Well O{}'.format(i)
    KEY_MAP[key + '_PDC'] = 'Well O{}'.format(i)
    KEY_MAP[key + '_PBH'] = 'Well O{}'.format(i)
    KEY_MAP[key + '_PWH'] = 'Well O{}'.format(i)
well_names=['F1','B2','D3','E1']
for i,key in zip(range(1,len(well_names)+1),well_names):
    KEY_MAP[key+'_QGAS']='Well G{}'.format(i)
KEY_MAP['GJOA_OIL_SUM_QGAS']='Total production'
KEY_MAP['GJOA_TOTAL_SUM_QOIL']='Total production'
KEY_MAP['GJOA_QGAS']='Total production'
#print(KEY_MAP)
#exit()
for key in ['A','B','C','D']:
    KEY_MAP[key+'_QGAS']='Well '+key
TICKSIZE=25
FONTSIZELEGEND=25
FONTSIZEX=25
FONTSIZEY=25
FONTSIZETITLE=30
TRUE_DATA_TAG='(Measured)'
PREDICTED_DATA_TAG='(Prediction)'
TICK_STEP_SIZE=10
TRUE_AND_PREDICTED_Y_LABEL='Gas flow rate [Sm3/h] (scaled)'

TRUE_TRAIN_LABEL='Train '+TRUE_DATA_TAG
TRUE_TEST_LABEL='Test '+TRUE_DATA_TAG
PRED_TRAIN_LABEL='Train '+PREDICTED_DATA_TAG
PRED_TEST_LABEL='Test '+PREDICTED_DATA_TAG


def count_number_of_cols_that_ends_with(col_list,tag):
    N=0
    for key in col_list:
        if key.split('_')[-1]==tag:
            N+=1
    return N

def count_number_of_different_sensors_tags(col_list):
    tag_list=[]
    N_sensors=0

    for col in col_list:
        tag_end=col.split('_')[-1]
        if tag_end not in tag_list:
            tag_list.append(tag_end)
            N_sensors+=1
    return N_sensors,tag_list

def split_col_list(col_list,tag_list):
    multi_tag_col_list=[]


    for tag in tag_list:
        temp_col_list=[]
        for col in col_list:
            if col.split('_')[-1]==tag:
                temp_col_list.append(col)
        multi_tag_col_list.append(temp_col_list)

    return multi_tag_col_list





def ends_with(name,end_tag):
    if name.split('_')[-1] == end_tag:
        return True
    return False
def visualize(model,data, X_train, X_test, Y_train ,Y_test, output_cols=[], input_cols=[],with_line_plot=False,with_separate_plot=False):

    remove_zero_chk=False
    plot_history(model)
    plot_cumulative_performance(model,data, X_train, X_test, Y_train, Y_test)
    #plot_input_vs_output(model, data, X_train, X_test, Y_train, Y_test, input_cols=input_cols, output_cols=output_cols,
    #                     remove_zero_chk=remove_zero_chk)
    #plot_true_and_predicted_with_input(model, data, X_train, X_test, Y_train, Y_test, output_cols=[])
    #plot_residuals(model, data, X_train, X_test, Y_train, Y_test, output_cols=output_cols, remove_zero_chk=remove_zero_chk)
    plot_true_and_predicted(model, data, X_train, X_test, Y_train, Y_test, output_cols=output_cols, remove_zero_chk=remove_zero_chk,with_separate_plot=with_separate_plot,with_line_plot=with_line_plot)
    #plot_chk_vs_multiphase(model, data, X_train, X_test, Y_train, Y_test, input_cols=input_cols, output_cols=output_cols,
    #                       remove_zero_chk=remove_zero_chk)
    #plt.show()
def resample(X,step):

    N=len(X)
    new_X=np.zeros((N//step,))
    print(N//step)
    j=0
    for i in range(0,N-step,step):
        new_X[j]=np.mean(X[i:i+step])
        j+=1


    return new_X

def plot_history(model):
    output_names=model.output_tags.keys()
    #output_names=model.sort_names(output_names)
    step_size=10
    fig,ax=plt.subplots(1,1)
    N=len(model.history.history['val_GJOA_TOTAL_loss'])

    for key in output_names:
        X=resample(model.history.history['val_'+key+'_loss'],step_size)
        ax.plot(range(1, (N//step_size)*step_size+1,step_size), X,marker='.', label=KEY_MAP[model.output_tags[key][0]])


    #ax.plot(range(1,N+1),model.history.history['GJOA_TOTAL_loss'], label='Training set loss',color='blue', linestyle='--', marker='o')
    #print(model.history.history)
    ax.grid(which='major', linestyle='-')
    ax.set_axisbelow(True)
    ax.set_title('Test error',fontsize=FONTSIZETITLE)
    ax.set_xlabel('Time (epochs)',fontsize=FONTSIZEX)
    ax.set_ylabel('Loss',fontsize=FONTSIZEY)
    ax.legend(fontsize=FONTSIZELEGEND)
    ax.tick_params(axis='both', labelsize=TICKSIZE)
    ax.set_yticklabels([])
    #ax.xaxis.set_ticks(np.arange(0, N, step_size))
    #plt.show()
def tags_to_list(tags):

    tags_list=[]
    for key in tags:
        for tag in tags[key]:
            tags_list.append(tag)
    return tags_list


def count_n_well_inputs(input_tags):

    prev_tag='1_1'
    n_list=[]
    i=0
    for tag in input_tags:
        if tag.split('_')[0]==prev_tag.split('_')[0]:
            i+=1
        else:
            n_list.append(i)
            i=1
            prev_tag=tag
    return np.max(n_list)


def get_suplot_dim(N_PLOTS):
    sp_y = int(N_PLOTS / 2 + 0.5)
    if sp_y == 0:
        sp_y = 1
    sp_x = int(N_PLOTS / sp_y + 0.5)
    if sp_x == 0:
        sp_x = 1
    print(sp_y, sp_x, N_PLOTS)
    return sp_x,sp_y


def get_train_test_scatter_plot(ax,model,data,X,Y,x_tag,y_tag,color,type):
    ax.scatter(data.inverse_transform(X,'X')[x_tag],
               data.inverse_transform(Y,'Y')[y_tag], color=color[0],
               label='true - '+type)
    ax.scatter(data.inverse_transform(X,'X')[x_tag],
               data.inverse_transform(model.predict(X),'Y')[y_tag], color=color[1],
               label='pred - '+type)
    return ax

def get_scatter_plot(fig_par,model,data,X_train,X_test,Y_train,Y_test,x_tag,y_tag,remove_zero_chk=(False,'name',0)):
    ax=fig_par[-1]
    fig=fig_par[0]

    if remove_zero_chk[0] and remove_zero_chk[1]!='GJOA':
        ind_train = X_train[remove_zero_chk[1] + '_CHK'] > remove_zero_chk[-1]
        ind_test = X_test[remove_zero_chk[1]+ '_CHK'] > remove_zero_chk[-1]

        X_train=X_train[ind_train]
        X_test=X_test[ind_test]
        Y_train = Y_train[ind_train]
        Y_test = Y_test[ind_test]
    ax.grid(which='major',linestyle='-')
    ax.set_axisbelow(True)
    SCALE=1
    ax.scatter(data.inverse_transform(X_train,'X')[x_tag],
                data.inverse_transform(Y_train,'Y')[y_tag]/SCALE, color='blue',
                label=TRUE_TRAIN_LABEL)
    ax.scatter(data.inverse_transform(X_train,'X')[x_tag],
                data.inverse_transform(model.predict(X_train),'Y').set_index(Y_train.index)[y_tag]/SCALE, color='black',
                label=PRED_TRAIN_LABEL)

    ax.scatter(data.inverse_transform(X_test,'X')[x_tag],
                data.inverse_transform(Y_test,'Y')[y_tag]/SCALE, color='red', label=TRUE_TEST_LABEL)
    ax.scatter(data.inverse_transform(X_test,'X')[x_tag],
                data.inverse_transform(model.predict(X_test),'Y').set_index(Y_test.index)[y_tag]/SCALE, color='green',
                label=PRED_TEST_LABEL)

    #ax.legend(bbox_to_anchor=(0., 1., 1.01, .0), loc=3,
    #           ncol=2, mode="expand", borderaxespad=0.2)
    ax.legend(fontsize=FONTSIZELEGEND)
    fig.subplots_adjust(wspace=0.4, hspace=.3, top=0.95, bottom=0.06, left=0.05, right=0.99)
    fig.canvas.set_window_title(model.model_name)
    ax.tick_params(axis='both', which='major', labelsize=TICKSIZE)
    return ax

def get_line_plot(fig_par,model,data,X_train,X_test,Y_train,Y_test,x_tag,y_tag,remove_zero_chk=(False,'name',0)):
    ax=fig_par[-1]
    fig=fig_par[0]

    if remove_zero_chk[0] and remove_zero_chk[1]!='GJOA':
        ind_train = X_train[remove_zero_chk[1] + '_CHK'] > remove_zero_chk[-1]
        ind_test = X_test[remove_zero_chk[1]+ '_CHK'] > remove_zero_chk[-1]

        X_train=X_train[ind_train]
        X_test=X_test[ind_test]
        Y_train = Y_train[ind_train]
        Y_test = Y_test[ind_test]
    ax.grid(which='major', linestyle='-')
    ax.set_axisbelow(True)
    ax.plot(data.inverse_transform(Y_train,'Y')[y_tag], color='blue',label=TRUE_TRAIN_LABEL)
    ax.plot(data.inverse_transform(model.predict(X_train),'Y').set_index(Y_train.index)[y_tag], color='black',label=PRED_TRAIN_LABEL)

    ax.plot(data.inverse_transform(Y_test,'Y')[y_tag], color='red', label=TRUE_TEST_LABEL)
    ax.plot(data.inverse_transform(model.predict(X_test),'Y').set_index(Y_test.index)[y_tag], color='green',label= PRED_TEST_LABEL)

    #ax.legend(bbox_to_anchor=(0., 1., 1.01, .0), loc=3,
    #           ncol=2, mode="expand", borderaxespad=0.2)
    ax.legend(fontsize=FONTSIZELEGEND)
    # plt.legend()
    fig.subplots_adjust(wspace=0.13, hspace=.2, top=0.95, bottom=0.06, left=0.05, right=0.99)
    fig.canvas.set_window_title(model.model_name)
    ax.tick_params(axis='both', which='major', labelsize=TICKSIZE)
    return ax
def get_residual_plot(fig_par,model,data,X_train,X_test,Y_train,Y_test,x_tag,y_tag,remove_zero_chk=(False,'name',0)):
    ax = fig_par[-1]
    fig = fig_par[0]

    if remove_zero_chk[0] and remove_zero_chk[1]!='GJOA':

        ind_train = X_train[remove_zero_chk[1] + '_CHK'] > remove_zero_chk[-1]
        ind_test = X_test[remove_zero_chk[1]+ '_CHK'] > remove_zero_chk[-1]

        X_train=X_train[ind_train]
        X_test=X_test[ind_test]
        Y_train = Y_train[ind_train]
        Y_test = Y_test[ind_test]

    ax.scatter(Y_train.index,
                data.inverse_transform(Y_train,'Y')[y_tag] - data.inverse_transform(model.predict(X_train),'Y')[
                    y_tag].values,
                color='black',
                label='train')

    ax.scatter(Y_test.index,
                data.inverse_transform(Y_test,'Y')[y_tag] - data.inverse_transform(model.predict(X_test),'Y')[
                    y_tag].values,
                color='green',
                label='test')

    ax.legend(bbox_to_anchor=(0., 1., 1.01, .0), loc=3,
               ncol=2, mode="expand", borderaxespad=0.2)

    fig.subplots_adjust(wspace=0.08, hspace=.18, top=0.95, bottom=0.06, left=0.04, right=0.99)
    fig.canvas.set_window_title(model.model_name)
    fig.tick_params(axis='both', which='major', labelsize=20)
    return ax

def get_cumulative_performance_plot(cumperf,data_tag):


    N_PLOTS = len(cumperf.columns) - N_PLOT_SUB
    sp_y, sp_x = get_suplot_dim(N_PLOTS)

    fig, axes = plt.subplots(sp_y, sp_x)
    axes = axes.flatten()

    for i in range(len(cumperf.columns)):
        axes[i].plot(cumperf.index, cumperf[cumperf.columns[i]])
        axes[i].set_title(cumperf.columns[i])
        axes[i].set_xlabel('Deviation (%)')
        axes[i].set_ylabel('Cumulative (% of {} set sample points)'.format(data_tag))
        axes[i].tick_params(axis='both', which='major', labelsize=15)
    fig.suptitle('Cumulative performance of {} data'.format(data_tag))
    fig.subplots_adjust(wspace=0.17, hspace=.18, top=0.93, bottom=0.06, left=0.04, right=0.99)

    return fig, axes

def get_cumulative_flow_plot(cumperf,data_tag):


    N_PLOTS = len(cumperf.columns) - N_PLOT_SUB
    sp_y, sp_x = get_suplot_dim(N_PLOTS)

    fig, axes = plt.subplots(sp_y, sp_x)

    print(N_PLOTS)
    if N_PLOTS==1:

        axes.scatter(cumperf.index, cumperf[cumperf.columns])
        axes.set_title(cumperf.columns)
        axes.set_xlabel('Deviation (%)')
        axes.set_ylabel('Cumulative (% of {} set sample points)'.format(data_tag))
    else:
        axes = axes.flatten()
        for i in range(len(cumperf.columns)):
            axes[i].grid(which='major', linestyle='-')
            axes[i].set_axisbelow(True)
            axes[i].scatter(cumperf.index, cumperf[cumperf.columns[i]])
            axes[i].set_title(cumperf.columns[i])
            axes[i].set_xlabel('Deviation (%)')
            axes[i].set_ylabel('Cumulative (% of {} set sample points)'.format(data_tag))
            axes[i].tick_params(axis='both', which='major', labelsize=20)

    fig.suptitle('Cumulative performance of {} data'.format(data_tag))
    fig.subplots_adjust(wspace=0.17, hspace=.18, top=0.93, bottom=0.06, left=0.04, right=0.99)
    return fig, axes
def get_cumulative_deviation_plot_single(cumperf_train,cumperf_test,data_tag):

    def plot(cumperf,axes_wells,axes_tot, colors,ii, data_type):
        axes_wells.grid(which='major', linestyle='-')
        axes_wells.set_axisbelow(True)
        axes_tot.grid(which='major', linestyle='-')
        axes_tot.set_axisbelow(True)
        axes_tot.set_xlim([-0.1, 3])
        axes_wells.set_xlim([-0.1, 40])
        axes_wells.set_ylim([20, 105])
        axes_tot.set_ylim([20, 105])

        #axes_tot.yaxis.set_ticks(np.arange(0, 110, TICK_STEP_SIZE))
        #axes_wells[ii].yaxis.set_ticks(np.arange(0, 110, TICK_STEP_SIZE))

        for i in range(len(cumperf.columns) - 1):
            axes_wells.plot(cumperf.index, cumperf[cumperf_train.columns[i]], label=KEY_MAP[cumperf.columns[i]],
                          color=colors[i])
            axes_wells.set_title('{} data'.format(data_type),fontsize=FONTSIZETITLE)

        axes_wells.set_xlabel('Deviation (%)',fontsize=FONTSIZEX)
        axes_wells.set_ylabel('Cumulative \n (% of {} set sample points)'.format(data_type),fontsize=FONTSIZEY)
        axes_wells.tick_params(axis='both', which='major', labelsize=TICKSIZE)
        axes_tot.tick_params(axis='both', which='major', labelsize=TICKSIZE)

        axes_tot.plot(cumperf.index, cumperf[cumperf_train.columns[-1]],
                          label='{} data'.format(data_type))
        #axes_tot.set_title('{} data'.format(data_type),fontsize=FONTSIZETITLE)
        axes_tot.set_xlabel('Deviation (%)',fontsize=FONTSIZEX)
        axes_tot.set_ylabel('Cumulative (% of training and test set sample points)'.format(data_type),fontsize=FONTSIZEY)

        axes_wells.legend(fontsize=FONTSIZELEGEND)

    fig_wells, axes_wells = plt.subplots(1, 1)
    fig_tot, axes_tot = plt.subplots(1, 1)
    cmap = plt.get_cmap('Vega10')
    colors = [cmap(i) for i in np.linspace(0, 1, len(cumperf_train.columns))]

    #axes_wells = axes_wells.flatten()
    #axes_tot = axes_tot.flatten()

    fig_wells.canvas.set_window_title(data_tag)
    fig_tot.canvas.set_window_title(data_tag)




    #plot(cumperf_train,axes_wells,axes_tot,colors,0,'Training')
    plot(cumperf_test,axes_wells,axes_tot,colors,1,'Test')

    axes_tot.legend(fontsize=FONTSIZELEGEND)

    fig_wells.suptitle('Cumulative performance of wells',fontsize=30)
    fig_wells.subplots_adjust(wspace=0.17, hspace=.18, top=0.88, bottom=0.09, left=0.08, right=0.99)


    fig_tot.suptitle('Cumulative performance of total production', fontsize=30)
    fig_tot.subplots_adjust(wspace=0.17, hspace=.18, top=0.9, bottom=0.1, left=0.07, right=0.99)
    #fig_tot.set_xlim([0, 20])
    #return fig, axes

def get_cumulative_flow_plot_single(cumperf_train,cumperf_test,data_tag):

    def plot(cumperf,axes, ii, data_type):
        for i in range(len(cumperf.columns) - 1):
            axes[ii].plot(cumperf.index, cumperf[cumperf_train.columns[i]], label=KEY_MAP[cumperf.columns[i]],
                          color=colors[i])
        axes[ii].set_title('Well performance' + ' ({} data)'.format(data_type))
        axes[ii].set_xlabel('Time')
        axes[ii].set_ylabel('Absolute per sample deviation {}'.format(data_type))

        axes[ii + 1].plot(cumperf.index, cumperf[cumperf_train.columns[-1]],
                          label=KEY_MAP[cumperf.columns[-1]])
        axes[ii + 1].set_title(cumperf.columns[-1] + ' ({} data)'.format(data_type))
        axes[ii + 1].set_xlabel('Time')
        axes[ii + 1].set_ylabel('Absolute per sample deviation {}'.format(data_type))

        axes[ii].legend()

    fig, axes = plt.subplots(2, 2)
    cmap = plt.get_cmap('Vega10')
    colors = [cmap(i) for i in np.linspace(0, 1, 7)]
    axes = axes.flatten()

    plot(cumperf_train,axes,0,'Training')
    plot(cumperf_test,axes,2,'Test')


    fig.suptitle('Absolute per sample deviation plot')
    fig.subplots_adjust(wspace=0.17, hspace=.18, top=0.93, bottom=0.06, left=0.04, right=0.99)
    return fig, axes

def plot_cumulative_performance(model,data, X_train, X_test, Y_train, Y_test):
    cumperf_train = get_cumulative_deviation(model, data, X_train, Y_train)
    cumperf_test = get_cumulative_deviation(model, data, X_test, Y_test)



    get_cumulative_deviation_plot_single(cumperf_train, cumperf_test,model.model_name)

    cumperf_test = get_absolute_deviation(model, data, X_test, Y_test)
    cumperf_train = get_absolute_deviation(model, data, X_train, Y_train)

    get_cumulative_flow_plot(cumperf_train, 'Training')
    get_cumulative_flow_plot(cumperf_test, 'Test')
    #get_cumulative_performance_plot_single(cumperf_test,'Test')


def plot_residuals(model, data, X_train, X_test, Y_train, Y_test, output_cols=[],remove_zero_chk=False):
    if len(output_cols) == 0:
        output_cols = model.output_tag_ordered_list

    N_PLOTS = len(output_cols)-N_PLOT_SUB
    sp_y,sp_x=get_suplot_dim(N_PLOTS)

    zero_chk_param = (False, 'name', 0)
    i = 0
    #print(N_PLOTS)
    fig, axes = plt.subplots(sp_y, sp_x)
    if N_PLOTS > 1:
        axes = axes.flatten()
        #if N_PLOTS!=sp_x*sp_y:
        #    fig.delaxes(axes[-1])
    for output_tag in output_cols:
        if output_tag in OUTPUT_COLS_ON_SINGLE_PLOT:
            fig, ax = plt.subplots(1, 1)
        else:
            if N_PLOTS>1:
                ax = axes[i]
                i += 1
            else:
                ax=axes

        if remove_zero_chk:
            zero_chk_param = (True, output_tag.split('_')[0], model.get_chk_threshold())
            #print(model.get_chk_threshold())

        ax = get_residual_plot((fig, ax), model, data, X_train, X_test, Y_train, Y_test, x_tag='time', y_tag=output_tag,remove_zero_chk=zero_chk_param)

        ax.set_title(output_tag + '-' + 'Residuals (true-pred)')


def plot_true_and_predicted(model, data, X_train, X_test, Y_train, Y_test, output_cols=[],remove_zero_chk=False,with_line_plot=False,with_separate_plot=False):
    if len(output_cols) == 0:
        output_cols = model.output_tag_ordered_list

    N_sensors, tag_list=count_number_of_different_sensors_tags(output_cols)
    multi_output_col_list=split_col_list(output_cols,tag_list)
    N_PLOT_SUB=0
    for col in output_cols:
        if col in OUTPUT_COLS_ON_SINGLE_PLOT:
            N_PLOT_SUB=1
            break

    for output_cols in multi_output_col_list:
        zero_chk_param = (False, 'name', 0)
        N_PLOTS = len(output_cols)-N_PLOT_SUB

        #N_PLOTS=count_number_of_cols_that_ends_with(output_cols,'QGAS')
        sp_y, sp_x = get_suplot_dim(N_PLOTS)
        #N_PLOTS=sp_y=sp_x=1
        if with_separate_plot==False:
            i = 0
            fig_sub, axes = plt.subplots(sp_y, sp_x)
            #print(len(axes))
            if N_PLOTS>1:
                axes = axes.flatten()
                if N_PLOTS!=sp_y*sp_x:
                     fig_sub.delaxes(axes[-1])
        for output_tag in output_cols:
                if with_separate_plot:
                    N_PLOTS=1
                    sp_y, sp_x = get_suplot_dim(N_PLOTS)
                    fig_sub, axes = plt.subplots(sp_y, sp_x)
            #if ends_with(output_tag,'QGAS'):
                if output_tag in OUTPUT_COLS_ON_SINGLE_PLOT:
                    fig_single,ax=plt.subplots(1,1)
                    fig=fig_single
                else:
                    if N_PLOTS>1:
                        ax=axes[i]
                        fig=fig_sub
                        #plt.subplot(sp_x, sp_y, i)
                        i += 1
                    else:
                        fig=fig_sub
                        ax=axes
                fig_par = (fig, ax)
                if remove_zero_chk:
                    zero_chk_param = (True, output_tag.split('_')[0], model.get_chk_threshold())
                    #print(zero_chk_param)

                if with_line_plot:
                    ax=get_line_plot(fig_par, model, data, X_train, X_test, Y_train, Y_test, x_tag='time', y_tag=output_tag,remove_zero_chk=zero_chk_param)

                #ax.set_title(output_tag, fontsize=20)
                #ax.set_ylabel(output_tag.split('_')[-1], fontsize=20)
                #ax.set_xlabel('Time', fontsize=20)

                else:
                    ax = get_scatter_plot(fig_par, model, data, X_train, X_test, Y_train, Y_test, x_tag='time',
                                   y_tag=output_tag, remove_zero_chk=zero_chk_param)
                # plt.tight_layout()
                if output_tag.split('_')[-1] in ['QGAS','QOIL']:
                    #ax.set_title('Well '+output_tag.split('_')[0], fontsize=20)
                    ax.set_title(KEY_MAP[output_tag], fontsize=FONTSIZETITLE)
                else:
                    ax.set_title(output_tag.split('_')[0]+' '+output_tag.split('_')[1],fontsize=FONTSIZETITLE)
                ax.set_ylabel(TRUE_AND_PREDICTED_Y_LABEL,fontsize=FONTSIZEY)
                ax.set_xlabel('Sample number',fontsize=FONTSIZEX)


def plot_input_vs_output(model, data, X_train, X_test, Y_train, Y_test, input_cols=[], output_cols=[],remove_zero_chk=False):
    if len(input_cols) == 0:
        input_cols = tags_to_list(model.input_tags)
    if len(output_cols) == 0:
        output_cols = model.output_tag_ordered_list

    zero_chk_param = (False, 'name', 0)
    N_PLOTS = count_n_well_inputs(input_cols)
    sp_y, sp_x = get_suplot_dim(N_PLOTS)

    for output_tag in output_cols:
        fig,axes=plt.subplots(sp_y,sp_x)
        axes=axes.flatten()
        i = 0
        for input_tag in input_cols:
            if input_tag.split('_')[0] == output_tag.split('_')[0] and output_tag not in OUTPUT_COLS_ON_SINGLE_PLOT:

                if remove_zero_chk:
                    zero_chk_param=(True,input_tag.split('_')[0],model.get_chk_threshold())

                ax=axes[i]
                i += 1
                ax=get_scatter_plot((fig,ax), model, data, X_train, X_test, Y_train, Y_test, x_tag=input_tag, y_tag=output_tag,remove_zero_chk=zero_chk_param)

                ax.set_xlabel(input_tag)
                ax.set_ylabel(output_tag)
                ax.set_title(input_tag + ' vs ' + output_tag)


def plot_chk_vs_multiphase(model, data, X_train, X_test, Y_train, Y_test, input_cols=[], output_cols=[],remove_zero_chk=False):
    if len(input_cols) == 0:
        input_cols = tags_to_list(model.input_tags)
    if len(output_cols) == 0:
        output_cols = model.output_tag_ordered_list


    zero_chk_param=(False,'name',0)
    N_PLOTS = len(output_cols)-N_PLOT_SUB
    sp_y, sp_x = get_suplot_dim(N_PLOTS)

    fig, axes = plt.subplots(sp_y, sp_x)
    axes = axes.flatten()
    i = 0
    for output_tag in output_cols:
        for input_tag in input_cols:
            if input_tag.split('_')[0] == output_tag.split('_')[0] \
                    and output_tag not in OUTPUT_COLS_ON_SINGLE_PLOT and input_tag.split('_')[1] == 'CHK':

                if remove_zero_chk:
                    zero_chk_param=(True,input_tag.split('_')[0],model.get_chk_threshold())

                ax = axes[i]
                i += 1
                ax = get_scatter_plot((fig, ax), model, data, X_train, X_test, Y_train, Y_test, x_tag=input_tag,
                                      y_tag=output_tag,remove_zero_chk=zero_chk_param)

                ax.set_xlabel(input_tag)
                ax.set_ylabel(output_tag)
                ax.set_title(input_tag + ' vs ' + output_tag)
    if i==N_PLOTS and N_PLOTS>1:
        fig.delaxes(axes[-1])

def plot_true_and_predicted_with_input(model, data, X_train, X_test, Y_train, Y_test, output_cols=[]):
    if len(output_cols) == 0:
        output_cols = tags_to_list(model.output_tags)
    input_cols = tags_to_list(model.input_tags)

    sp_y = count_n_well_inputs(input_cols) + 1
    sp_x = 1
    #print(sp_y, sp_x, len(input_cols))

    zero_chk_param = (False, 'name', 0)

    i = 1

    for output_tag in output_cols:
        # plt.figure()
        i = 1
        tag = output_tag
        # ax.subplot(sp_y, sp_x, i)
        fig, axes = plt.subplots(1, sp_y, sharex=True)


        ax = axes[0]
        ax = get_scatter_plot((fig,ax), model, data, X_train, X_test, Y_train, Y_test, x_tag='time', y_tag=output_tag,
                              remove_zero_chk=zero_chk_param)
        ax.set_title(tag)
        ax.set_ylabel(tag)
        ax.set_xlabel('time')
        i = 1
        for input_tag in input_cols:
            if input_tag.split('_')[0] == output_tag.split('_')[0]:
                # plt.subplot(sp_y, sp_x, i)
                ax = axes[i]
                i += 1
                #print(input_tag)
                ax.scatter(data.inverse_transform(X_train,'X')['time'], data.inverse_transform(X_train,'X')[input_tag], color='black', label='true - train')
                ax.scatter(data.inverse_transform(X_test),'X'['time'], data.inverse_transform(X_test,'X')[input_tag], color='blue', label='true - test')
                ax.set_title(input_tag)
                ax.set_ylabel(input_tag)
                ax.set_xlabel('time')
                # plt.legend(['Y_true - train','Y_pred - train','Y_true - test', 'Y_pred - test'],pos)
                # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                #           ncol=2, mode="expand", borderaxespad=0., fontsize=10)
