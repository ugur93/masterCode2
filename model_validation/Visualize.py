from matplotlib import pyplot as plt
import numpy as np
from .base import *

OUTPUT_COLS_ON_SINGLE_PLOT=['GJOA_QGAS','GJOA_TOTAL_QOIL','GJOA_TOTAL_QOIL_SUM','GJOA_OIL_QGASss']

N_PLOT_SUB=0

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
def visualize(model,data, X_train, X_test, Y_train ,Y_test, output_cols=[], input_cols=[]):

    remove_zero_chk=False
    plot_cumulative_performance(model,data, X_train, X_test, Y_train, Y_test)
    #plot_input_vs_output(model, data, X_train, X_test, Y_train, Y_test, input_cols=input_cols, output_cols=output_cols,
    #                     remove_zero_chk=remove_zero_chk)
    #plot_true_and_predicted_with_input(model, data, X_train, X_test, Y_train, Y_test, output_cols=[])
    #plot_residuals(model, data, X_train, X_test, Y_train, Y_test, output_cols=output_cols, remove_zero_chk=remove_zero_chk)
    plot_true_and_predicted(model, data, X_train, X_test, Y_train, Y_test, output_cols=output_cols, remove_zero_chk=remove_zero_chk)
    #plot_chk_vs_multiphase(model, data, X_train, X_test, Y_train, Y_test, input_cols=input_cols, output_cols=output_cols,
    #                       remove_zero_chk=remove_zero_chk)
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

    ax.scatter(data.inverse_transform(X_train,'X')[x_tag],
                data.inverse_transform(Y_train,'Y')[y_tag], color='blue',
                label='true - train')
    ax.scatter(data.inverse_transform(X_train,'X')[x_tag],
                data.inverse_transform(model.predict(X_train),'Y')[y_tag], color='black',
                label='pred - train')

    ax.scatter(data.inverse_transform(X_test,'X')[x_tag],
                data.inverse_transform(Y_test,'Y')[y_tag], color='red', label='true - test')
    ax.scatter(data.inverse_transform(X_test,'X')[x_tag],
                data.inverse_transform(model.predict(X_test),'Y')[y_tag], color='green',
                label= 'pred - test')

    ax.legend(bbox_to_anchor=(0., 1., 1.01, .0), loc=3,
               ncol=2, mode="expand", borderaxespad=0.2)
    # plt.legend()
    fig.subplots_adjust(wspace=0.13, hspace=.2, top=0.95, bottom=0.06, left=0.05, right=0.99)
    fig.canvas.set_window_title(model.model_name)
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
    fig.tick_params(axis='both', which='major', labelsize=10)
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
    fig.suptitle('Cumulative performance of {} data'.format(data_tag))
    fig.subplots_adjust(wspace=0.17, hspace=.18, top=0.93, bottom=0.06, left=0.04, right=0.99)
    return fig, axes
def get_cumulative_performance_plot_single(cumperf_train,cumperf_test,data_tag):

    def plot(cumperf,axes, ii, data_type):
        for i in range(len(cumperf.columns) - 1):
            axes[ii].plot(cumperf.index, cumperf[cumperf_train.columns[i]], label=cumperf.columns[i],
                          color=colors[i])
        axes[ii].set_title('Well performance' + ' ({} data)'.format(data_type))
        axes[ii].set_xlabel('Deviation (%)')
        axes[ii].set_ylabel('Cumulative \n (% of {} set sample points)'.format(data_type))

        axes[ii + 1].plot(cumperf.index, cumperf[cumperf_train.columns[-1]],
                          label=cumperf.columns[-1])
        axes[ii + 1].set_title(cumperf.columns[-1] + ' ({} data)'.format(data_type))
        axes[ii + 1].set_xlabel('Deviation (%)')
        axes[ii + 1].set_ylabel('Cumulative (% of {} set sample points)'.format(data_type))

        axes[ii].legend()

    fig, axes = plt.subplots(2, 2)
    cmap = plt.get_cmap('Vega10')
    colors = [cmap(i) for i in np.linspace(0, 1, 7)]
    axes = axes.flatten()

    fig.canvas.set_window_title(data_tag)


    plot(cumperf_train,axes,0,'Training')
    plot(cumperf_test,axes,2,'Test')


    fig.suptitle('Cumulative performance')
    fig.subplots_adjust(wspace=0.17, hspace=.18, top=0.93, bottom=0.06, left=0.04, right=0.99)
    return fig, axes

def get_cumulative_flow_plot_single(cumperf_train,cumperf_test,data_tag):

    def plot(cumperf,axes, ii, data_type):
        for i in range(len(cumperf.columns) - 1):
            axes[ii].plot(cumperf.index, cumperf[cumperf_train.columns[i]], label=cumperf.columns[i],
                          color=colors[i])
        axes[ii].set_title('Well performance' + ' ({} data)'.format(data_type))
        axes[ii].set_xlabel('Time')
        axes[ii].set_ylabel('Cumulative \n (% of {} set sample points)'.format(data_type))

        axes[ii + 1].plot(cumperf.index, cumperf[cumperf_train.columns[-1]],
                          label=cumperf.columns[-1])
        axes[ii + 1].set_title(cumperf.columns[-1] + ' ({} data)'.format(data_type))
        axes[ii + 1].set_xlabel('Time')
        axes[ii + 1].set_ylabel('Cumulative (% of {} set sample points)'.format(data_type))

        axes[ii].legend()

    fig, axes = plt.subplots(2, 2)
    cmap = plt.get_cmap('Vega10')
    colors = [cmap(i) for i in np.linspace(0, 1, 7)]
    axes = axes.flatten()

    plot(cumperf_train,axes,0,'Training')
    plot(cumperf_test,axes,2,'Test')


    fig.suptitle('Cumulative performance')
    fig.subplots_adjust(wspace=0.17, hspace=.18, top=0.93, bottom=0.06, left=0.04, right=0.99)
    return fig, axes

def plot_cumulative_performance(model,data, X_train, X_test, Y_train, Y_test):
    cumperf_train = get_cumulative_deviation(model, data, X_train, Y_train)
    cumperf_test = get_cumulative_deviation(model, data, X_test, Y_test)



    get_cumulative_performance_plot_single(cumperf_train, cumperf_test,model.model_name)

    cumperf_test = get_cumulative_flow(model, data, X_test, Y_test)
    cumperf_train = get_cumulative_flow(model, data, X_train, Y_train)

    get_cumulative_performance_plot_single(cumperf_train, cumperf_test, model.model_name)
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


def plot_true_and_predicted(model, data, X_train, X_test, Y_train, Y_test, output_cols=[],remove_zero_chk=False):
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

        i = 0
        fig_sub, axes = plt.subplots(sp_y, sp_x)
        #print(len(axes))
        if N_PLOTS>1:
            axes = axes.flatten()
            if N_PLOTS!=sp_y*sp_x:
                 fig_sub.delaxes(axes[-1])
        for output_tag in output_cols:
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


                ax=get_scatter_plot(fig_par, model, data, X_train, X_test, Y_train, Y_test, x_tag='time', y_tag=output_tag,remove_zero_chk=zero_chk_param)
                # plt.tight_layout()
                ax.set_title(output_tag,fontsize=20)
                ax.set_ylabel(output_tag.split('_')[-1],fontsize=20)
                ax.set_xlabel('Time',fontsize=20)


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
