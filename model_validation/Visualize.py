from matplotlib import pyplot as plt
import numpy as np


OUTPUT_COLS_ON_SINGLE_PLOT=['GJOA_QGAS','GJOA_TOTAL_QOIL','GJOA_TOTAL_QOIL_SUM','GJOA_OIL_QGASss']

N_PLOT_SUB=0


def visualize(model,data, X_train, X_test, Y_train ,Y_test, output_cols=[], input_cols=[]):

    remove_zero_chk=True

    #plot_input_vs_output(model, data, X_train, X_test, Y_train, Y_test, input_cols=input_cols, output_cols=output_cols,
    #                     remove_zero_chk=remove_zero_chk)
    #plot_true_and_predicted_with_input(model, data, X_train, X_test, Y_train, Y_test, output_cols=[])
    plot_residuals(model, data, X_train, X_test, Y_train, Y_test, output_cols=output_cols, remove_zero_chk=remove_zero_chk)
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
    fig.subplots_adjust(wspace=0.08, hspace=.18, top=0.95, bottom=0.06, left=0.04, right=0.99)
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
    return ax


def plot_residuals(model, data, X_train, X_test, Y_train, Y_test, output_cols=[],remove_zero_chk=False):
    if len(output_cols) == 0:
        output_cols = model.output_tag_ordered_list

    N_PLOTS = len(output_cols)-N_PLOT_SUB
    sp_y,sp_x=get_suplot_dim(N_PLOTS)

    zero_chk_param = (False, 'name', 0)
    i = 0
    print(N_PLOTS)
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
            print(model.get_chk_threshold())

        ax = get_residual_plot((fig, ax), model, data, X_train, X_test, Y_train, Y_test, x_tag='time', y_tag=output_tag,remove_zero_chk=zero_chk_param)

        ax.set_title(output_tag + '-' + 'Residuals (true-pred)')


def plot_true_and_predicted(model, data, X_train, X_test, Y_train, Y_test, output_cols=[],remove_zero_chk=False):
    if len(output_cols) == 0:
        output_cols = model.output_tag_ordered_list

    zero_chk_param = (False, 'name', 0)
    N_PLOTS = len(output_cols)-N_PLOT_SUB
    sp_y, sp_x = get_suplot_dim(N_PLOTS)

    i = 0
    fig_sub, axes = plt.subplots(sp_y, sp_x)
    #print(len(axes))
    if N_PLOTS>1:
        axes = axes.flatten()
        if N_PLOTS!=sp_y*sp_x:
             fig_sub.delaxes(axes[-1])
    for output_tag in output_cols:
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
            print(zero_chk_param)


        ax=get_scatter_plot(fig_par, model, data, X_train, X_test, Y_train, Y_test, x_tag='time', y_tag=output_tag,remove_zero_chk=zero_chk_param)
        # plt.tight_layout()
        ax.set_title(output_tag)


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
    print(sp_y, sp_x, len(input_cols))

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
