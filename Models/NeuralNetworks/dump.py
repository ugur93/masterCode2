def visualize(self, data, X_train, X_test, Y_train, Y_test, output_cols=[], input_cols=[]):
    self.X_SCALE = 100
    # self.plot_scatter_input_output(data,X_train, X_test, Y_train, Y_test, input_cols=input_cols,output_cols=output_cols)
    # self.plot_scatter_chk_well(data,X_train, X_test, Y_train, Y_test, input_cols=input_cols, output_cols=output_cols)
    self.plot_residuals(data, X_train, X_test, Y_train, Y_test, output_cols)
    self.plot_true_and_predicted(data, X_train, X_test, Y_train, Y_test, output_cols)
    # self.plot_true_and_predicted_with_input(data,X_train, X_test, Y_train, Y_test, output_cols=output_cols)
    plt.show()


def plot_residuals(self, data, X_train, X_test, Y_train, Y_test, output_cols=[], remove_zeros=False):
    if len(output_cols) == 0:
        output_cols = self.output_tag_ordered_list
    N_PLOTS = len(output_cols)
    sp_y = int(N_PLOTS / 2 + 0.5)
    if sp_y == 0:
        sp_y = 1
    sp_x = int(N_PLOTS / sp_y + 0.5)
    if sp_x == 0:
        sp_x = 1
    print(sp_y, sp_x, len(output_cols))

    i = 1
    plt.figure()
    for output_tag in output_cols:
        if output_tag == 'GJOA_QGAS' or output_tag == 'GJOA_TOTAL_QOIL_SUM' or output_tag == 'GJOA_TOTAL_QOIL':
            plt.figure()
        else:
            plt.subplot(sp_x, sp_y, i)
            i += 1
        # print(self.predict(X_test, output_tag))
        plt.scatter(Y_train.index,
                    data.inverse_transform(Y_train)[output_tag] - data.inverse_transform(self.predict(X_train))[
                        output_tag].values,
                    color='black',
                    label='train')

        plt.scatter(Y_test.index,
                    data.inverse_transform(Y_test)[output_tag] - data.inverse_transform(self.predict(X_test))[
                        output_tag].values,
                    color='green',
                    label='test')
        plt.title(output_tag + '-' + 'Residuals (true-pred)')
        # plt.legend(['Y_true - train','Y_pred - train','Y_true - test', 'Y_pred - test'],pos)
        plt.legend(bbox_to_anchor=(0., 1., 1.01, .0), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.2)
        # plt.legend()
        plt.subplots_adjust(wspace=0.08, hspace=.45, top=0.95, bottom=0.06, left=0.04, right=0.99)
        # plt.tight_layout()


def plot_true_and_predicted(self, data, X_train, X_test, Y_train, Y_test, output_cols=[]):
    if len(output_cols) == 0:
        output_cols = self.output_tag_ordered_list

    N_PLOTS = len(output_cols)
    sp_y = int(N_PLOTS / 2 + 0.5)
    if sp_y == 0:
        sp_y = 1
    sp_x = int(N_PLOTS / sp_y + 0.5)
    if sp_x == 0:
        sp_x = 1
    print(sp_y, sp_x, len(output_cols))

    i = 1
    plt.figure()
    for output_tag in output_cols:
        if output_tag == 'GJOA_QGAS' or output_tag == 'GJOA_TOTAL_QOIL_SUM' or output_tag == 'GJOA_TOTAL_QOIL':
            plt.figure()
        else:
            plt.subplot(sp_x, sp_y, i)
            i += 1
        plt.scatter(Y_train.index, data.inverse_transform(Y_train)[output_tag], color='blue', label='true - train')
        plt.scatter(Y_train.index, data.inverse_transform(self.predict(X_train))[output_tag], color='black',
                    label='pred - train')

        plt.scatter(Y_test.index, data.inverse_transform(Y_test)[output_tag], color='red', label='true - test')
        plt.scatter(Y_test.index, data.inverse_transform(self.predict(X_test))[output_tag], color='green',
                    label='pred - test')
        plt.title(output_tag)
        # plt.legend(['Y_true - train','Y_pred - train','Y_true - test', 'Y_pred - test'],pos)
        plt.legend(bbox_to_anchor=(0., 1., 1.01, .0), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.2)
        # plt.legend()
        plt.subplots_adjust(wspace=0.08, hspace=.2, top=0.94, bottom=0.06, left=0.04, right=0.99)
        # plt.tight_layout()


def plot_scatter_input_output(self, data, X_train, X_test, Y_train, Y_test, input_cols=[], output_cols=[]):
    if len(input_cols) == 0:
        input_cols = tags_to_list(self.input_tags)
    if len(output_cols) == 0:
        output_cols = self.output_tag_ordered_list

    i = 1
    N_plots = count_n_well_inputs(input_cols)
    sp_y = int(N_plots / 2 + 0.5)
    if sp_y == 0:
        sp_y = 1
    sp_x = int(N_plots / sp_y + 0.5)
    if sp_x == 0:
        sp_x = 1
    print(sp_y, sp_x)
    for output_tag in output_cols:
        plt.figure()
        i = 1
        for input_tag in input_cols:
            if input_tag.split('_')[0] == output_tag.split('_')[
                0] and output_tag != 'GJOA_QGAS' and output_tag != 'GJOA_TOTAL_QOIL_SUM' and output_tag != 'GJOA_TOTAL_QOIL':
                plt.subplot(sp_y, sp_x, i)
                i += 1
                plt.scatter(data.inverse_transform(X_train)[input_tag],
                            data.inverse_transform(Y_train)[output_tag], color='blue',
                            label=output_tag + '_true - train')
                plt.scatter(data.inverse_transform(X_train)[input_tag],
                            data.inverse_transform(self.predict(X_train))[output_tag], color='black',
                            label=output_tag + '_pred - train')

                plt.scatter(data.inverse_transform(X_test)[input_tag],
                            data.inverse_transform(Y_test)[output_tag], color='red',
                            label=output_tag + '_true - test')
                plt.scatter(data.inverse_transform(X_test)[input_tag],
                            data.inverse_transform(self.predict(X_test))[output_tag], color='green',
                            label=output_tag + '_pred - test')

                plt.xlabel(input_tag)
                plt.ylabel(output_tag)
                plt.title(input_tag + ' vs ' + output_tag)

                plt.legend(bbox_to_anchor=(0., 1., 1.01, .0), loc=3,
                           ncol=2, mode="expand", borderaxespad=0.2)
                # plt.legend()
                plt.subplots_adjust(wspace=0.08, hspace=.45, top=0.95, bottom=0.06, left=0.04, right=0.99)
                # plt.tight_layout()


def plot_scatter_chk_well(self, data, X_train, X_test, Y_train, Y_test, input_cols=[], output_cols=[]):
    if len(input_cols) == 0:
        input_cols = tags_to_list(self.input_tags)
    if len(output_cols) == 0:
        output_cols = self.output_tag_ordered_list

    i = 1
    N_plots = len(output_cols)  # count_n_well_inputs(input_cols)
    sp_y = int(N_plots / 2 + 0.5)
    if sp_y == 0:
        sp_y = 1
    sp_x = int(N_plots / sp_y + 0.5)
    if sp_x == 0:
        sp_x = 1
    print(sp_y, sp_x, N_plots)
    plt.figure()
    i = 1
    for output_tag in output_cols:
        for input_tag in input_cols:
            if input_tag.split('_')[0] == output_tag.split('_')[0] and output_tag != 'GJOA_QGAS' and \
                            input_tag.split('_')[1] == 'CHK':
                plt.subplot(sp_x, sp_y, i)
                i += 1
                plt.scatter(data.inverse_transform(X_train)[input_tag], data.inverse_transform(Y_train)[output_tag],
                            color='blue', label='true - train')
                plt.scatter(data.inverse_transform(X_train)[input_tag],
                            data.inverse_transform(self.predict(X_train))[output_tag], color='black',
                            label='pred - train')

                plt.scatter(data.inverse_transform(X_test)[input_tag], data.inverse_transform(Y_test)[output_tag],
                            color='red', label='true - test')
                plt.scatter(data.inverse_transform(X_test)[input_tag],
                            data.inverse_transform(self.predict(X_test))[output_tag], color='green',
                            label='pred - test')

                plt.xlabel(input_tag)
                plt.ylabel(output_tag)
                plt.title(input_tag + ' vs ' + output_tag)

                plt.legend(bbox_to_anchor=(0., 1., 1.01, .0), loc=3,
                           ncol=2, mode="expand", borderaxespad=0.2)
                # plt.legend()
                plt.subplots_adjust(wspace=0.08, hspace=.45, top=0.95, bottom=0.06, left=0.04, right=0.99)
                # plt.tight_layout()


def plot_true_and_predicted_with_input(self, data, X_train, X_test, Y_train, Y_test, output_cols=[]):
    if len(output_cols) == 0:
        output_cols = tags_to_list(self.output_tags)
    input_cols = tags_to_list(self.input_tags)

    sp_y = count_n_well_inputs(input_cols) + 1
    sp_x = 1
    print(sp_y, sp_x, len(input_cols))

    i = 1

    for output_tag in output_cols:
        # plt.figure()
        i = 1
        tag = output_tag
        # ax.subplot(sp_y, sp_x, i)
        _, axes = plt.subplots(sp_y, 1, sharex=True)
        ax = axes[0]
        ax.scatter(Y_train.index, self.SCALE * Y_train[tag], color='blue', label=tag + '_true - train')
        ax.scatter(Y_train.index, self.SCALE * self.predict(X_train, tag), color='black',
                   label=tag + '_pred - train')
        ax.scatter(Y_test.index, self.SCALE * Y_test[tag], color='red', label=tag + '_true - test')
        ax.scatter(Y_test.index, self.SCALE * self.predict(X_test, tag), color='green', label=tag + '_pred - test')
        ax.set_title(tag)
        ax.set_ylabel(tag)
        ax.set_xlabel('time')
        i = 1
        for input_tag in input_cols:
            if input_tag.split('_')[0] == output_tag.split('_')[0]:
                # plt.subplot(sp_y, sp_x, i)
                tag = input_tag
                ax = axes[i]
                i += 1
                ax.scatter(Y_train.index, self.SCALE * X_train[tag], color='black', label=tag + '_true - train')
                ax.scatter(Y_test.index, self.SCALE * X_test[tag], color='blue', label=tag + '_true - test')
                ax.set_title(tag)
                ax.set_ylabel(tag)
                ax.set_xlabel('time')
                # plt.legend(['Y_true - train','Y_pred - train','Y_true - test', 'Y_pred - test'],pos)
                # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                #           ncol=2, mode="expand", borderaxespad=0., fontsize=10)


def plot_true_and_predicted_zeros(self, X_train, X_test, Y_train, Y_test, output_cols=[]):
    if len(output_cols) == 0:
        output_cols = self.output_tag_ordered_list

    N_PLOTS = len(output_cols)
    sp_y = int(N_PLOTS / 2 + 0.5)
    if sp_y == 0:
        sp_y = 1
    sp_x = int(N_PLOTS / sp_y + 0.5)
    if sp_x == 0:
        sp_x = 1
    print(sp_y, sp_x, len(output_cols))

    i = 1
    plt.figure()
    for output_tag in output_cols:
        if output_tag == 'GJOA_QGAS' or output_tag == 'GJOA_TOTAL_QOIL_SUM' or output_tag == 'GJOA_TOTAL_QOIL':
            plt.figure()
        else:
            plt.subplot(sp_x, sp_y, i)
            i += 1
        well = output_tag.split('_')[0]
        ind_train = X_train[well + '_CHK'] > 0.05
        ind_test = X_test[well + '_CHK'] > 0.05
        plt.scatter(Y_train.index[ind_train], self.SCALE * Y_train[output_tag][ind_train], color='blue',
                    label='true - train')
        plt.scatter(Y_train.index[ind_train], self.SCALE * self.predict(X_train[ind_train], output_tag),
                    color='black', label='pred - train')

        plt.scatter(Y_test.index[ind_test], self.SCALE * Y_test[output_tag][ind_test], color='red',
                    label='true - val')
        plt.scatter(Y_test.index[ind_test], self.SCALE * self.predict(X_test[ind_test], output_tag), color='green',
                    label='pred - val')
        plt.title(output_tag)
        # plt.legend(['Y_true - train','Y_pred - train','Y_true - test', 'Y_pred - test'],pos)
        plt.legend(bbox_to_anchor=(0., 1.03, 1., .112), loc=3,
                   ncol=2, mode="expand", borderaxespad=0., fontsize=10)


def plot_residuals_zeros(self, X_train, X_test, Y_train, Y_test, output_cols=[], remove_zeros=False):
    if len(output_cols) == 0:
        output_cols = self.output_tag_ordered_list

    sp_y = int((len(output_cols) - 1) / 2 + 0.5)
    if sp_y == 0:
        sp_y = 1
    sp_x = int((len(output_cols)) / sp_y + 0.5)
    if sp_x == 0:
        sp_x = 1
    print(sp_y, sp_x, len(output_cols))

    i = 1
    plt.figure()
    for output_tag in output_cols:
        if output_tag == 'GJOA_QGAS' or output_tag == 'GJOA_TOTAL_QOIL_SUM' or output_tag == 'GJOA_TOTAL_QOIL':
            plt.figure()
        else:
            plt.subplot(sp_x, sp_y, i)
            i += 1
        well = output_tag.split('_')[0]
        ind_train = X_train[well + '_CHK'] > 0.05
        ind_test = X_test[well + '_CHK'] > 0.05
        plt.scatter(Y_train.index[ind_train],
                    self.SCALE * (Y_train[output_tag] - self.predict(X_train, output_tag))[ind_train],
                    color='black',
                    label='train')

        plt.scatter(Y_test.index[ind_test],
                    self.SCALE * (Y_test[output_tag] - self.predict(X_test, output_tag))[ind_test],
                    color='green',
                    label='val')
        plt.title(output_tag + '-' + 'Residuals (true-pred)')
        # plt.legend(['Y_true - train','Y_pred - train','Y_true - test', 'Y_pred - test'],pos)
        plt.legend(bbox_to_anchor=(0., 1.04, 1., .102), loc=9,
                   ncol=2, mode="expand", borderaxespad=0., fontsize=10)


def plot_scatter_input_output_zeros(self, X_train, X_test, Y_train, Y_test, input_cols=[], output_cols=[]):
    if len(input_cols) == 0:
        input_cols = tags_to_list(self.input_tags)
    if len(output_cols) == 0:
        output_cols = self.output_tag_ordered_list

    i = 1
    N_plots = count_n_well_inputs(input_cols)
    sp_y = int(N_plots / 2 + 0.5)
    if sp_y == 0:
        sp_y = 1
    sp_x = int(N_plots / sp_y + 0.5)
    if sp_x == 0:
        sp_x = 1
    print(sp_y, sp_x)
    for output_tag in output_cols:
        plt.figure()
        i = 1
        for input_tag in input_cols:
            if input_tag.split('_')[0] == output_tag.split('_')[
                0] and output_tag != 'GJOA_QGAS' and output_tag != 'GJOA_TOTAL_QOIL_SUM' and output_tag != 'GJOA_TOTAL_QOIL':
                plt.subplot(sp_y, sp_x, i)
                well = output_tag.split('_')[0]
                ind_train = X_train[well + '_CHK'] > 0.05
                ind_test = X_test[well + '_CHK'] > 0.05
                i += 1
                plt.scatter(X_train[input_tag][ind_train], self.SCALE * Y_train[output_tag][ind_train],
                            color='blue',
                            label=output_tag + '_true - train')
                plt.scatter(X_train[input_tag][ind_train],
                            self.SCALE * self.predict(X_train, output_tag)[ind_train],
                            color='black', label=output_tag + '_pred - train')

                plt.scatter(X_test[input_tag][ind_test], self.SCALE * Y_test[output_tag][ind_test], color='red',
                            label=output_tag + '_true - test')
                plt.scatter(X_test[input_tag][ind_test], self.SCALE * self.predict(X_test, output_tag)[ind_test],
                            color='green',
                            label=output_tag + '_pred - test')

                plt.xlabel(input_tag)
                plt.ylabel(output_tag)
                plt.title(input_tag + ' vs ' + output_tag)

                plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                           ncol=2, mode="expand", borderaxespad=0., fontsize=10)
