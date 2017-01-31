


from .base import *




class NETTEST:



    def __init__(self):
        self.init_model()
        self.nb_epoch=10000
        self.batch_size=1000
        self.verbose=0
        self.callbacks=[EpochVerbose()]

    def init_model(self):
        main_input = Input(shape=(2,), dtype='float32')

        main_model=add_layers(main_input,3,25,0.0)

        out=Dense(1)(main_model)

        self.model = Model(input=main_input, output=out)
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self,X,Y):
        self.model.fit(X.values, Y.values, nb_epoch=self.nb_epoch, batch_size=self.batch_size, verbose=self.verbose,
                       callbacks=self.callbacks, shuffle=False)

    def predict(self,X):
        return self.model.predict(X.values)

    def evaluate(self,X_train,X_test,Y_train,Y_test):

        #cols=output_tags_to_list(self.output_tags)

        cols2='GJOA_QGAS'
        score_test_MSE = metrics.mean_squared_error(Y_test, self.predict(X_test), multioutput='raw_values')
        score_train_MSE = metrics.mean_squared_error(Y_train, self.predict(X_train), multioutput='raw_values')
        score_test_r2 = metrics.r2_score(Y_test, self.predict(X_test), multioutput='raw_values')
        score_train_r2 = metrics.r2_score(Y_train, self.predict(X_train), multioutput='raw_values')

        return score_train_MSE,score_test_MSE,score_train_r2,score_test_r2