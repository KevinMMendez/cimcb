import numpy as np
from keras.callbacks import Callback
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from keras import backend as K
from .BaseModel import BaseModel
from ..utils import YpredCallback


class NN_LinearSigmoid(BaseModel):
    """2 Layer linear-logistic neural network using Keras"""

    parametric = True
    bootlist = None

    def __init__(self, n_neurons=2, epochs=200, learning_rate=0.01, momentum=0.0, decay=0.0, nesterov=False, loss="binary_crossentropy", batch_size=None, verbose=0):
        self.n_neurons = n_neurons
        self.verbose = verbose
        self.n_epochs = epochs
        self.k = n_neurons
        self.batch_size = batch_size
        self.loss = loss
        self.decay = decay
        self.nesterov = nesterov
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.optimizer = SGD(lr=learning_rate, momentum=momentum, decay=decay, nesterov=nesterov)

        self.__name__ = 'cimcb.model.NN_LinearSigmoid'
        self.__params__ = {'n_neurons': n_neurons, 'epochs': epochs, 'learning_rate': learning_rate, 'momentum': momentum, 'decay': decay, 'nesterov': nesterov, 'loss': loss, 'batch_size': batch_size, 'verbose': verbose}

    def set_params(self, params):
        self.__init__(**params)

    def train(self, X, Y, epoch_ypred=False, epoch_xtest=None):
        """ Fit the neural network model, save additional stats (as attributes) and return Y predicted values.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Predictor variables, where n_samples is the number of samples and n_features is the number of predictors.

        Y : array-like, shape = [n_samples, 1]
            Response variables, where n_samples is the number of samples.

        Returns
        -------
        y_pred_train : array-like, shape = [n_samples, 1]
            Predicted y score for samples.
        """

        # # If using Keras, set tf to 1 core
        # config = K.tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8, allow_soft_placement=True)
        # session = tf.Session(config=config)
        # K.set_session(session)

        # If batch-size is None:
        if self.batch_size is None:
            self.batch_size = len(X)

        self.model = Sequential()
        self.model.add(Dense(self.n_neurons, activation="linear", input_dim=len(X.T)))
        self.model.add(Dense(1, activation="sigmoid"))
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=["accuracy"])

        # If epoch_ypred is True, calculate ypred for each epoch
        if epoch_ypred is True:
            self.epoch = YpredCallback(self.model, X, epoch_xtest)
        else:
            self.epoch = Callback()

        # Fit
        self.model.fit(X, Y, epochs=self.n_epochs, batch_size=self.batch_size, verbose=self.verbose, callbacks=[self.epoch])

        layer1_weight = self.model.layers[0].get_weights()[0]
        layer1_bias = self.model.layers[0].get_weights()[1]
        layer2_weight = self.model.layers[1].get_weights()[0]
        layer2_bias = self.model.layers[1].get_weights()[1]

        # Not sure about the naming scheme (trying to match PLS)
        self.model.x_loadings_ = layer1_weight
        self.model.x_scores_ = np.matmul(X, self.model.x_loadings_) + layer1_bias
        self.model.x_scores_alt = self.model.x_scores_
        self.model.y_loadings_ = layer2_weight
        self.model.y_scores = np.matmul(self.model.x_scores_alt, self.model.y_loadings_) + layer2_bias

        # self.model.pctvar_ = np.ones((1, len(self.model.y_loadings_[0])))
        # explained variance = y_pred where other nodes are 0, sum((y_pred - y true) ** 2) / sum(y_true ** 2)

        self.xcols_num = len(X.T)
        ytrue = self.model.predict(X)
        y_pred_train = self.model.predict(X).flatten()

        pctvar_ = []
        for i in range(len(self.model.x_scores_alt.T)):
            x_scores_i = deepcopy(self.model.x_scores_alt)
            for j in range(len(self.model.x_scores_alt.T)):
                if j != i:
                    for k in range(len(self.model.x_scores_alt)):
                        x_scores_i[k, i] = 0
            ypred_i = logistic.cdf(np.matmul(x_scores_i, self.model.y_loadings_) + layer2_bias)
            pctvar_i = np.sum((ypred_i - ytrue) ** 2) / np.sum(ytrue) * 100
            pctvar_.append(pctvar_i)
        self.model.pctvar_ = np.array(pctvar_)

        # Test new x_scores_
        for i in range(len(layer2_weight)):
            if layer2_weight[i] < 0:
                self.model.x_scores_[:, i] = -self.model.x_scores_[:, i]
                self.model.x_scores_alt[:, i] = -self.model.x_scores_alt[:, i]
                self.model.y_loadings_[i] = -self.model.y_loadings_[i]
                self.model.x_loadings_[i] = -self.model.x_loadings_[i]
        #self.model.pctvar_ = sum(abs(self.model.x_loadings_) ** 2) / sum(sum(abs(X) ** 2)) * 100

        # Resort by pctvar
        order = np.argsort(self.model.pctvar_)[::-1]
        x_scores_ = deepcopy(self.model.x_scores_)
        y_load_ = deepcopy(self.model.y_loadings_)
        x_load_ = deepcopy(self.model.x_loadings_)
        x_scores_alt = deepcopy(self.model.x_scores_alt)
        for i in range(len(order)):
            self.model.x_scores_[:, i] = x_scores_[:, order[i]]
            self.model.x_scores_alt[:, i] = x_scores_alt[:, order[i]]
            self.model.y_loadings_[:, 0][i] = y_load_[:, 0][order[i]]
            self.model.x_loadings_[:, i] = x_load_[:, order[i]]
        self.model.y_loadings_ = self.model.y_loadings_.reshape(1, len(self.model.y_loadings_))
        self.model.pctvar_ = np.sort(self.model.pctvar_)[::-1]
        self.model.coef_ = connectionweight(self.model.x_loadings_, self.model.y_loadings_.flatten())
        self.model.vip_ = garson(self.model.x_loadings_, self.model.y_loadings_.flatten())

        self.model.x_scores_alt = self.model.x_scores_

        # Storing X, Y, and Y_pred
        self.Y_pred = y_pred_train
        self.X = X
        self.Y = Y
        return y_pred_train

    def test(self, X, Y=None):
        """Calculate and return Y predicted value.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Test variables, where n_samples is the number of samples and n_features is the number of predictors.

        Returns
        -------
        y_pred_test : array-like, shape = [n_samples, 1]
            Predicted y score for samples.
        """

        layer1_weight = self.model.layers[0].get_weights()[0]
        layer1_bias = self.model.layers[0].get_weights()[1]
        layer2_weight = self.model.layers[1].get_weights()[0]
        layer2_bias = self.model.layers[1].get_weights()[1]

        self.model.x_scores_ = np.matmul(X, layer1_weight) + layer1_bias
        self.model.x_scores_alt = logistic.cdf(self.model.x_scores_)
        for i in range(len(layer2_weight)):
            if layer2_weight[i] < 0:
                self.model.x_scores_[:, i] = -self.model.x_scores_[:, i]
                self.model.x_scores_alt[:, i] = -self.model.x_scores_alt[:, i]

        order = np.argsort(self.model.pctvar_)[::-1]
        x_scores_ = deepcopy(self.model.x_scores_)
        x_scores_alt = deepcopy(self.model.x_scores_alt)
        for i in range(len(order)):
            self.model.x_scores_[:, i] = x_scores_[:, order[i]]
            self.model.x_scores_alt[:, i] = x_scores_alt[:, order[i]]

        self.model.x_scores_alt = self.model.x_scores_

        y_pred_test = self.model.predict(X).flatten()

        return y_pred_test
