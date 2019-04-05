import numpy as np
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense
from .BaseModel import BaseModel


class NN_LinearLinear(BaseModel):
    """2 Layer linear-linear neural network using Keras"""

    parametric = False
    bootlist = None

    def __init__(self, n_nodes=2, epochs=200, learning_rate=0.01, momentum=0.0, decay=0.0, nesterov=False, loss="binary_crossentropy", batch_size=None, verbose=0):
        self.n_nodes = n_nodes
        self.verbose = verbose
        self.epochs = epochs
        self.k = n_nodes
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = SGD(lr=learning_rate, momentum=momentum, decay=decay, nesterov=nesterov)

    def train(self, X, Y):
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

        # If batch-size is None:
        if self.batch_size is None:
            self.batch_size = min(200, len(X))

        # Ensure array and error check
        X, Y = self.input_check(X, Y)

        self.model = Sequential()
        self.model.add(Dense(self.n_nodes, activation="linear", input_dim=len(X.T)))
        self.model.add(Dense(1, activation="linear"))
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=["accuracy"])
        self.model.fit(X, Y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

        layer1_weight = self.model.layers[0].get_weights()[0]
        layer1_bias = self.model.layers[0].get_weights()[1]
        layer2_weight = self.model.layers[1].get_weights()[0]
        layer2_bias = self.model.layers[1].get_weights()[1]

        # Not sure about the naming scheme (trying to match PLS)
        self.model.x_loadings_ = layer1_weight
        self.model.x_scores_ = np.matmul(X, self.model.x_loadings_) + layer1_bias
        self.model.y_loadings_ = layer2_weight
        self.model.pctvar_ = np.ones((1, len(self.model.y_loadings_[0])))
        self.xcols_num = len(X.T)
        self.model.pctvar_ = sum(abs(self.model.x_scores_) ** 2) / sum(sum(abs(X) ** 2)) * 100
        y_pred_train = self.model.predict(X).flatten()

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

        self.model.y_loadings_ = layer2_weight
        self.model.y_loadings_ = self.model.y_loadings_.reshape(1, len(self.model.y_loadings_))
        self.model.x_loadings_ = layer1_weight
        self.model.x_scores_ = np.matmul(X, self.model.x_loadings_) + layer1_bias
        y_pred_test = self.model.predict(X).flatten()
        self.model.pctvar_ = sum(abs(self.model.x_scores_) ** 2) / sum(sum(abs(X) ** 2)) * 100
        return y_pred_test
