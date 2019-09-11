import numpy as np
from keras.callbacks import Callback
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from scipy.stats import logistic
from copy import deepcopy, copy
from sklearn.metrics import r2_score
from keras import backend as K
from sklearn.cross_decomposition import PLSRegression
from copy import deepcopy
import torch
from torch.autograd import Variable
from keras.constraints import max_norm, non_neg, min_max_norm, unit_norm
from .BaseModel import BaseModel
from ..utils import YpredCallback


class SWNN_LinearLinear(BaseModel):
    """2 Layer linear-logistic neural network using Keras"""

    parametric = True
    bootlist = ["model.vip_", "model.coef_", "model.x_loadings_", "model.x_scores_", "Y_pred"]  # list of metrics to bootstrap

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
        self.optimizer = 1
        #self.model = PLSRegression()

        self.__name__ = 'cimcb.model.SWNN_LinearLinear'
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

        n_neurons = self.n_neurons
        epochs = self.n_epochs
        x = X
        y = Y
        learning_rate = self.learning_rate
        momentum = self.momentum
        decay = self.decay

        # N is batch size; D_in is input dimension;
        # H is hidden dimension; D_out is output dimension.
        N = len(y)
        D_in = len(x.T)
        H = n_neurons
        D_out = 1

        # Create random Tensors to hold inputs and outputs
        y = np.expand_dims(y, axis=-1)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        x = Variable(x).float()
        y = Variable(y).float()

        # Use the nn package to define our model and loss function.
        class NNet(torch.nn.Module):
            def __init__(self, D_in, H, D_out):
                """
                In the constructor we instantiate two nn.Linear modules and assign them as
                member variables.
                """
                super(NNet, self).__init__()
                self.layer1 = torch.nn.Linear(D_in, H)
                self.layer2 = torch.nn.Linear(H, D_out, bias=False)

            def forward(self, x):
                """
                In the forward function we accept a Tensor of input data and we must return
                a Tensor of output data. We can use Modules defined in the constructor as
                well as arbitrary operators on Tensors.
                """
                h_relu = self.layer1(x)
                y_pred = self.layer2(h_relu)
                return y_pred

            def eval(self, x):
                val = self(x)
                return val

        model = NNet(D_in, H, D_out)
        loss_fn = torch.nn.MSELoss(reduction='sum')

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=learning_rate,
                                    momentum=momentum,
                                    dampening=0,
                                    weight_decay=decay,
                                    nesterov=False)

        # Set Weights/Bias except first to zero
        weight_dict = model.state_dict()
        init_weight_dict = deepcopy(weight_dict)
        init_weight_dict['layer2.weight'] = torch.ones(1, n_neurons)
        new_weight_dict = deepcopy(weight_dict)
        new_weight_dict['layer1.weight'][:][:] = 0
        new_weight_dict['layer1.bias'][:] = 0
        new_weight_dict['layer2.weight'][0][:] = 0
        model.load_state_dict(new_weight_dict)

        for p in range(n_neurons):
            # Initalise weight
            weight_dict = model.state_dict()
            new_weight_dict = deepcopy(weight_dict)
            new_weight_dict['layer1.weight'][:][p] = init_weight_dict['layer1.weight'][:][p]
            new_weight_dict['layer1.bias'][p] = init_weight_dict['layer1.bias'][p]
            new_weight_dict['layer2.weight'][0][p] = init_weight_dict['layer2.weight'][0][p]
            model.load_state_dict(new_weight_dict)

            for t in range(epochs):
                # Forward pass: compute predicted y by passing x to the model.
                y_pred = model(x)

                # Compute and print loss.
                loss = loss_fn(y_pred, y)
                # if t % 100 == 99:
                #     y_pred_round = (y_pred>0.5).float()
                #     correct = (y_pred_round == y).float().sum()
                #     acc = correct / y.shape[0]
                #     print("Epoch: {}, Loss: {:.3f}, Accuracy: {:.3f}".format(t, loss.item(), acc))

                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable
                # weights of the model). This is because by default, gradients are
                # accumulated in buffers( i.e, not overwritten) whenever .backward()
                # is called. Checkout docs of torch.autograd.backward for more details.
                optimizer.zero_grad()

                # Backward pass: compute gradient of the loss with respect to model
                # parameters
                loss.backward()

                # Set things to Zero if not p
                for param in model.layer1.parameters():
                    # Layer 1 Weight
                    if len(param.grad.T) == D_in:
                        for i in range(n_neurons):
                            if i == p:
                                pass
                            else:
                                param.grad[i][:] = 0
                    # Layer 1 Bias
                    else:
                        for i in range(n_neurons):
                            if i == p:
                                pass
                            else:
                                param.grad[i] = 0

                for param in model.layer2.parameters():
                    # Layer 2 weights
                    for i in range(n_neurons):
                        if i == p:
                            pass
                        else:
                            param.grad[0][i] = 0

                # Calling the step function on an Optimizer makes an update to its
                # parameters
                optimizer.step()

        self.model = model

        # Layer weights
        layer1_weight = model.layer1.weight.data.numpy().T
        layer1_bias = model.layer1.bias.data.numpy()
        layer2_weight = model.layer2.weight.data.numpy().T
        layer2_bias = 0
        self.model.pctvar_ = pctvar_calc(X, Y, layer1_weight, layer1_bias, layer2_weight)
        self.model.vip_ = garson(layer1_weight, layer2_weight.flatten())
        self.model.coef_ = connectionweight(layer1_weight, layer2_weight.flatten())

        # Not sure about the naming scheme (trying to match PLS)
        self.model.x_loadings_ = layer1_weight
        self.model.x_scores_ = np.matmul(X, self.model.x_loadings_) + layer1_bias
        self.model.x_scores_alt = self.model.x_scores_
        self.model.y_loadings_ = layer2_weight
        self.model.y_scores = np.matmul(self.model.x_scores_alt, self.model.y_loadings_).flatten()
        #y_pred_train = self.model(x).data.numpy().T[0]
        y_pred_train = self.model.y_scores
        #y_pred_train = self.swann(x).data.numpy().T[0]
        #y_pred_train = self.swann(x).data.numpy().T[0]

        self.model.y_loadings_ = self.model.y_loadings_.T
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
        x = X
        # Layer weights
        layer1_weight = self.model.layer1.weight.data.numpy().T
        layer1_bias = self.model.layer1.bias.data.numpy()
        layer2_weight = self.model.layer2.weight.data.numpy().T
        layer2_bias = 0

        self.model.x_scores_ = np.matmul(X, self.model.x_loadings_) + layer1_bias
        self.model.x_scores_alt = self.model.x_scores_
        self.model.y_loadings_ = layer2_weight
        self.model.y_scores = np.matmul(self.model.x_scores_alt, self.model.y_loadings_)

        #self.model.y_scores = np.matmul(self.model.x_scores_alt, self.model.y_loadings_) + layer2_bias
        y_pred_test = self.model.y_scores.flatten()
        self.Y_pred = y_pred_test

        return y_pred_test


def pctvar_calc(X, Y, w1, b1, w2):
    x1 = X
    x2 = np.matmul(x1, w1) + b1

    pctvar = []
    if len(w2) == 1:
        y = np.matmul(x2, w2)
        r2_i = r2_score(Y, y)
        pctvar.append(r2_i)
    else:
        for i in range(len(w2)):
            w2_i = deepcopy(w2)
            w2_i[~i] = 0
            y = np.matmul(x2, w2_i)
            r2_i = r2_score(Y, y)
            pctvar.append(r2_i)

    pct = np.array(pctvar)
    return pct


def garson(A, B):
    """
    Computes Garson's algorithm
    A = matrix of weights of input-hidden layer (rows=input & cols=hidden)
    B = vector of weights of hidden-output layer
    """
    B = np.diag(B)

    # connection weight through the different hidden node
    cw = np.dot(A, B)

    # weight through node (axis=0 is column; sum per input feature)
    cw_h = abs(cw).sum(axis=0)

    # relative contribution of input neuron to outgoing signal of each hidden neuron
    # sum to find relative contribution of input neuron
    rc = np.divide(abs(cw), abs(cw_h))
    rc = rc.sum(axis=1)

    # normalize to 100% for relative importance
    ri = rc / rc.sum()
    return(ri)


def connectionweight(A, B):
    """
    Computes Garson's algorithm
    A = matrix of weights of input-hidden layer (rows=input & cols=hidden)
    B = vector of weights of hidden-output layer
    """
    #B = np.diag(B)

    # connection weight through the different hidden node
    cw = np.dot(A, B)

    return cw
