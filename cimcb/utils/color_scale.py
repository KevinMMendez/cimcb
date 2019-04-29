import numpy as np
from sklearn import preprocessing


def color_scale(x, method="linear", beta=None, alpha=1, beta_method=1):

    # Set-up for non-linear scaling for heatmap color
    if method is "log":
        scale_x = np.log(x)
    elif method is "square":
        scale_x = x ** 2
    elif method is "square root":
        scale_x = np.sqrt(x)
    elif method is "log+1":
        scale_x = np.log(x + 1)
    else:
        scale_x = x

    # Basic Min_Max for heatmap (changing alpha (opaque) rather than colour)... linear from 0 to 1
    # Clean all this up!!!
    scaler = preprocessing.MinMaxScaler(feature_range=(0.1, 1))
    scale_x_final = scaler.fit_transform(scale_x[:, np.newaxis])
    if beta is not None:
        if beta_method == 1:
            scale_i = []
            for i in scale_x_final[:, 0]:
                stat = 1 / beta * np.tanh(beta * (alpha + i))
                scale_i.append(stat)
            scale_x = scale_i
        elif beta_method == 2:
            scale_i = []
            for i in scale_x_final[:, 0]:
                stat = 1 / beta * np.arctan(beta * (alpha + i))
                scale_i.append(stat)
            scale_x = scale_i
        elif beta_method == 3:
            scale_i = []
            for i in scale_x_final[:, 0]:
                stat = 1 + np.tanh(beta * (alpha + i))
                scale_i.append(stat)
            scale_x = scale_i
        else:
            scale_i = []
            for i in scale_x_final[:, 0]:
                stat = np.tanh(beta * (alpha + i))
                scale_i.append(stat)
            scale_x = scale_i
        scale_x = np.array(scale_x)
        scale_x_final = scaler.fit_transform(scale_x[:, np.newaxis])

    return scale_x_final
